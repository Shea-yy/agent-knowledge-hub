"""
知识抽取 Agent — 从文档块中提取实体、关系、事件，构建知识图谱三元组

核心能力:
  1. 命名实体识别 (NER)
  2. 关系抽取 (RE) — 带 evidence 溯源
  3. 事件抽取 — ACE 论元结构（带角色，保留方向性）
  4. 三元组生成 → 写入 Neo4j

实现方式:
  with_structured_output + Pydantic Schema（现代结构化输出）。
  Schema 设计原则:
    - Field(description=...) 是"隐形说明书"——编译进 JSON Schema 后
      由 LLM 读取，主客体倒置的幻觉显著减少
    - 核心 Schema 与系统元数据物理隔离（继承拆分），
      避免 source_chunk_id 污染 LLM 的生成空间
"""

from __future__ import annotations

import asyncio
from typing import Literal

import httpx
import openai
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from agents.doc_parser_agent import DocumentChunk
from config import settings

log = logger.bind(module="knowledge_extract")

# ── 图谱 Schema 契约：Literal 枚举 ──
# 编译进 JSON Schema 后成为 enum，OpenAI 服务端强制遵守（不可输出枚举外值）。
# 附带安全收益: relation 会拼进 Cypher 查询，白名单枚举堵死注入面。
EntityType = Literal[
    "Person", "Organization", "Location", "Product",
    "Technology", "Concept", "Event", "Time",
]
RelationType = Literal[
    "belongs_to", "works_at", "located_in", "developed_by",
    "related_to", "part_of", "uses", "depends_on",
]

# 网络瞬态异常白名单：这些异常重试有意义（限流/超时/网关抖动）
# ValidationError / BadRequestError 是确定性错误，不重试
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    httpx.TransportError,
)


def _log_retry(retry_state) -> None:
    """tenacity 重试回调：每次重试前记录日志，保证可观测性"""
    log.warning(
        "LLM 调用失败 ({}), 第 {} 次重试, 等待 {:.0f}s",
        retry_state.outcome.exception(),
        retry_state.attempt_number,
        retry_state.next_action.sleep if retry_state.next_action else 0,
    )

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的知识抽取引擎。给定一段文本，请提取其中的实体、关系、事件。

实体类型限定: Person, Organization, Location, Product, Technology, Concept, Event, Time
关系类型限定: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on

关系方向性要求（重要）:
  "张伟开发了文档解析模块" → head=文档解析模块, relation=developed_by, tail=张伟
  "公司使用了 LangGraph"   → head=公司, relation=uses, tail=LangGraph
  主语在前还是宾语在前，取决于关系类型的语义方向，请仔细判断。

evidence 要求: 每个关系必须附带原文片段作为证据，不允许编造原文。
"""


# ═══════════════════════════════════════════════════════════════
# Pydantic Schema — description 是 LLM 的隐形说明书
# ═══════════════════════════════════════════════════════════════

class _StrippedModel(BaseModel):
    """
    所有字符串字段自动 strip 的基类。

    动机: LLM 常输出首尾带空格的实体名（如 "OpenAI "），
    若不做归一化，"OpenAI " in global_entities 判定为 False，
    合法关系被 _filter_dangling 当成悬空关系误杀——一个空格丢一条关系。
    """

    from pydantic import field_validator

    @field_validator("*", mode="before")
    @classmethod
    def _strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v


class Entity(_StrippedModel):
    name: str = Field(description="实体名称，如人名、组织名、产品名")
    type: EntityType = Field(
        description="实体类型（Schema 枚举强制，从限定集合中选择）"
    )
    description: str = Field(default="", description="基于原文对实体的简短描述")

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


class Relation(_StrippedModel):
    head: str = Field(description="头实体名称，须与 entities 中的 name 完全一致")
    relation: RelationType = Field(
        description="关系类型（Schema 枚举强制，从限定集合中选择）"
    )
    tail: str = Field(description="尾实体名称，须与 entities 中的 name 完全一致")
    confidence: float = Field(default=0.0, description="0-1 之间的置信度")
    evidence: str = Field(
        default="",
        description="支持该关系的原文片段（溯源证据）。逐字引用原文，不允许改写或编造。",
    )


class EventArgument(_StrippedModel):
    """事件论元 — 带角色，保留方向性（如 收购方/被收购方 可区分）"""
    role: str = Field(
        description="论元在事件中的角色，如: agent(施事方), patient(受事方), 收购方, 被收购方, 时间, 地点"
    )
    entity: str = Field(description="论元对应的实体名称")


class KnowledgeEvent(_StrippedModel):
    trigger: str = Field(description="事件触发词，如: 收购, 发布, 任命, 融资")
    type: str = Field(description="事件类型，如: 并购, 产品发布, 人事变动, 融资")
    arguments: list[EventArgument] = Field(
        default_factory=list,
        description="事件论元列表。每个论元必须标注 role 角色，以保留事件方向性。",
    )


class ExtractionCore(BaseModel):
    """
    发给 LLM 的核心 Schema（进入 with_structured_output 的模型）。

    设计原则: 只包含需要 LLM 生成的字段。
    系统元数据（source_chunk_id）由子类物理隔离，不进 JSON Schema。
    """
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[KnowledgeEvent] = Field(default_factory=list)


class ExtractionResult(ExtractionCore):
    """内部完整模型 = 核心 Schema + 代码附加的溯源 ID 与执行状态。"""
    source_chunk_id: str = ""
    status: Literal["success", "failed"] = "success"
    error: str = ""


class KnowledgeExtractAgent:
    """
    知识抽取 Agent

    工作流:
      receive_chunks → 并发抽取（asyncio.gather）→ 悬空过滤 → 按 chunk 输出（血缘优先）
    """

    BATCH_SIZE = 5  # 并发窗口大小：每批 gather 并发 N 个 LLM 请求

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        # 绑定核心 Schema（不含 source_chunk_id，避免 LLM 编造系统元数据）
        self.structured_llm = self.llm.with_structured_output(
            ExtractionCore, method="json_schema"
        )

    # ── public API ───────────────────────────────────────────

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """
        从一组文档块中抽取知识（批内并发，批间顺序推进）。

        清洗策略（血缘优先）:
          - 只做悬空引用过滤，不做跨 chunk 去重
          - 实体/关系保留在各自 chunk 的结果中，供下游构建
            (Chunk)-[:MENTIONS]->(Entity) 血缘边
          - 节点唯一性由 Neo4j MERGE 保证，不在 Python 层提前丢弃信息
          - 顺序上"先洗脏数据"：若先去重再过滤，悬空论元被过滤后
            可能产生新的重复项（去重防线被时序击穿）
        """
        results: list[ExtractionResult] = []
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            # asyncio.gather: 网络 IO 并发打满，批内总耗时 ≈ 单次请求耗时
            batch_results = await asyncio.gather(
                *[self._extract_from_chunk(c) for c in batch]
            )
            results.extend(batch_results)

        # 先清洗（悬空过滤），后不做去重——血缘优先
        cleaned = self._filter_dangling(results)

        unique_entities = len({e.name for r in cleaned for e in r.entities})
        unique_relations = len({
            (rel.head, rel.relation, rel.tail)
            for r in cleaned for rel in r.relations
        })
        failed_count = sum(result.status == "failed" for result in cleaned)
        log.info(
            "抽取完成: {} chunks → {} 唯一实体, {} 唯一关系, {} 失败",
            len(chunks), unique_entities, unique_relations, failed_count,
        )
        return cleaned

    # ── core extraction ──────────────────────────────────────

    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id)

    async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
        """
        抽取入口：网络瞬态错误由 tenacity 指数退避重试，
        重试耗尽后降级为空结果（记录 error，不中断整批）。
        """
        try:
            return await self._extract_with_retry(text, source_id)
        except Exception as e:
            log.error("结构化抽取彻底失败 (重试已耗尽): {} → {}", source_id, e)
            return ExtractionResult(
                source_chunk_id=source_id,
                status="failed",
                error=f"{type(e).__name__}: {e}",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),  # 2s → 4s → 8s
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _extract_with_retry(self, text: str, source_id: str) -> ExtractionResult:
        """核心调用：仅网络瞬态异常触发重试，确定性错误直接抛给上层兜底"""
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
        ]
        core: ExtractionCore = await self.structured_llm.ainvoke(messages)
        # 核心 Schema → 完整模型：source_chunk_id 由代码附加
        return ExtractionResult(**core.model_dump(), source_chunk_id=source_id)

    # ── referential integrity ────────────────────────────────

    @staticmethod
    def _filter_dangling(results: list[ExtractionResult]) -> list[ExtractionResult]:
        """
        全局引用完整性校验：过滤引用未声明实体的悬空引用（关系 + 事件论元）。

        为什么不用 model_validator 抛异常:
          1. 抛异常会让 except 兜底吞掉整块抽取（含有效实体），一条悬空关系
             连累全部数据 —— 过滤 + 告警才是正确姿势
          2. 实体分散在不同 chunk 声明，必须用跨 chunk 的全局实体集合校验，
             逐 chunk 校验会误杀合法跨块引用
        """
        global_entities: set[str] = set()
        for r in results:
            global_entities.update(e.name for e in r.entities)

        dropped_relations = 0
        dropped_arguments = 0
        cleaned: list[ExtractionResult] = []
        for r in results:
            # 1. 关系: 头尾实体都必须已声明
            valid_relations = [
                rel for rel in r.relations
                if rel.head in global_entities and rel.tail in global_entities
            ]
            dropped_relations += len(r.relations) - len(valid_relations)

            # 2. 事件论元: 引用未声明实体的论元同步过滤
            valid_events = []
            for ev in r.events:
                valid_args = [
                    arg for arg in ev.arguments if arg.entity in global_entities
                ]
                dropped_arguments += len(ev.arguments) - len(valid_args)
                valid_events.append(ev.model_copy(update={"arguments": valid_args}))

            cleaned.append(r.model_copy(update={
                "relations": valid_relations,
                "events": valid_events,
            }))

        if dropped_relations or dropped_arguments:
            log.warning(
                "过滤悬空引用: 关系 {} 条, 事件论元 {} 个",
                dropped_relations, dropped_arguments,
            )
        return cleaned
