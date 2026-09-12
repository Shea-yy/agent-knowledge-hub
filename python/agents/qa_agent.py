"""
问答 Agent — ReAct 模式，自主工具调用

核心变化 (vs 原版硬编码流水线):
    LLM 拿到问题 → 自主决定调哪个工具 → 根据结果决定是否继续调工具 → 生成答案
        ↑ ReAct (Reasoning + Acting) 循环，Agent 自己决策

工具:
  - vector_search: 语义搜索向量库（概念解释、背景知识）
  - cypher_query:  执行图查询（实体关系、多跳推理）
  - entity_lookup: 查询知识图谱中的实体
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from config import settings
from core.aop_interceptors import CYPHER_MAX_ROWS, break_react_loop, require_readonly_cypher


CYPHER_QUERY_TIMEOUT_SECONDS = 5


# ── Data types ──────────────────────────────────────────────

class QueryIntent(str, Enum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str  # "vector" | "graph" | "table" | "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


# ── System prompt ───────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一个企业知识问答助手，可以访问以下工具来检索信息：

**可用工具：**
- `list_tables()`: 列出所有已导入的 Excel/CSV。
- `table_search(table_name, keyword)`: 在表格中搜索关键词。
- `table_info(table_name)`: 查看表格列名和行数。
- `vector_search(query, top_k)`: 语义搜索知识库文档。
- `entity_lookup(name)`: 查询知识图谱中的实体。
- `cypher_query(cypher)`: 执行 Neo4j Cypher 图查询。

**工具选择策略（重要）：**
1. 任何涉及 "XX是什么/谁/代码/邮箱/电话/多少钱" 等具体数据查询 → **先 list_tables，再 table_search**
2. 表格搜不到 → 再试 entity_lookup + vector_search
3. 概念解释/原理说明 → vector_search
4. 实体关系推理 → entity_lookup + cypher_query
5. 同一个工具最多调 2 次，无结果就换工具，不要重复

**知识图谱 Schema：**
- 实体节点统一使用 `:Entity` 标签，实体类型存放于 `type` 属性。
- 合法 type: Person, Organization, Location, Product, Technology, Concept, Event, Time。
- 文本溯源节点使用 `:Chunk` 标签，通过 `(:Chunk)-[:MENTIONS]->(:Entity)` 关联。
- 关系: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on

**Cypher 安全边界：**
- 只能使用 `MATCH ... RETURN ... LIMIT n`，且 `n` 必须在 1 到 50 之间。
- 不可使用 `CALL`、`SHOW`、`PROFILE` 或任何写操作。

**规则：**
- 答案必须基于检索结果，不要编造
- 信息不足时明确告知用户
- 引用来源（如 [来源: xxx]）
"""


# ── ReAct QA Agent ──────────────────────────────────────────

class QAAgent:
    """
    ReAct 问答 Agent

    使用 LangChain/LangGraph create_agent 实现工具调用循环：
      LLM 拿到问题 → 决定调哪个工具 → 拿到工具结果 → 决定是否再调 → 生成最终答案
    """

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        table_store: Any = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.table_store = table_store
        self._agent = self._build_agent()

    # ── public API ───────────────────────────────────────────

    async def answer(self, question: str, thread_id: str | None = None) -> QAResult:
        """
        执行 ReAct 问答，返回结构化结果。

        Args:
            question: 用户问题
            thread_id: 对话线程 ID（同一线程保留对话历史）；未传入时创建独立会话
        """
        config = self._new_turn_config(thread_id)
        result = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        return self._parse_agent_result(question, result)

    async def astream(self, question: str, thread_id: str | None = None):
        """
        流式执行问答，逐事件返回中间结果。

        用法:
            async for event in qa_agent.astream("问题"):
                # event 包含当前步骤的 messages
                yield event
        """
        config = self._new_turn_config(thread_id)
        async for event in self._agent.astream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="values",
        ):
            yield event

    @staticmethod
    def _new_turn_config(thread_id: str | None) -> dict[str, Any]:
        """为一次用户提问创建配置：记忆按会话、熔断按轮次隔离。"""
        session_id = thread_id.strip() if isinstance(thread_id, str) else ""
        return {
            "configurable": {
                "thread_id": session_id or uuid4().hex,
                "turn_id": uuid4().hex,
            },
            "recursion_limit": 10,
        }

    # ── agent construction ───────────────────────────────────

    def _build_agent(self):
        """构建 ReAct agent，工具通过闭包捕获 vector_store / knowledge_graph / table_store"""
        vector_store = self.vector_store
        knowledge_graph = self.knowledge_graph
        table_store = self.table_store

        @tool
        @break_react_loop(2)
        async def vector_search(query: str, top_k: int = 5, config: RunnableConfig = None) -> str:
            """
            语义搜索知识库文档。用于概念解释、原理说明、背景知识检索。
            返回相关文档片段及其内容。
            不适合关系查询（如"A 和 B 什么关系"），那种问题请用 cypher_query 或 entity_lookup。
            """
            if not vector_store:
                return json.dumps({"error": "向量库未连接", "contexts": []}, ensure_ascii=False)

            results = await vector_store.search(query, top_k=top_k)
            if not results:
                return json.dumps({"message": "未找到相关文档", "contexts": []}, ensure_ascii=False)

            formatted = []
            for i, (doc, score) in enumerate(results):
                formatted.append({
                    "rank": i + 1,
                    "score": round(score, 4),
                    "content": doc.get("content", "")[:500],
                    "source": doc.get("source", "unknown"),
                    "metadata": doc.get("metadata", {}),
                })
            return json.dumps({"contexts": formatted}, ensure_ascii=False, indent=2)

        @tool
        @require_readonly_cypher
        async def cypher_query(cypher: str) -> str:
            """
            执行 Neo4j Cypher 查询。用于实体关系、多跳推理、路径查询。

            知识图谱 Schema:
            - 实体节点标签统一为 :Entity，类型使用 n.type 属性过滤
            - 关系类型: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on

            示例:
              MATCH (p:Entity {name:'张三'})-[r]->(n:Entity) RETURN p, r, n LIMIT 20
              MATCH path = shortestPath((a:Entity {name:'张三'})-[*..3]-(b:Entity {name:'李四'})) RETURN path LIMIT 1
            """
            if not knowledge_graph:
                return json.dumps({"error": "知识图谱未连接", "contexts": []}, ensure_ascii=False)

            try:
                records = await asyncio.wait_for(
                    knowledge_graph.execute_cypher(cypher),
                    timeout=CYPHER_QUERY_TIMEOUT_SECONDS,
                )
                # 即便服务端实现或配置意外忽略 LIMIT，也不将无界记录喂回模型。
                formatted = [dict(record) for record in records[:CYPHER_MAX_ROWS]]
                contexts = [
                    {
                        "content": json.dumps(record, ensure_ascii=False, default=str)[:500],
                        "source": "knowledge_graph",
                        "score": 0.8,
                        "retrieval_type": "graph",
                    }
                    for record in formatted
                ]
                return json.dumps(
                    {"records": formatted, "contexts": contexts},
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            except TimeoutError:
                return json.dumps({
                    "error": f"图查询超时（超过 {CYPHER_QUERY_TIMEOUT_SECONDS} 秒）",
                    "contexts": [],
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e), "contexts": []}, ensure_ascii=False)

        @tool
        @break_react_loop(2)
        async def entity_lookup(entity_name: str, config: RunnableConfig = None) -> str:
            """
            查询知识图谱中的实体及其关联关系。
            先查实体了解基本情况，再决定是否需要 cypher_query 做深层查询。
            适用于 "XXX 是什么"、"XXX 和谁有关" 等问题。
            """
            if not knowledge_graph:
                return json.dumps({"error": "知识图谱未连接", "contexts": []}, ensure_ascii=False)

            neighbors: list[Any] = []
            mentions: list[Any] = []
            errors: list[str] = []

            # 关系和溯源片段来自两次独立查询。一方失败不能丢弃另一方已经
            # 获得的结果，也不能把“查询不完整”误报为“未找到实体”。
            try:
                neighbors = await knowledge_graph.get_neighbors(entity_name, hops=1)
            except Exception as exc:
                errors.append(f"获取实体关系失败: {exc}")

            try:
                mentions = await knowledge_graph.get_entity_mentions(entity_name, limit=3)
            except Exception as exc:
                errors.append(f"获取实体溯源失败: {exc}")

            if not neighbors and not mentions:
                if errors:
                    return json.dumps({
                        "error": "；".join(errors),
                        "entity": entity_name,
                        "contexts": [],
                    }, ensure_ascii=False)
                return json.dumps({
                    "message": f"未找到实体 '{entity_name}'",
                    "entity": entity_name,
                    "contexts": [],
                }, ensure_ascii=False)

            contexts = [
                {
                    "content": (mention.get("content") or "")[:300],
                    "source": mention.get("source") or mention.get("chunk_id") or "knowledge_graph",
                    "score": 0.85,
                    "retrieval_type": "graph",
                    "metadata": {
                        "entity": entity_name,
                        "chunk_id": mention.get("chunk_id"),
                        "page": mention.get("page"),
                    },
                }
                for mention in mentions if mention.get("content")
            ]
            # 图谱中已有的关系也是可审计的查询结果。若没有原文溯源片段，直接将
            # 关系写入同一 contexts 契约，而不是由结果解析器猜测旧版返回格式。
            if not contexts:
                contexts = [
                    {
                        "content": json.dumps(dict(neighbor), ensure_ascii=False, default=str),
                        "source": "knowledge_graph",
                        "score": 0.8,
                        "retrieval_type": "graph",
                        "metadata": {"entity": entity_name},
                    }
                    for neighbor in neighbors
                ]
            response = {
                "entity": entity_name,
                "relationships": [dict(n) for n in neighbors],
                "contexts": contexts,
            }
            if errors:
                response["warnings"] = errors
            return json.dumps(response, ensure_ascii=False, indent=2, default=str)

        @tool
        async def list_tables(dummy: str = "") -> str:
            """列出所有已导入的 Excel/CSV 表格文件名。调用后根据文件名选择要查哪张表。"""
            if not table_store:
                return json.dumps({"error": "表格存储未初始化", "contexts": []}, ensure_ascii=False)
            tables = table_store.list_tables()
            if not tables:
                return json.dumps({"message": "暂无已导入的表格", "tables": [], "contexts": []}, ensure_ascii=False)
            return json.dumps({"tables": tables, "contexts": []}, ensure_ascii=False, indent=2)

        @tool
        @break_react_loop(2)
        async def table_search(table_name: str, keyword: str, config: RunnableConfig = None) -> str:
            """
            在指定表格的所有列中搜索关键词，返回匹配的行。

            table_name: 表格文件名（先用 list_tables 查看有哪些表）
            keyword: 要搜索的关键词，如 "平安银行"、"000001" 等
            """
            if not table_store:
                return json.dumps({"error": "表格存储未初始化", "contexts": []}, ensure_ascii=False)
            result = table_store.query(table_name, "search", keyword=keyword)
            # “0 行”结果只用于指导下一次检索，不能作为支持答案的高置信度证据。
            if "找到 0 行" in result:
                return json.dumps({"message": result, "contexts": []}, ensure_ascii=False)
            return json.dumps({
                "contexts": [{
                    "content": result[:2000],
                    "source": table_name,
                    "score": 0.9,
                    "retrieval_type": "table",
                    "metadata": {"table_name": table_name, "keyword": keyword},
                }],
            }, ensure_ascii=False)

        @tool
        async def table_info(table_name: str) -> str:
            """
            查看表格的列名、行数、数据类型。先看有哪些列，才知道怎么搜。

            table_name: 表格文件名
            """
            if not table_store:
                return json.dumps({"error": "表格存储未初始化", "contexts": []}, ensure_ascii=False)
            result = table_store.query(table_name, "columns")
            return json.dumps({
                # 表结构只用于让 Agent 选择下一步查询，不是回答问题的证据。
                "message": result,
                "contexts": [],
                "metadata": {"table_name": table_name},
            }, ensure_ascii=False)

        tools = [vector_search, cypher_query, entity_lookup, list_tables, table_search, table_info]

        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
        )

    # ── result parsing ───────────────────────────────────────

    def _parse_agent_result(self, question: str, result: dict) -> QAResult:
        """
        从 ReAct agent 的返回 state 中提取结构化 QAResult。

        result["messages"] 结构示例:
          [HumanMessage, AIMessage(tool_calls=[...]), ToolMessage(...), AIMessage("最终答案")]
        """
        messages: list = result.get("messages", [])

        # 先隔离当前轮；找不到当前问题锚点时返回空列表，绝不能回退历史。
        turn_messages = self._current_turn_messages(messages, question)

        # 提取本轮最终答案（最后一条非工具调用 AI 消息的内容）。
        answer = ""
        for msg in reversed(turn_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break

        # 只返回本轮的检索证据；同一 thread 的历史工具结果不应重复出现在新回答中。
        contexts = self._extract_contexts(turn_messages)

        # 意图只由用户问题判断，不能被工具输出或最终答案中的关键词污染。
        intent = self._infer_intent(question)

        # 仅暴露可审计的工具轨迹，不把模型文本伪装成可解释的内部思维链。
        reasoning_steps = self._extract_reasoning_steps(turn_messages)

        # 置信度估算（基于检索结果数量和质量）
        confidence = self._calc_confidence(contexts)

        return QAResult(
            question=question,
            answer=answer,
            contexts=contexts,
            intent=intent,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
        )

    @staticmethod
    def _current_turn_messages(messages: list, question: str) -> list:
        """截取最后一个匹配用户问题后的消息，隔离当前回答的证据来源。"""
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage) and message.content == question:
                return messages[index:]
        return []

    @staticmethod
    def _extract_contexts(messages: list) -> list[RetrievedContext]:
        """只从统一的 ToolMessage ``contexts`` 字段提取可引用证据。"""
        contexts: list[RetrievedContext] = []
        seen: set[tuple[str, str]] = set()

        def append_context(item: dict[str, Any], default_type: str, default_source: str) -> None:
            content = str(item.get("content", "")).strip()
            if not content:
                return
            source = str(item.get("source") or default_source)
            truncated_content = content[:500]
            # API 最终只返回截断后的片段，故须以该表示去重，避免两条
            # 仅在第 500 字符后不同的上下文展示为完全相同的引用。
            key = (source, truncated_content)
            if key in seen:
                return
            seen.add(key)
            try:
                score = float(item.get("score", 0.7))
            except (TypeError, ValueError):
                score = 0.7
            contexts.append(RetrievedContext(
                content=truncated_content,
                source=source,
                score=max(0.0, min(score, 1.0)),
                retrieval_type=str(item.get("retrieval_type") or default_type),
                metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            ))

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue

            tool_name = getattr(msg, "name", "")
            if "vector" in tool_name:
                retrieval_type = "vector"
            elif "cypher" in tool_name or "entity" in tool_name:
                retrieval_type = "graph"
            else:
                retrieval_type = "hybrid"

            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(data, dict):
                    for item in data.get("contexts", []):
                        if isinstance(item, dict):
                            append_context(item, retrieval_type, tool_name)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        return contexts

    @staticmethod
    def _infer_intent(question: str) -> QueryIntent:
        """从用户问题推断查询意图。"""
        intent_keywords = {
            QueryIntent.COMPARATIVE: ["对比", "区别", "比较", "不同"],
            QueryIntent.PROCEDURAL: ["怎么", "如何", "步骤", "流程"],
            QueryIntent.EXPLORATORY: ["有哪些", "概述", "总结", "介绍"],
            QueryIntent.ANALYTICAL: ["为什么", "分析", "原因", "影响"],
        }
        for intent, keywords in intent_keywords.items():
            if any(keyword in question for keyword in keywords):
                return intent
        return QueryIntent.FACTOID

    @staticmethod
    def _extract_reasoning_steps(messages: list) -> list[str]:
        """提取可审计的工具调用轨迹，不暴露模型自由文本。"""
        steps: list[str] = []
        tool_count = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_count += 1
                        steps.append(f"调用工具 #{tool_count}: {tc.get('name', 'unknown')}")
            elif isinstance(msg, ToolMessage):
                steps.append(f"工具结果返回 ({getattr(msg, 'name', 'unknown')})")
        steps.append("生成最终答案")
        return steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        """基于已引用证据估算分数，不把它当作模型正确率。"""
        if not contexts:
            return 0.0
        avg_score = sum(max(0.0, min(c.score, 1.0)) for c in contexts) / len(contexts)
        retrieval_types = {context.retrieval_type for context in contexts}
        diversity_bonus = 0.05 if len(retrieval_types) > 1 else 0.0
        return min(avg_score + diversity_bonus, 1.0)
