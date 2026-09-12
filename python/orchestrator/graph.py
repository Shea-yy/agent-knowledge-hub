"""
LangGraph 编排引擎 — 入库/更新 StateGraph + QA ReAct Agent

升级要点 (vs 原版 langgraph>=0.3.0):
  1. 入库、更新工作流使用 StateGraph(TypedDict) — 类型安全
  2. 入库、更新工作流使用 MemorySaver checkpoint — 支持暂停/恢复/回放
  3. QA 直接使用 create_agent 生成的 ReAct Graph，避免重复嵌套 checkpoint
  4. astream() streaming — 实时输出工作流执行进度

三条流水线:
  1. 文档入库: DocParser → KnowledgeExtract → (VectorStore ∥ KnowledgeGraph)
  2. 智能问答: ReAct Agent → 自主调用工具 → 综合回答
  3. 增量更新: CDC Event → UpdateAgent → (失败 → 重试) → 完成
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import DocumentChange, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent
from orchestrator.state import IngestState, UpdateState
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


# ── 全局 checkpoint 实例（开发环境用内存，生产切 SqliteSaver） ──
_checkpointer = MemorySaver()


def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None = None,
    knowledge_graph: KnowledgeGraphService | None = None,
    table_store: Any = None,
) -> dict[str, Any]:  # key: "ingest" | "qa" | "update"
    """
    构建三条流水线，返回 dict 供 API 层调用。

    ingest / update 是编译后的 StateGraph；qa 是 QAAgent，内部持有
    create_agent 生成且带 checkpoint 的 ReAct Graph。QA 不再被外层
    StateGraph 包装，以确保 API 的 thread_id 传到实际对话状态。

    Args:
        vector_store: 向量库服务实例（可选，未提供时对应节点跳过）
        knowledge_graph: 知识图谱服务实例（可选）
    """
    doc_parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()
    qa_agent = QAAgent(vector_store=vector_store, knowledge_graph=knowledge_graph, table_store=table_store)
    update_agent = KnowledgeUpdateAgent(
        doc_parser=doc_parser,
        knowledge_extractor=extractor,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    return {
        "ingest": _build_ingest_graph(doc_parser, extractor, vector_store, knowledge_graph),
        # create_agent 本身已经是一个带循环和 checkpoint 的 LangGraph。
        # 不再额外包一层 StateGraph，避免 thread_id 丢失和双重状态持久化。
        "qa": qa_agent,
        "update": _build_update_graph(update_agent),
    }


# ═══════════════════════════════════════════════════════════════
# 入库流水线
# ═══════════════════════════════════════════════════════════════

def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
) -> StateGraph:
    """
    文档入库工作流:

      parse → extract → store_vectors  (并行)
                      → store_graph    (并行)
    """

    async def parse_documents(state: IngestState) -> dict[str, Any]:
        file_paths = state.get("file_paths", [])
        chunks = await doc_parser.parse_batch(file_paths)
        return {"chunks": chunks}

    async def extract_knowledge(state: IngestState) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        # 表格数据跳过知识抽取（结构化查询走 TableStore）
        non_table = [c for c in chunks if getattr(c, "doc_type", None) not in ("table",)]
        if not non_table:
            return {"extractions": [], "extraction_failures": []}
        extractions = await extractor.extract(non_table)
        failures = [
            {"chunk_id": item.source_chunk_id, "error": item.error}
            for item in extractions
            if getattr(item, "status", "success") == "failed"
        ]
        return {"extractions": extractions, "extraction_failures": failures}

    async def store_vectors(state: IngestState) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        count = 0
        if vector_store and chunks:
            count = await vector_store.add_chunks(chunks)
        return {"vectors_stored": count}

    async def store_graph(state: IngestState) -> dict[str, Any]:
        extractions = state.get("extractions", [])
        # chunk_id → 原文：MENTIONS 血缘边需要把原文落进 Chunk 节点，
        # 供问答溯源时直接带回原文片段
        chunks_by_id = {c.chunk_id: c for c in state.get("chunks", [])}
        entity_count = 0
        event_count = 0
        if knowledge_graph:
            for ext in extractions:
                if getattr(ext, "status", "success") == "failed":
                    continue
                chunk = chunks_by_id.get(ext.source_chunk_id)
                source = chunk.metadata.get("source", "") if chunk else ""
                page = chunk.metadata.get("page") if chunk else None
                content = chunk.content if chunk else ""
                doc_id = chunk.doc_id if chunk else ""
                revision_id = chunk.metadata.get("revision_id", "") if chunk else ""
                for ent in ext.entities:
                    await knowledge_graph.upsert_entity(ent, source=source)
                    entity_count += 1
                # 血缘边: (Chunk)-[:MENTIONS]->(Entity)，每 chunk 一条批量查询
                if ext.entities:
                    await knowledge_graph.upsert_chunk_mentions(
                        ext.source_chunk_id,
                        [e.name for e in ext.entities],
                        content=content,
                        source=source,
                        page=page,
                        doc_id=doc_id,
                        revision_id=revision_id,
                    )
                for rel in ext.relations:
                    await knowledge_graph.add_relation(
                        rel,
                        source=source,
                        source_chunk_id=ext.source_chunk_id,
                        doc_id=doc_id,
                        revision_id=revision_id,
                    )
                for event in ext.events:
                    await knowledge_graph.upsert_event(
                        event,
                        ext.source_chunk_id,
                        content=content,
                        source=source,
                        page=page,
                        doc_id=doc_id,
                        revision_id=revision_id,
                    )
                    event_count += 1
        return {"entities_stored": entity_count, "events_stored": event_count}

    # -- 构建图 --
    graph = StateGraph(IngestState)

    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("store_graph", store_graph)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    # 并行分发: extract → store_vectors + store_graph 同时执行
    graph.add_edge("extract", "store_vectors")
    graph.add_edge("extract", "store_graph")
    graph.add_edge("store_vectors", END)
    graph.add_edge("store_graph", END)

    return graph.compile(checkpointer=_checkpointer)


# ═══════════════════════════════════════════════════════════════
# 增量更新流水线
# ═══════════════════════════════════════════════════════════════

def _build_update_graph(update_agent: KnowledgeUpdateAgent) -> StateGraph:
    """
    更新工作流:

      process → should_continue?
                 ├─ "retry" → retry → END
                 └─ "done"  → END

    条件路由：有失败项时自动重试一次。
    """

    async def process_updates(state: UpdateState) -> dict[str, Any]:
        changes = state.get("changes", [])
        results = await update_agent.process_batch(changes)
        return {"results": results}

    def should_continue(state: UpdateState) -> Literal["retry", "done"]:
        results = state.get("results", [])
        failed = [r for r in results if not r.success]
        return "retry" if failed else "done"

    async def retry_failed(state: UpdateState) -> dict[str, Any]:
        results = state.get("results", [])
        failed_changes = [r.change for r in results if not r.success]
        retried = await update_agent.process_batch(failed_changes)
        # 合并成功的原始结果 + 重试结果
        all_results = [r for r in results if r.success] + retried
        return {"results": all_results}

    graph = StateGraph(UpdateState)

    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)

    graph.set_entry_point("process")
    graph.add_conditional_edges(
        "process",
        should_continue,
        {"retry": "retry", "done": END},
    )
    graph.add_edge("retry", END)

    return graph.compile(checkpointer=_checkpointer)
