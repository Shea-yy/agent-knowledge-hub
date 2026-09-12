"""
LangGraph 工作流 State 定义 — 使用 TypedDict 提供类型安全

LangGraph 1.x 推荐使用 TypedDict 定义 state schema:
  - 每个 key 标注类型，IDE 有自动补全
  - Annotated[list, add_messages] 实现消息追加而非覆盖
  - total=False 允许增量填充 state（各节点只返回部分字段）
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class WorkflowType(str, Enum):
    INGEST = "ingest"
    QA = "qa"
    UPDATE = "update"


# ── Ingest State ────────────────────────────────────────────

class IngestState(TypedDict, total=False):
    """文档入库流程 state

    各节点按需填充：
      parse        → chunks
      extract      → extractions, extraction_failures
      store_vectors → vectors_stored
      store_graph   → entities_stored, events_stored
    """
    file_paths: list[str]
    chunks: list[Any]            # list[DocumentChunk] — 避免循环导入，用 Any
    extractions: list[Any]        # list[ExtractionResult]
    extraction_failures: list[dict[str, str]]
    vectors_stored: int
    entities_stored: int
    events_stored: int
    messages: Annotated[list, add_messages]


# ── QA State ────────────────────────────────────────────────

class QAState(TypedDict, total=False):
    """问答流程 state

    ReAct agent 运行时逐步填充：
      question → agent 自主调用 tool → result
    """
    question: str
    result: Any                  # QAResult | None
    messages: Annotated[list, add_messages]


# ── Update State ────────────────────────────────────────────

class UpdateState(TypedDict, total=False):
    """增量更新流程 state

    process → 填充 results
    失败时 retry → 合并 results
    """
    changes: list[Any]           # list[DocumentChange]
    results: list[Any]           # list[UpdateResult]
    messages: Annotated[list, add_messages]
