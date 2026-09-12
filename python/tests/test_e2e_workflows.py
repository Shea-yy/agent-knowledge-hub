"""确定性端到端测试：跨 Agent、编排和评测契约，但不依赖外部服务或真实 LLM。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import (
    Entity,
    EventArgument,
    ExtractionResult,
    KnowledgeEvent,
    Relation,
)
from agents.qa_agent import QAAgent
from core.qa_evaluation import evaluate_agent, load_gold_cases
from orchestrator.graph import _build_ingest_graph


GOLD_CASES_PATH = Path(__file__).with_name("qa_gold_cases.json")


class ScriptedReActGraph:
    """替代网络模型：按问题返回固定的 ReAct 消息轨迹。"""

    def __init__(self, responses: dict[str, list]) -> None:
        self.responses = responses
        self.configs: list[dict] = []

    async def ainvoke(self, inputs: dict, config: dict) -> dict:
        question = inputs["messages"][0].content
        self.configs.append(config)
        return {"messages": self.responses[question]}


def _tool_turn(question: str, tool_name: str, tool_content: str, answer: str) -> list:
    return [
        HumanMessage(content=question),
        AIMessage(content="", tool_calls=[{
            "name": tool_name,
            "args": {},
            "id": f"call-{tool_name}",
        }]),
        ToolMessage(content=tool_content, name=tool_name, tool_call_id=f"call-{tool_name}"),
        AIMessage(content=answer),
    ]


@pytest.mark.asyncio
async def test_ingest_workflow_fans_out_to_vector_and_graph_with_provenance():
    chunk = DocumentChunk(
        content="张三在腾讯工作，并负责报销系统。",
        doc_id="deterministic-doc",
        chunk_index=0,
        doc_type=DocType.TEXT,
        metadata={"source": "/fixtures/employee.md", "page": 1, "revision_id": "rev-e2e"},
    )
    extraction = ExtractionResult(
        entities=[
            Entity(name="张三", type="Person"),
            Entity(name="腾讯", type="Organization"),
        ],
        relations=[Relation(head="张三", relation="works_at", tail="腾讯")],
        events=[KnowledgeEvent(
            trigger="负责",
            type="职责分配",
            arguments=[EventArgument(role="负责人", entity="张三")],
        )],
        source_chunk_id=chunk.chunk_id,
    )
    parser = MagicMock()
    parser.parse_batch = AsyncMock(return_value=[chunk])
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=[extraction])
    vector_store = MagicMock()
    vector_store.add_chunks = AsyncMock(return_value=1)
    graph_store = MagicMock()
    graph_store.upsert_entity = AsyncMock()
    graph_store.upsert_chunk_mentions = AsyncMock()
    graph_store.add_relation = AsyncMock()
    graph_store.upsert_event = AsyncMock()

    workflow = _build_ingest_graph(parser, extractor, vector_store, graph_store)
    result = await workflow.ainvoke(
        {"file_paths": ["/fixtures/employee.md"]},
        config={"configurable": {"thread_id": "deterministic-e2e-ingest"}},
    )

    assert result["vectors_stored"] == 1
    assert result["entities_stored"] == 2
    assert result["events_stored"] == 1
    assert result["extraction_failures"] == []
    vector_store.add_chunks.assert_awaited_once_with([chunk])
    graph_store.upsert_chunk_mentions.assert_awaited_once_with(
        chunk.chunk_id,
        ["张三", "腾讯"],
        content=chunk.content,
        source="/fixtures/employee.md",
        page=1,
        doc_id="deterministic-doc",
        revision_id="rev-e2e",
    )
    graph_store.add_relation.assert_awaited_once_with(
        extraction.relations[0],
        source="/fixtures/employee.md",
        source_chunk_id=chunk.chunk_id,
        doc_id="deterministic-doc",
        revision_id="rev-e2e",
    )


@pytest.mark.asyncio
async def test_qa_answer_entrypoint_meets_all_gold_contracts_deterministically():
    cases = load_gold_cases(GOLD_CASES_PATH)
    responses = {
        "张三在哪里工作？": _tool_turn(
            "张三在哪里工作？",
            "table_search",
            json.dumps({"contexts": [{
                "content": "姓名: 张三，所属公司: 腾讯",
                "source": "员工名录.csv",
                "score": 0.9,
                "retrieval_type": "table",
            }]}, ensure_ascii=False),
            "张三在腾讯工作。[来源: 员工名录.csv]",
        ),
        "张三和腾讯有什么关系？": _tool_turn(
            "张三和腾讯有什么关系？",
            "entity_lookup",
            json.dumps({"contexts": [{
                "content": "张三 works_at 腾讯",
                "source": "knowledge_graph",
                "score": 0.85,
                "retrieval_type": "graph",
            }]}, ensure_ascii=False),
            "张三是腾讯的员工。[来源: knowledge_graph]",
        ),
        "如何提交报销申请？": _tool_turn(
            "如何提交报销申请？",
            "vector_search",
            json.dumps({"contexts": [{
                "content": "提交报销单后进入审批流程。",
                "source": "expense-policy.md",
                "score": 0.88,
            }]}, ensure_ascii=False),
            "提交报销单后等待审批。[来源: expense-policy.md]",
        ),
    }
    scripted_graph = ScriptedReActGraph(responses)
    agent = object.__new__(QAAgent)
    agent._agent = scripted_graph

    summary = await evaluate_agent(agent, cases)

    assert summary.total == 3
    assert summary.passed == 3
    assert summary.pass_rate == 1.0
    assert summary.average_source_recall == 1.0
    assert all(config["configurable"]["thread_id"].startswith("gold-eval-") for config in scripted_graph.configs)
    assert len({config["configurable"]["turn_id"] for config in scripted_graph.configs}) == 3
