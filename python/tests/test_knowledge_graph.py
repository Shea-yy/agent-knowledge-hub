"""知识图谱写入契约测试，不需要运行 Neo4j。"""

from __future__ import annotations

from unittest.mock import AsyncMock

from agents.knowledge_extract_agent import EventArgument, KnowledgeEvent
from services.knowledge_graph import KnowledgeGraphService


class _RecordingSession:
    def __init__(self, calls: list[tuple[str, dict]]):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def run(self, cypher: str, params: dict):
        self.calls.append((cypher, params))


class _RecordingDriver:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def session(self):
        return _RecordingSession(self.calls)


async def test_chunk_mentions_persist_source_and_page():
    service = KnowledgeGraphService()
    driver = _RecordingDriver()
    service._driver = driver

    await service.upsert_chunk_mentions(
        "doc#chunk-1",
        ["OpenAI"],
        content="source content",
        source="/docs/report.pdf",
        page=3,
        doc_id="doc-001",
        revision_id="rev-001",
    )

    _, params = driver.calls[0]
    assert params["source"] == "/docs/report.pdf"
    assert params["page"] == 3
    assert params["doc_id"] == "doc-001"
    assert params["revision_id"] == "rev-001"
    assert "c.page" in driver.calls[0][0]


async def test_event_is_saved_as_hub_with_role_arguments():
    service = KnowledgeGraphService()
    driver = _RecordingDriver()
    service._driver = driver
    event = KnowledgeEvent(
        trigger="收购",
        type="并购",
        arguments=[EventArgument(role="收购方", entity="甲公司")],
    )

    await service.upsert_event(
        event,
        "doc#chunk-1",
        content="甲公司收购乙公司",
        source="/docs/news.pdf",
        page=2,
        doc_id="doc-001",
        revision_id="rev-001",
    )

    cypher, params = driver.calls[0]
    assert "Event" in cypher
    assert "CONTAINS_EVENT" in cypher
    assert "HAS_ARGUMENT" in cypher
    assert params["arguments"] == [{"role": "收购方", "entity": "甲公司"}]
    assert params["doc_id"] == "doc-001"
    assert params["revision_id"] == "rev-001"


async def test_delete_document_removes_subgraph_but_only_orphan_entities():
    service = KnowledgeGraphService()
    service.execute_cypher = AsyncMock(side_effect=[[], [], [{"deleted": 2}], []])

    deleted = await service.delete_document("doc-001", source="/docs/news.pdf")

    assert deleted == 2
    calls = service.execute_cypher.await_args_list
    assert len(calls) == 4
    assert "CONTAINS_EVENT" in calls[0].args[0]
    assert "source_doc_id" in calls[1].args[0]
    assert "MATCH (c:Chunk)" in calls[2].args[0]
    assert "NOT (e)--()" in calls[3].args[0]
    assert calls[2].args[1] == {"doc_id": "doc-001", "source": "/docs/news.pdf"}
