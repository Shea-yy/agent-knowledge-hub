"""入库工作流的事件、失败状态和溯源传递测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import (
    Entity,
    EventArgument,
    ExtractionResult,
    KnowledgeEvent,
    Relation,
)
from orchestrator.graph import _build_ingest_graph


@pytest.mark.asyncio
async def test_ingest_writes_chunk_provenance_and_events():
    chunk = DocumentChunk(
        content="甲公司收购乙公司",
        doc_id="doc-001",
        chunk_index=0,
        doc_type=DocType.PDF,
        metadata={"source": "/docs/news.pdf", "page": 2, "revision_id": "rev-001"},
    )
    extraction = ExtractionResult(
        entities=[
            Entity(name="甲公司", type="Organization"),
            Entity(name="乙公司", type="Organization"),
        ],
        relations=[Relation(head="甲公司", relation="related_to", tail="乙公司")],
        events=[KnowledgeEvent(
            trigger="收购",
            type="并购",
            arguments=[EventArgument(role="收购方", entity="甲公司")],
        )],
        source_chunk_id=chunk.chunk_id,
    )
    parser = MagicMock()
    parser.parse_batch = AsyncMock(return_value=[chunk])
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=[extraction])
    graph_service = MagicMock()
    graph_service.upsert_entity = AsyncMock()
    graph_service.upsert_chunk_mentions = AsyncMock()
    graph_service.add_relation = AsyncMock()
    graph_service.upsert_event = AsyncMock()

    workflow = _build_ingest_graph(parser, extractor, vector_store=None, knowledge_graph=graph_service)
    result = await workflow.ainvoke(
        {"file_paths": ["/docs/news.pdf"]},
        config={"configurable": {"thread_id": "ingest-event-test"}},
    )

    assert result["events_stored"] == 1
    assert result["extraction_failures"] == []
    graph_service.upsert_chunk_mentions.assert_awaited_once_with(
        chunk.chunk_id,
        ["甲公司", "乙公司"],
        content="甲公司收购乙公司",
        source="/docs/news.pdf",
        page=2,
        doc_id="doc-001",
        revision_id="rev-001",
    )
    graph_service.add_relation.assert_awaited_once_with(
        extraction.relations[0],
        source="/docs/news.pdf",
        source_chunk_id=chunk.chunk_id,
        doc_id="doc-001",
        revision_id="rev-001",
    )
    graph_service.upsert_event.assert_awaited_once_with(
        extraction.events[0],
        chunk.chunk_id,
        content="甲公司收购乙公司",
        source="/docs/news.pdf",
        page=2,
        doc_id="doc-001",
        revision_id="rev-001",
    )


@pytest.mark.asyncio
async def test_ingest_exposes_extraction_failure_without_aborting():
    chunk = DocumentChunk(
        content="unavailable provider test",
        doc_id="doc-002",
        chunk_index=0,
        doc_type=DocType.TEXT,
        metadata={"source": "/docs/a.txt"},
    )
    failed = ExtractionResult(
        source_chunk_id=chunk.chunk_id,
        status="failed",
        error="APITimeoutError: provider timeout",
    )
    parser = MagicMock()
    parser.parse_batch = AsyncMock(return_value=[chunk])
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=[failed])

    workflow = _build_ingest_graph(parser, extractor, vector_store=None, knowledge_graph=None)
    result = await workflow.ainvoke(
        {"file_paths": ["/docs/a.txt"]},
        config={"configurable": {"thread_id": "ingest-failure-test"}},
    )

    assert result["extraction_failures"] == [{
        "chunk_id": chunk.chunk_id,
        "error": "APITimeoutError: provider timeout",
    }]
