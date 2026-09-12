"""增量更新路径应与首次入库保持相同的溯源和事件写入语义。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, EventArgument, ExtractionResult, KnowledgeEvent
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent


@pytest.mark.asyncio
async def test_create_persists_mentions_and_events():
    chunk = DocumentChunk(
        content="OpenAI 发布产品",
        doc_id="doc-001",
        chunk_index=0,
        doc_type=DocType.PDF,
        metadata={"source": "/docs/release.pdf", "page": 4, "revision_id": "rev-001"},
    )
    extraction = ExtractionResult(
        entities=[Entity(name="OpenAI", type="Organization")],
        events=[KnowledgeEvent(
            trigger="发布",
            type="产品发布",
            arguments=[EventArgument(role="发布方", entity="OpenAI")],
        )],
        source_chunk_id=chunk.chunk_id,
    )
    parser = MagicMock()
    parser.parse = AsyncMock(return_value=[chunk])
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=[extraction])
    knowledge_graph = MagicMock()
    knowledge_graph.upsert_entity = AsyncMock()
    knowledge_graph.upsert_chunk_mentions = AsyncMock()
    knowledge_graph.add_relation = AsyncMock()
    knowledge_graph.upsert_event = AsyncMock()

    agent = KnowledgeUpdateAgent(
        doc_parser=parser,
        knowledge_extractor=extractor,
        vector_store=None,
        knowledge_graph=knowledge_graph,
    )
    result = await agent.process_change(DocumentChange(
        file_path="/docs/release.pdf",
        change_type=ChangeType.CREATED,
    ))

    assert result.success
    assert result.events_added == 1
    knowledge_graph.upsert_chunk_mentions.assert_awaited_once_with(
        chunk.chunk_id,
        ["OpenAI"],
        content="OpenAI 发布产品",
        source="/docs/release.pdf",
        page=4,
        doc_id="doc-001",
        revision_id="rev-001",
    )
    knowledge_graph.upsert_event.assert_awaited_once_with(
        extraction.events[0],
        chunk.chunk_id,
        content="OpenAI 发布产品",
        source="/docs/release.pdf",
        page=4,
        doc_id="doc-001",
        revision_id="rev-001",
    )


@pytest.mark.asyncio
async def test_modify_prepares_new_version_before_cleaning_old_records():
    """解析或抽取失败前，不得删除上一版本的可检索数据。"""
    chunk = DocumentChunk(
        content="new content",
        doc_id="doc-001",
        chunk_index=0,
        doc_type=DocType.TEXT,
        metadata={"source": "/docs/report.txt", "revision_id": "rev-new"},
    )
    calls: list[str] = []

    async def parse(_path):
        calls.append("parse")
        return [chunk]

    async def extract(_chunks):
        calls.append("extract")
        return []

    async def delete_vectors(_doc_id):
        calls.append("delete_vectors")
        return 2

    async def add_vectors(_chunks):
        calls.append("add_vectors")
        return 1

    async def delete_graph(**_kwargs):
        calls.append("delete_graph")
        return 3

    parser = MagicMock()
    parser.parse = AsyncMock(side_effect=parse)
    extractor = MagicMock()
    extractor.extract = AsyncMock(side_effect=extract)
    vector_store = MagicMock()
    vector_store.delete_by_doc_id = AsyncMock(side_effect=delete_vectors)
    vector_store.add_chunks = AsyncMock(side_effect=add_vectors)
    knowledge_graph = MagicMock()
    knowledge_graph.delete_document = AsyncMock(side_effect=delete_graph)

    agent = KnowledgeUpdateAgent(parser, extractor, vector_store, knowledge_graph)
    result = await agent.process_change(DocumentChange(
        file_path="/docs/report.txt",
        change_type=ChangeType.MODIFIED,
        diff_chunks=["a small textual change"],
    ))

    assert result.success
    assert result.update_mode == "full_replace"
    assert result.chunks_processed == 1
    assert result.vectors_deleted == 2
    assert result.graph_records_deleted == 3
    assert calls == ["parse", "extract", "delete_vectors", "delete_graph", "add_vectors"]


@pytest.mark.asyncio
async def test_modify_parse_failure_preserves_old_records():
    parser = MagicMock()
    parser.parse = AsyncMock(side_effect=ValueError("malformed document"))
    vector_store = MagicMock()
    vector_store.delete_by_doc_id = AsyncMock()
    knowledge_graph = MagicMock()
    knowledge_graph.delete_document = AsyncMock()

    agent = KnowledgeUpdateAgent(parser, vector_store=vector_store, knowledge_graph=knowledge_graph)
    result = await agent.process_change(DocumentChange(
        file_path="/docs/broken.pdf",
        change_type=ChangeType.MODIFIED,
    ))

    assert not result.success
    assert "malformed document" in result.error
    vector_store.delete_by_doc_id.assert_not_awaited()
    knowledge_graph.delete_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_attempts_graph_cleanup_when_vector_cleanup_fails():
    vector_store = MagicMock()
    vector_store.delete_by_doc_id = AsyncMock(side_effect=RuntimeError("vector unavailable"))
    knowledge_graph = MagicMock()
    knowledge_graph.delete_document = AsyncMock(return_value=4)

    agent = KnowledgeUpdateAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)
    result = await agent.process_change(DocumentChange(
        file_path="/docs/deleted.txt",
        change_type=ChangeType.DELETED,
    ))

    assert not result.success
    assert "vector unavailable" in result.error
    knowledge_graph.delete_document.assert_awaited_once()
    assert result.graph_records_deleted == 4
