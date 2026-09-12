"""向量存储的文档级删除与统一分数语义测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.vector_store import VectorStoreService


@pytest.mark.asyncio
async def test_chroma_delete_by_doc_id_returns_actual_deleted_count():
    service = object.__new__(VectorStoreService)
    service._backend = "chroma"
    service._store = MagicMock()
    service._store.get.return_value = {"ids": ["doc-1#chunk-0", "doc-1#chunk-1"]}

    deleted = await service.delete_by_doc_id("doc-1")

    assert deleted == 2
    service._store.get.assert_called_once_with(where={"doc_id": "doc-1"}, include=[])
    service._store.delete.assert_called_once_with(ids=["doc-1#chunk-0", "doc-1#chunk-1"])


@pytest.mark.asyncio
async def test_pgvector_delete_delegates_to_metadata_filtered_implementation():
    service = object.__new__(VectorStoreService)
    service._backend = "pgvector"
    service._delete_pgvector_by_doc_id = MagicMock(return_value=3)

    deleted = await service.delete_by_doc_id("doc-1")

    assert deleted == 3
    service._delete_pgvector_by_doc_id.assert_called_once_with("doc-1")


@pytest.mark.asyncio
async def test_chroma_distance_is_clamped_to_public_similarity_range():
    service = object.__new__(VectorStoreService)
    service._backend = "chroma"
    service.embeddings = MagicMock()
    service.embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2])
    service._store = MagicMock()
    service._store.query.return_value = {
        "documents": [["opposite vector"]],
        "metadatas": [[{"source": "doc.txt"}]],
        "distances": [[1.6]],
    }

    results = await service.search("query", top_k=1)

    assert results == [(
        {"content": "opposite vector", "source": "doc.txt", "metadata": {"source": "doc.txt"}},
        0.0,
    )]
