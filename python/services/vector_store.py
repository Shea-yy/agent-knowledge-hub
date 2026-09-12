"""
向量存储服务 — 支持 ChromaDB / PGVector 双后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（支持增量更新）
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from agents.doc_parser_agent import DocumentChunk
from config import settings

log = logger.bind(module="vector_store")


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB / PGVector"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        # check_embedding_ctx_length=False: 跳过 tiktoken 校验
        # （tiktoken 需要从 Azure 下载编码文件，国内网络不稳定）
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            check_embedding_ctx_length=False,
        )
        self._store: Any = None
        self._backend = settings.vector_store_type

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()
        log.info("向量库已初始化 ({})", self._backend)

    async def _init_chroma(self) -> None:
        import chromadb
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        self._store = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector
        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            # 文档级删除依赖 metadata 过滤；JSONB 是 PGVector 推荐的可查询格式。
            use_jsonb=True,
        )

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """向量化并存储文档块"""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type.value,
                "source": c.metadata.get("source", ""),
                "chunk_index": c.chunk_index,
                "revision_id": c.metadata.get("revision_id", ""),
                "page": c.metadata.get("page"),
            }
            for c in chunks
        ]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)

        log.info("向量入库: {} chunks", len(chunks))
        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """语义搜索，返回 (文档, 分数) 列表"""
        if self._backend == "chroma":
            q_vec = await self.embeddings.aembed_query(query)
            results = self._store.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
            out: list[tuple[dict, float]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                # Chroma cosine distance 的理论范围是 [0, 2]；上层统一使用
                # 0~1 相似度，避免反向向量得到负分。
                score = max(0.0, min(1.0, 1.0 - float(dist)))
                out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
            return out
        else:
            # PGVector 的 with_score 返回距离；统一接口向上层承诺 0~1 的相似度，
            # 所以必须使用其 relevance-score API。
            results = await self._store.asimilarity_search_with_relevance_scores(query, k=top_k)
            return [
                ({"content": doc.page_content, "source": doc.metadata.get("source", ""), "metadata": doc.metadata}, score)
                for doc, score in results
            ]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关向量"""
        if self._backend == "chroma":
            existing = self._store.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
            return len(ids)
        return await asyncio.to_thread(self._delete_pgvector_by_doc_id, doc_id)

    def _delete_pgvector_by_doc_id(self, doc_id: str) -> int:
        """删除 PGVector collection 中指定文档的向量，返回实际删除数量。

        ``langchain_community.PGVector.delete`` 只接受 vector ID，不支持
        metadata 过滤；这里先用同一 collection 的 JSONB metadata 找出 ID，
        再调用其公开 delete API。若底层接口不兼容则明确失败，不能伪装成
        “删除 0 条”。
        """
        try:
            from sqlalchemy import select
            from sqlalchemy.orm import Session
        except ImportError as exc:  # pragma: no cover - PGVector 的运行时依赖
            raise RuntimeError("PGVector 删除需要 SQLAlchemy") from exc

        embedding_store = getattr(self._store, "EmbeddingStore", None)
        bind = getattr(self._store, "_bind", None)
        get_collection = getattr(self._store, "get_collection", None)
        if embedding_store is None or bind is None or get_collection is None:
            raise RuntimeError("当前 PGVector 实现不支持按 doc_id 删除")

        with Session(bind) as session:
            collection = get_collection(session)
            if not collection:
                return 0
            statement = (
                select(embedding_store.custom_id)
                .where(embedding_store.collection_id == collection.uuid)
                .where(embedding_store.cmetadata["doc_id"].as_string() == doc_id)
            )
            ids = [row[0] for row in session.execute(statement) if row[0]]

        if ids:
            self._store.delete(ids=ids, collection_only=True)
        return len(ids)

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if self._backend == "chroma":
            count = self._store.count()
            return {"backend": "chroma", "total_vectors": count, "collection": self.COLLECTION_NAME}
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
