"""
知识更新 Agent — 监听文档变更并一致性更新向量库和知识图谱

核心能力:
  1. 文件系统监听 (Watchdog) / Kafka CDC 消费
  2. 文档级替换：修改前先完成新版本解析，再清理旧版本数据
  3. 向量库与知识图谱使用相同 doc_id 生命周期
  4. 版本管理：知识节点带时间戳和版本号
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from agents.doc_parser_agent import DocParserAgent
from config import settings

log = logger.bind(module="knowledge_update")


class ChangeType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    # 仅保留给 CDC 诊断/审计；当前 chunk_id 依赖 chunk_index，不能安全地
    # 依据文本行差异局部替换，因此它不参与写入路由。
    diff_chunks: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    change: DocumentChange
    chunks_processed: int = 0
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    events_added: int = 0
    graph_records_deleted: int = 0
    extraction_failures: list[dict[str, str]] = field(default_factory=list)
    update_mode: str = ""
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0


class KnowledgeUpdateAgent:
    """
    知识更新 Agent

    支持两种模式:
      1. 文件监听模式 (Watchdog): 监听本地文件系统变更
      2. CDC 模式 (Kafka): 消费来自消息队列的变更事件

    工作流:
      detect_change → diff_analysis → incremental_parse → update_vector_store → update_knowledge_graph → log
    """

    def __init__(
        self,
        doc_parser: Any = None,
        knowledge_extractor: Any = None,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.doc_parser = doc_parser
        self.knowledge_extractor = knowledge_extractor
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self._file_hashes: dict[str, str] = {}
        self._version_counter: dict[str, int] = {}

    # ── public API ───────────────────────────────────────────

    async def process_change(self, change: DocumentChange) -> UpdateResult:
        """处理单个文档变更"""
        start = time.time()
        result = UpdateResult(change=change)

        try:
            if change.change_type == ChangeType.DELETED:
                await self._handle_delete(change, result)
            elif change.change_type == ChangeType.CREATED:
                await self._handle_create(change, result)
            elif change.change_type == ChangeType.MODIFIED:
                await self._handle_modify(change, result)
        except Exception as e:
            result.success = False
            result.error = str(e)
            log.error("更新失败: {} → {}", change.file_path, e)

        result.processing_time_ms = (time.time() - start) * 1000
        log.info("更新完成: {} ({}) → {:.0f}ms",
                 change.file_path, change.change_type.value, result.processing_time_ms)
        return result

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        """批量处理文档变更"""
        results: list[UpdateResult] = []
        for change in changes:
            results.append(await self.process_change(change))
        return results

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        """扫描文件列表，检测变更"""
        changes: list[DocumentChange] = []
        current_files = set(file_paths)

        for fp in current_files:
            new_hash = self._compute_hash(fp)
            old_hash = self._file_hashes.get(fp, "")

            if not old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                ))
            elif new_hash != old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
            self._file_hashes[fp] = new_hash

        for fp in set(self._file_hashes) - current_files:
            changes.append(DocumentChange(
                file_path=fp,
                change_type=ChangeType.DELETED,
                old_hash=self._file_hashes[fp],
            ))
            del self._file_hashes[fp]

        return changes

    # ── watchdog mode ────────────────────────────────────────

    def start_watching(self, directory: str) -> None:
        """启动文件系统监听（非阻塞，在独立线程运行）"""
        import threading
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        agent = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.CREATED)
                    asyncio.run(agent.process_change(change))

            def on_modified(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.MODIFIED)
                    asyncio.run(agent.process_change(change))

            def on_deleted(self, event):
                if not event.is_directory:
                    import asyncio
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.DELETED)
                    asyncio.run(agent.process_change(change))

        observer = Observer()
        observer.schedule(_Handler(), directory, recursive=True)

        def _run():
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        log.info("Watchdog 已启动，监听目录: {}", directory)

    # ── kafka CDC mode ───────────────────────────────────────

    async def start_kafka_consumer(self) -> None:
        """启动 Kafka CDC 消费者"""
        import json
        from confluent_kafka import Consumer

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "knowledge-update-agent",
            "auto.offset.reset": "latest",
        }
        consumer = Consumer(conf)
        consumer.subscribe([settings.kafka_topic_doc_changes])

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                payload = json.loads(msg.value().decode("utf-8"))
                change = DocumentChange(
                    file_path=payload["file_path"],
                    change_type=ChangeType(payload["change_type"]),
                    old_hash=payload.get("old_hash", ""),
                    new_hash=payload.get("new_hash", ""),
                )
                await self.process_change(change)
        finally:
            consumer.close()

    # ── internal handlers ────────────────────────────────────

    async def _handle_create(self, change: DocumentChange, result: UpdateResult) -> None:
        """首次写入一个文档。"""
        chunks, extractions = await self._prepare_document(change, result)
        await self._store_document(chunks, extractions, result)
        result.update_mode = "create"

    async def _prepare_document(
        self,
        change: DocumentChange,
        result: UpdateResult,
    ) -> tuple[list[Any], list[Any]]:
        """在删除旧数据前完成解析与知识抽取，解析失败时保留旧版本。"""
        if not self.doc_parser:
            raise RuntimeError("文档解析器未初始化")

        chunks = await self.doc_parser.parse(change.file_path)
        result.chunks_processed = len(chunks)
        extractions: list[Any] = []
        if self.knowledge_extractor and self.knowledge_graph:
            extractions = await self.knowledge_extractor.extract(chunks)
            result.extraction_failures.extend(
                {
                    "chunk_id": item.source_chunk_id,
                    "error": item.error,
                }
                for item in extractions
                if getattr(item, "status", "success") == "failed"
            )
        return chunks, extractions

    async def _store_document(
        self,
        chunks: list[Any],
        extractions: list[Any],
        result: UpdateResult,
    ) -> None:
        """将已准备好的新版本写入两个存储后端。"""
        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
            result.vectors_added = len(chunks)

        if not self.knowledge_graph:
            return

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
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
                version = self._bump_version(ent.name)
                await self.knowledge_graph.upsert_entity(
                    ent, version=version, source=source,
                )
                result.entities_added += 1
            if ext.entities:
                await self.knowledge_graph.upsert_chunk_mentions(
                    ext.source_chunk_id,
                    [entity.name for entity in ext.entities],
                    content=content,
                    source=source,
                    page=page,
                    doc_id=doc_id,
                    revision_id=revision_id,
                )
            for rel in ext.relations:
                await self.knowledge_graph.add_relation(
                    rel,
                    source=source,
                    source_chunk_id=ext.source_chunk_id,
                    doc_id=doc_id,
                    revision_id=revision_id,
                )
                result.relations_added += 1
            for event in ext.events:
                await self.knowledge_graph.upsert_event(
                    event,
                    ext.source_chunk_id,
                    content=content,
                    source=source,
                    page=page,
                    doc_id=doc_id,
                    revision_id=revision_id,
                )
                result.events_added += 1

    async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
        """安全的文档级全量替换。

        当前 chunk_id 由 ``doc_id + chunk_index`` 构成。插入或删除一行即可
        改变其后的所有索引，不能根据 ``diff_chunks`` 做可靠的局部更新。
        因此先准备新版本，成功后才清理旧数据并重建，避免旧 Chunk/Event
        残留，也避免把这种策略误称为增量更新。
        """
        chunks, extractions = await self._prepare_document(change, result)
        await self._remove_document(change, result)
        await self._store_document(chunks, extractions, result)
        result.update_mode = "full_replace"

    async def _handle_delete(self, change: DocumentChange, result: UpdateResult) -> None:
        await self._remove_document(change, result)
        result.update_mode = "delete"

    async def _remove_document(self, change: DocumentChange, result: UpdateResult) -> None:
        """尽力清理两个存储后端；任一失败也不会阻止另一端清理。"""
        doc_id = DocParserAgent._make_doc_id(change.file_path)
        errors: list[str] = []

        if self.vector_store:
            try:
                result.vectors_deleted = await self.vector_store.delete_by_doc_id(doc_id)
            except Exception as exc:
                errors.append(f"vector_store: {exc}")

        if self.knowledge_graph:
            try:
                result.graph_records_deleted = await self.knowledge_graph.delete_document(
                    doc_id=doc_id,
                    source=change.file_path,
                )
            except Exception as exc:
                errors.append(f"knowledge_graph: {exc}")

        if errors:
            raise RuntimeError("; ".join(errors))

    # ── utilities ────────────────────────────────────────────

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""

    def _bump_version(self, entity_name: str) -> int:
        ver = self._version_counter.get(entity_name, 0) + 1
        self._version_counter[entity_name] = ver
        return ver
