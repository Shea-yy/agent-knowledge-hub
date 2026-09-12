"""
知识图谱服务 — Neo4j 图数据库操作

职责:
  1. 实体 (Node) CRUD — 带版本号和时间戳
  2. 关系 (Relationship) CRUD
  3. Cypher 查询执行
  4. 子图检索（多跳遍历）
  5. 按稳定 doc_id 删除文档子图
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from loguru import logger

from agents.knowledge_extract_agent import Entity, KnowledgeEvent, Relation
from config import settings

log = logger.bind(module="knowledge_graph")


class KnowledgeGraphService:
    """Neo4j 知识图谱服务"""

    def __init__(self) -> None:
        self._driver: Any = None

    # ── lifecycle ────────────────────────────────────────────

    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()
        log.info("Neo4j 已连接: {}", settings.neo4j_uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        """创建常用索引以加速查询"""
        index_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:Chunk) ON (n.doc_id)",
        ]
        async with self._driver.session() as session:
            for q in index_queries:
                await session.run(q)

    # ── entity operations ────────────────────────────────────

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
        """
        创建或更新实体节点 — MERGE 语义
        带版本号和时间戳，支持增量更新追踪
        """
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.version = $version,
            e.source = $source,
            e.created_at = $now,
            e.updated_at = $now
        ON MATCH SET
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
            e.version = $version,
            e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "version": version,
                "source": source,
                "now": int(time.time()),
            })

    async def add_relation(
        self,
        relation: Relation,
        source: str = "",
        source_chunk_id: str = "",
        doc_id: str = "",
        revision_id: str = "",
    ) -> None:
        """创建带文档/Chunk 级溯源的实体关系。

        同一对实体可由多个 Chunk 提供证据；因此关系的幂等键包含
        ``source_chunk_id``，而不能只按实体对 MERGE。这样删除一个文档时
        不会误删另一个文档为同一关系提供的证据。
        """
        rel_type = "".join(
            char if char.isalnum() or char == "_" else "_"
            for char in relation.relation.upper().replace(" ", "_")
        ).strip("_") or "RELATED_TO"
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{rel_type} {{source_chunk_id: $source_chunk_id}}]->(t)
        SET r.confidence = $confidence,
            r.source = $source,
            r.source_doc_id = $doc_id,
            r.revision_id = $revision_id,
            r.updated_at = $now,
            r.evidence = CASE WHEN $evidence <> '' THEN $evidence ELSE r.evidence END
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "head": relation.head,
                "tail": relation.tail,
                "confidence": relation.confidence,
                "evidence": getattr(relation, "evidence", ""),
                "source": source,
                "source_chunk_id": source_chunk_id,
                "doc_id": doc_id,
                "revision_id": revision_id,
                "now": int(time.time()),
            })

    async def upsert_chunk_mentions(
        self,
        chunk_id: str,
        entity_names: list[str],
        content: str = "",
        source: str = "",
        page: int | None = None,
        doc_id: str = "",
        revision_id: str = "",
    ) -> None:
        """
        血缘边: (Chunk)-[:MENTIONS]->(Entity)，每 chunk 一条批量查询。

        GraphRAG 溯源基础——支持反向追问"这个实体被哪些文档段落提及"。
        同时把 chunk 原文、来源与页码落进节点，使图谱查询能直接带回
        可定位的原文片段。MERGE 保证幂等。
        """
        if not chunk_id or not entity_names:
            return
        cypher = """
        MERGE (c:Chunk {id: $chunk_id})
        ON CREATE SET c.content = $content, c.source = $source,
                      c.page = $page, c.doc_id = $doc_id,
                      c.revision_id = $revision_id, c.created_at = $now
        ON MATCH SET
            c.content = CASE WHEN $content <> '' THEN $content ELSE c.content END,
            c.source = CASE WHEN $source <> '' THEN $source ELSE c.source END,
            c.page = CASE WHEN $page IS NOT NULL THEN $page ELSE c.page END,
            c.doc_id = CASE WHEN $doc_id <> '' THEN $doc_id ELSE c.doc_id END,
            c.revision_id = CASE WHEN $revision_id <> '' THEN $revision_id ELSE c.revision_id END,
            c.updated_at = $now
        WITH c
        UNWIND $names AS name
        MERGE (e:Entity {name: name})
        MERGE (c)-[:MENTIONS]->(e)
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "chunk_id": chunk_id,
                "names": list(entity_names),
                "content": content,
                "source": source,
                "page": page,
                "doc_id": doc_id,
                "revision_id": revision_id,
                "now": int(time.time()),
            })

    async def upsert_event(
        self,
        event: KnowledgeEvent,
        source_chunk_id: str,
        content: str = "",
        source: str = "",
        page: int | None = None,
        doc_id: str = "",
        revision_id: str = "",
    ) -> None:
        """写入事件 Hub 节点、论元角色边及其 Chunk 溯源。"""
        if not source_chunk_id:
            return

        arguments = [argument.model_dump() for argument in event.arguments]
        signature = json.dumps(
            {
                "chunk_id": source_chunk_id,
                "trigger": event.trigger,
                "type": event.type,
                "arguments": arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        event_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
        cypher = """
        MERGE (c:Chunk {id: $chunk_id})
        ON CREATE SET c.content = $content, c.source = $source,
                      c.page = $page, c.doc_id = $doc_id,
                      c.revision_id = $revision_id, c.created_at = $now
        ON MATCH SET
            c.content = CASE WHEN $content <> '' THEN $content ELSE c.content END,
            c.source = CASE WHEN $source <> '' THEN $source ELSE c.source END,
            c.page = CASE WHEN $page IS NOT NULL THEN $page ELSE c.page END,
            c.doc_id = CASE WHEN $doc_id <> '' THEN $doc_id ELSE c.doc_id END,
            c.revision_id = CASE WHEN $revision_id <> '' THEN $revision_id ELSE c.revision_id END,
            c.updated_at = $now
        MERGE (ev:Event {id: $event_id})
        ON CREATE SET ev.trigger = $trigger, ev.type = $type,
                      ev.source = $source, ev.doc_id = $doc_id,
                      ev.revision_id = $revision_id, ev.page = $page,
                      ev.source_chunk_id = $chunk_id, ev.created_at = $now
        ON MATCH SET ev.doc_id = CASE WHEN $doc_id <> '' THEN $doc_id ELSE ev.doc_id END,
                     ev.revision_id = CASE WHEN $revision_id <> '' THEN $revision_id ELSE ev.revision_id END,
                     ev.updated_at = $now
        MERGE (c)-[:CONTAINS_EVENT]->(ev)
        WITH ev
        UNWIND $arguments AS argument
        MATCH (e:Entity {name: argument.entity})
        MERGE (ev)-[r:HAS_ARGUMENT {role: argument.role}]->(e)
        SET r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "chunk_id": source_chunk_id,
                "content": content,
                "source": source,
                "page": page,
                "doc_id": doc_id,
                "revision_id": revision_id,
                "event_id": event_id,
                "trigger": event.trigger,
                "type": event.type,
                "arguments": arguments,
                "now": int(time.time()),
            })

    async def get_entity_mentions(self, entity_name: str, limit: int = 3) -> list[dict]:
        """
        溯源查询: 某实体被哪些 chunk 提及，返回原文片段。

        参数化查询——entity_name 来自 LLM 生成，绝不字符串插值。
        """
        cypher = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $name})
        RETURN c.id AS chunk_id, c.content AS content, c.source AS source, c.page AS page
        LIMIT $limit
        """
        return await self.execute_cypher(
            cypher, {"name": entity_name, "limit": limit}
        )

    # ── query operations ─────────────────────────────────────

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """执行任意 Cypher 查询"""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def get_entity(self, name: str) -> dict | None:
        """查询单个实体"""
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        """
        多跳子图检索 — GraphRAG 的核心能力
        从指定实体出发，遍历 N 跳内的所有关联实体和关系
        """
        cypher = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor)
        RETURN
            start.name AS source,
            [r IN relationships(path) | type(r)] AS relations,
            neighbor.name AS target,
            neighbor.type AS target_type,
            neighbor.description AS target_desc
        LIMIT 50
        """
        return await self.execute_cypher(cypher, {"name": entity_name})

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        """模糊搜索实体"""
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    # ── delete operations ────────────────────────────────────

    async def delete_document(self, doc_id: str, source: str = "") -> int:
        """删除一个文档写入的图谱子图，并保留仍被其他文档引用的实体。

        ``source`` 是兼容旧数据的回退条件：旧版本 Chunk 尚未存储
        ``doc_id`` 时，仍可由其规范化路径清理。实体是跨文档共享节点，
        所以只在清理完成后删除孤立实体。
        """
        if not doc_id and not source:
            raise ValueError("删除图谱文档时必须提供 doc_id 或 source")

        where = "c.doc_id = $doc_id OR ($source <> '' AND c.source = $source)"
        event_query = f"""
        MATCH (c:Chunk)-[:CONTAINS_EVENT]->(ev:Event)
        WHERE {where}
        DETACH DELETE ev
        """
        relation_query = """
        MATCH ()-[r]->()
        WHERE r.source_doc_id = $doc_id OR ($source <> '' AND r.source = $source)
        DELETE r
        """
        chunk_query = f"""
        MATCH (c:Chunk)
        WHERE {where}
        DETACH DELETE c
        RETURN count(c) AS deleted
        """
        orphan_query = """
        MATCH (e:Entity)
        WHERE NOT (e)--()
        DELETE e
        """
        params = {"doc_id": doc_id, "source": source}
        await self.execute_cypher(event_query, params)
        await self.execute_cypher(relation_query, params)
        records = await self.execute_cypher(chunk_query, params)
        await self.execute_cypher(orphan_query)
        return records[0].get("deleted", 0) if records else 0

    async def delete_by_source(self, source: str) -> int:
        """兼容旧调用：按来源清理一个文档子图。"""
        return await self.delete_document(doc_id="", source=source)

    # ── stats ────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """获取图谱统计信息"""
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }
