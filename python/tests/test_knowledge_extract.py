"""知识抽取 Agent 的失败状态与事件清洗测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agents.knowledge_extract_agent import (
    Entity,
    EventArgument,
    ExtractionResult,
    KnowledgeEvent,
    KnowledgeExtractAgent,
)


class TestExtractionFailureState:
    @pytest.mark.asyncio
    async def test_retains_failure_status_and_error(self):
        agent = object.__new__(KnowledgeExtractAgent)
        agent._extract_with_retry = AsyncMock(side_effect=RuntimeError("provider unavailable"))

        result = await agent._extract_from_text("test", "doc#chunk-1")

        assert result.source_chunk_id == "doc#chunk-1"
        assert result.status == "failed"
        assert "RuntimeError" in result.error

    def test_filter_preserves_event_and_failure_status(self):
        entity = Entity(name="OpenAI", type="Organization")
        event = KnowledgeEvent(
            trigger="发布",
            type="产品发布",
            arguments=[EventArgument(role="发布方", entity="OpenAI")],
        )
        result = ExtractionResult(
            entities=[entity],
            events=[event],
            source_chunk_id="doc#chunk-1",
            status="failed",
            error="partial provider failure",
        )

        cleaned = KnowledgeExtractAgent._filter_dangling([result])

        assert cleaned[0].events[0].arguments[0].entity == "OpenAI"
        assert cleaned[0].status == "failed"
        assert cleaned[0].error == "partial provider failure"
