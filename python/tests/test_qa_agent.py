"""
QA Agent 单元测试

测试 _parse_agent_result 及其静态方法 — 纯逻辑，不调 LLM。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.qa_agent import QAAgent, QAResult, QueryIntent, RetrievedContext


class TestExtractContexts:
    """从 messages 中提取检索上下文"""

    def test_extracts_from_tool_messages(self):
        """ToolMessage 中的 JSON 被正确解析为 RetrievedContext"""
        messages = [
            HumanMessage(content="test"),
            ToolMessage(
                content='[{"rank":1,"score":0.95,"content":"test content","source":"test.pdf"}]',
                name="vector_search",
                tool_call_id="call_1",
            ),
            AIMessage(content="answer"),
        ]
        contexts = QAAgent._extract_contexts(messages)
        assert len(contexts) == 1
        assert contexts[0].content == "test content"
        assert contexts[0].source == "test.pdf"
        assert contexts[0].score == 0.95
        assert contexts[0].retrieval_type == "vector"

    def test_classifies_graph_retrieval(self):
        """cypher_query / entity_lookup 标记为 graph 类型"""
        messages = [
            HumanMessage(content="test"),
            ToolMessage(
                content='{"entity":"张三","relationships":[{"relation":"works_at","target":"腾讯"}]}',
                name="entity_lookup",
                tool_call_id="call_1",
            ),
            AIMessage(content="answer"),
        ]
        contexts = QAAgent._extract_contexts(messages)
        assert len(contexts) == 1
        assert contexts[0].retrieval_type == "graph"

    def test_does_not_treat_missing_entity_as_evidence(self):
        """entity 是查询条件，查无结果时不能伪造图谱引用。"""
        messages = [
            ToolMessage(
                content='{"message":"未找到实体 \'不存在的实体\'","entity":"不存在的实体"}',
                name="entity_lookup",
                tool_call_id="call_1",
            ),
        ]

        assert QAAgent._extract_contexts(messages) == []

    def test_extracts_generic_tool_contexts(self):
        """表格和 Cypher 工具的统一 contexts 契约可被解析。"""
        messages = [
            ToolMessage(
                content=(
                    '{"contexts":[{"content":"员工邮箱: a@example.com",'
                    '"source":"员工名录.csv","score":0.9,"retrieval_type":"table"}]}'
                ),
                name="table_search",
                tool_call_id="call_1",
            ),
        ]
        contexts = QAAgent._extract_contexts(messages)
        assert len(contexts) == 1
        assert contexts[0].source == "员工名录.csv"
        assert contexts[0].retrieval_type == "table"

    def test_handles_invalid_json(self):
        """非 JSON 的 ToolMessage 不崩溃"""
        messages = [
            ToolMessage(content="not json at all", name="vector_search", tool_call_id="call_1"),
        ]
        contexts = QAAgent._extract_contexts(messages)
        assert contexts == []

    def test_empty_messages(self):
        """空 messages 返回空列表"""
        assert QAAgent._extract_contexts([]) == []

    def test_deduplicates_by_truncated_content(self):
        """仅在 API 截断边界后不同的片段，不能显示为两条相同引用。"""
        shared_prefix = "x" * 500
        messages = [
            ToolMessage(
                content=json.dumps([
                    {"content": shared_prefix + "first", "source": "doc.md", "score": 0.8},
                    {"content": shared_prefix + "second", "source": "doc.md", "score": 0.9},
                ]),
                name="vector_search",
                tool_call_id="call_1",
            ),
        ]

        contexts = QAAgent._extract_contexts(messages)

        assert len(contexts) == 1
        assert contexts[0].content == shared_prefix

    @pytest.mark.asyncio
    async def test_entity_lookup_keeps_relationships_when_mentions_fail(self, monkeypatch):
        """溯源查询失败不能抹掉已经成功取得的实体关系。"""
        import agents.qa_agent as qa_module

        captured: dict = {}
        monkeypatch.setattr(
            qa_module,
            "create_agent",
            lambda **kwargs: captured.update(kwargs) or MagicMock(),
        )
        graph = MagicMock()
        graph.get_neighbors = AsyncMock(return_value=[{
            "entity": "张三", "relation": "works_at", "target": "腾讯",
        }])
        graph.get_entity_mentions = AsyncMock(side_effect=RuntimeError("mentions unavailable"))

        agent = object.__new__(QAAgent)
        agent.llm = MagicMock()
        agent.vector_store = None
        agent.knowledge_graph = graph
        agent.table_store = None
        agent._build_agent()
        entity_lookup = next(tool for tool in captured["tools"] if tool.name == "entity_lookup")

        data = json.loads(await entity_lookup.ainvoke({"entity_name": "张三"}))

        assert data["relationships"][0]["target"] == "腾讯"
        assert data["contexts"] == []
        assert data["warnings"]

    @pytest.mark.asyncio
    async def test_table_info_returns_guidance_not_evidence(self, monkeypatch):
        """列元数据可供 Agent 规划查询，但不能成为回答引用。"""
        import agents.qa_agent as qa_module

        captured: dict = {}
        monkeypatch.setattr(
            qa_module,
            "create_agent",
            lambda **kwargs: captured.update(kwargs) or MagicMock(),
        )
        table_store = MagicMock()
        table_store.query.return_value = "列名: ['姓名', '邮箱']\n行数: 10"

        agent = object.__new__(QAAgent)
        agent.llm = MagicMock()
        agent.vector_store = None
        agent.knowledge_graph = None
        agent.table_store = table_store
        agent._build_agent()
        table_info = next(tool for tool in captured["tools"] if tool.name == "table_info")

        data = json.loads(await table_info.ainvoke({"table_name": "员工.csv"}))

        assert data["contexts"] == []
        assert "列名" in data["message"]


class TestCypherTool:
    """图查询工具除了 AOP 语法约束，还要验证运行时资源边界。"""

    @staticmethod
    def _build_tool(monkeypatch, graph):
        import agents.qa_agent as qa_module

        captured: dict = {}
        monkeypatch.setattr(
            qa_module,
            "create_agent",
            lambda **kwargs: captured.update(kwargs) or MagicMock(),
        )
        agent = object.__new__(QAAgent)
        agent.llm = MagicMock()
        agent.vector_store = None
        agent.knowledge_graph = graph
        agent.table_store = None
        agent._build_agent()
        return next(tool for tool in captured["tools"] if tool.name == "cypher_query")

    @pytest.mark.asyncio
    async def test_caps_cypher_records_even_if_backend_ignores_limit(self, monkeypatch):
        graph = MagicMock()
        graph.execute_cypher = AsyncMock(return_value=[{"id": index} for index in range(60)])
        cypher_query = self._build_tool(monkeypatch, graph)

        data = json.loads(await cypher_query.ainvoke({
            "cypher": "MATCH (n:Entity) RETURN n LIMIT 50",
        }))

        assert len(data["records"]) == 50
        assert len(data["contexts"]) == 50

    @pytest.mark.asyncio
    async def test_returns_controlled_error_when_cypher_times_out(self, monkeypatch):
        import agents.qa_agent as qa_module

        async def slow_query(_cypher):
            await asyncio.sleep(0.05)
            return []

        graph = MagicMock()
        graph.execute_cypher = slow_query
        monkeypatch.setattr(qa_module, "CYPHER_QUERY_TIMEOUT_SECONDS", 0.001)
        cypher_query = self._build_tool(monkeypatch, graph)

        data = json.loads(await cypher_query.ainvoke({
            "cypher": "MATCH (n:Entity) RETURN n LIMIT 1",
        }))

        assert "超时" in data["error"]


class TestInferIntent:
    """从消息内容中推断查询意图"""

    def test_factoid_default(self):
        """普通问句 → factoid"""
        assert QAAgent._infer_intent("张三的职位是什么？") == QueryIntent.FACTOID

    def test_comparative(self):
        """包含对比关键词 → comparative"""
        assert QAAgent._infer_intent("A 和 B 有什么区别？") == QueryIntent.COMPARATIVE

    def test_procedural(self):
        """包含如何/步骤 → procedural"""
        assert QAAgent._infer_intent("怎么搭建 CI/CD 流程？") == QueryIntent.PROCEDURAL

    def test_exploratory(self):
        """包含有哪些/概述 → exploratory"""
        assert QAAgent._infer_intent("有哪些技术选型方案？") == QueryIntent.EXPLORATORY

    def test_analytical(self):
        """包含为什么/分析 → analytical"""
        assert QAAgent._infer_intent("为什么选择 Neo4j 而不是其他图数据库？") == QueryIntent.ANALYTICAL

    def test_ignores_keywords_from_model_output(self):
        """意图只能由用户问题决定，答案中的关键词不能污染它。"""
        assert QAAgent._infer_intent("张三是谁？") == QueryIntent.FACTOID


class TestCalcConfidence:
    """置信度计算"""

    def test_no_contexts(self):
        """无检索结果 → 0"""
        assert QAAgent._calc_confidence([]) == 0.0

    def test_average_plus_retrieval_type_bonus(self):
        """置信度 = 平均分 + 跨检索类型奖励（上限 1.0）"""
        contexts = [
            RetrievedContext(content="a", source="s1", score=0.8, retrieval_type="vector"),
            RetrievedContext(content="b", source="s2", score=0.9, retrieval_type="graph"),
        ]
        conf = QAAgent._calc_confidence(contexts)
        avg = (0.8 + 0.9) / 2  # 0.85
        bonus = 0.05
        assert conf == pytest.approx(avg + bonus)
        assert conf <= 1.0

    def test_capped_at_one(self):
        """置信度不超过 1.0"""
        contexts = [
            RetrievedContext(content="x", source="s", score=1.0, retrieval_type="graph"),
        ]
        assert QAAgent._calc_confidence(contexts) <= 1.0


class TestExtractReasoningSteps:
    """提取 Agent 推理步骤"""

    def test_tracks_tool_calls(self, sample_messages):
        """能记录每一步工具调用"""
        steps = QAAgent._extract_reasoning_steps(sample_messages)
        tool_steps = [s for s in steps if "调用工具" in s]
        assert len(tool_steps) == 1
        assert "entity_lookup" in tool_steps[0]

    def test_ends_with_answer_generation(self):
        """最后一步是'生成最终答案'"""
        steps = QAAgent._extract_reasoning_steps([])
        assert steps[-1] == "生成最终答案"

    def test_does_not_expose_model_text_as_reasoning(self):
        steps = QAAgent._extract_reasoning_steps([AIMessage(content="模型内部文本")])
        assert steps == ["生成最终答案"]


class TestParseAgentResult:
    """集成测试 _parse_agent_result"""

    def test_full_result_parsing(self, sample_messages):
        """从完整 message history 中提取 QAResult"""
        # 构造一个不初始化 LLM 的 agent（只测解析逻辑）
        agent = object.__new__(QAAgent)
        result = agent._parse_agent_result("张三是谁？", {"messages": sample_messages})

        assert isinstance(result, QAResult)
        assert result.question == "张三是谁？"
        assert "腾讯" in result.answer
        assert result.intent == QueryIntent.FACTOID
        assert result.confidence > 0
        assert len(result.reasoning_steps) >= 2

    def test_empty_messages_returns_defaults(self):
        """空 messages → 返回合理的默认值"""
        agent = object.__new__(QAAgent)
        result = agent._parse_agent_result("test", {"messages": []})

        assert result.question == "test"
        assert result.answer == ""
        assert result.intent == QueryIntent.FACTOID
        assert result.confidence == 0.0
        assert result.contexts == []

    def test_only_returns_contexts_from_current_turn(self):
        """同一会话的历史工具结果不能被当作当前轮的引用来源。"""
        agent = object.__new__(QAAgent)
        messages = [
            HumanMessage(content="上一轮问题"),
            ToolMessage(
                content='[{"content":"上一轮证据","source":"old.md","score":0.9}]',
                name="vector_search",
                tool_call_id="old-call",
            ),
            AIMessage(content="上一轮答案"),
            HumanMessage(content="这一轮问题"),
            ToolMessage(
                content='[{"content":"这一轮证据","source":"new.md","score":0.8}]',
                name="vector_search",
                tool_call_id="new-call",
            ),
            AIMessage(content="这一轮答案"),
        ]

        result = agent._parse_agent_result("这一轮问题", {"messages": messages})

        assert [context.source for context in result.contexts] == ["new.md"]

    def test_missing_turn_anchor_does_not_fall_back_to_history(self):
        """本轮异常时不能把旧答案、旧证据或旧轨迹伪装成当前结果。"""
        agent = object.__new__(QAAgent)
        messages = [
            HumanMessage(content="上一轮问题"),
            ToolMessage(
                content='[{"content":"上一轮证据","source":"old.md","score":0.9}]',
                name="vector_search",
                tool_call_id="old-call",
            ),
            AIMessage(content="上一轮答案"),
        ]

        result = agent._parse_agent_result("本轮未完成的问题", {"messages": messages})

        assert result.answer == ""
        assert result.contexts == []
        assert result.reasoning_steps == ["生成最终答案"]


class TestAnswerInvocation:
    @pytest.mark.asyncio
    async def test_passes_thread_id_to_react_agent(self):
        """会话 ID 必须传入唯一的 ReAct Agent checkpoint。"""
        agent = object.__new__(QAAgent)
        agent._agent = MagicMock()
        agent._agent.ainvoke = AsyncMock(return_value={
            "messages": [
                HumanMessage(content="张三是谁？"),
                AIMessage(content="张三是员工。"),
            ],
        })

        await agent.answer("张三是谁？", thread_id="conversation-42")

        call = agent._agent.ainvoke.await_args
        assert call.kwargs["config"]["configurable"]["thread_id"] == "conversation-42"
        assert call.kwargs["config"]["configurable"]["turn_id"]
        assert call.kwargs["config"]["recursion_limit"] == 10

    @pytest.mark.asyncio
    async def test_missing_thread_id_creates_isolated_session(self):
        agent = object.__new__(QAAgent)
        agent._agent = MagicMock()
        agent._agent.ainvoke = AsyncMock(return_value={"messages": []})

        await agent.answer("第一个问题")
        await agent.answer("第二个问题")

        first_config = agent._agent.ainvoke.await_args_list[0].kwargs["config"]
        second_config = agent._agent.ainvoke.await_args_list[1].kwargs["config"]
        assert first_config["configurable"]["thread_id"] != second_config["configurable"]["thread_id"]
        assert first_config["configurable"]["turn_id"] != second_config["configurable"]["turn_id"]
