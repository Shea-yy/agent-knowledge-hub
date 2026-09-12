"""
AOP 拦截器单元测试 — core/aop_interceptors.py

纯逻辑测试，不依赖 LLM / 数据库。
"""

from __future__ import annotations

import pytest

from core.aop_interceptors import break_react_loop, require_readonly_cypher


@pytest.fixture(autouse=True)
def _reset_streak():
    """熔断器状态是模块级全局的，每个用例前清零，避免用例间串扰"""
    import core.aop_interceptors as aop
    aop._streak.clear()
    yield


# ═══════════════════════════════════════════════════════════════
# require_readonly_cypher
# ═══════════════════════════════════════════════════════════════

@require_readonly_cypher
async def _readonly_tool(cypher: str = "") -> str:
    return "EXECUTED"


class TestCypherReadonly:
    def test_allows_match(self):
        assert _run(_readonly_tool, cypher="MATCH (n:Entity) RETURN n LIMIT 10") == "EXECUTED"

    def test_blocks_delete(self):
        result = _run(_readonly_tool, cypher="MATCH (n) DELETE n LIMIT 1")
        assert "拦截" in result

    def test_blocks_embedded_delete_in_match(self):
        """首关键字合法但嵌写操作: MATCH (n) DELETE n"""
        result = _run(_readonly_tool, cypher="MATCH (n:Entity) DELETE n LIMIT 1")
        assert "拦截" in result

    def test_blocks_embedded_set(self):
        result = _run(_readonly_tool, cypher="MATCH (n) SET n.name = 'hacked' LIMIT 1")
        assert "拦截" in result

    def test_blocks_merge(self):
        result = _run(_readonly_tool, cypher="MERGE (n:Entity {name: 'x'})")
        assert "拦截" in result

    def test_blocks_multi_statement(self):
        """分号走私: 首句合法但携带第二句写操作"""
        result = _run(_readonly_tool, cypher="MATCH (n) RETURN n LIMIT 1; MATCH (n) DELETE n")
        assert "拦截" in result

    def test_no_false_positive_on_string_literal(self):
        """字符串字面量含危险词不误伤（全文正则方案的缺陷）"""
        result = _run(_readonly_tool, cypher="MATCH (e) WHERE e.description CONTAINS 'delete' RETURN e LIMIT 10")
        assert result == "EXECUTED"

    def test_comments_before_match(self):
        """行注释/块注释后的 MATCH 正常放行"""
        result = _run(_readonly_tool, cypher="// 查询所有实体\nMATCH (n) RETURN n LIMIT 10")
        assert result == "EXECUTED"

    def test_blocks_procedure_calls_even_when_they_are_read_only(self):
        result = _run(_readonly_tool, cypher="CALL db.labels() YIELD label RETURN label LIMIT 10")
        assert "拦截" in result

    def test_requires_explicit_bounded_limit(self):
        assert "LIMIT" in _run(_readonly_tool, cypher="MATCH (n) RETURN n")
        assert "LIMIT" in _run(_readonly_tool, cypher="MATCH (n) RETURN n LIMIT 51")

    def test_accepts_limit_in_valid_range(self):
        assert _run(_readonly_tool, cypher="MATCH (n) RETURN n LIMIT 50") == "EXECUTED"


# ═══════════════════════════════════════════════════════════════
# break_react_loop
# ═══════════════════════════════════════════════════════════════

class TestReActBreaker:
    def test_success_content_with_marker_does_not_increment_streak(self):
        """“未找到”出现在成功文档内容中时，不应触发熔断。"""
        calls = {"n": 0}

        @break_react_loop(2)
        async def fake_search(config=None, **kw):
            calls["n"] += 1
            return '[{"content":"文档说明：未找到配置时应如何处理","source":"guide.md"}]'

        _run(fake_search)
        _run(fake_search)
        _run(fake_search)

        assert calls["n"] == 3

    def test_trips_after_two_no_results(self):
        calls = {"n": 0}

        @break_react_loop(2)
        async def fake_search(config=None, **kw):
            calls["n"] += 1
            return "未找到实体"

        assert "未找到" in _run(fake_search)
        assert "未找到" in _run(fake_search)
        # 第三次触发熔断，核心逻辑不再执行
        result = _run(fake_search)
        assert "系统强制干预" in result
        assert calls["n"] == 2

    def test_latch_holds_after_trip(self):
        calls = {"n": 0}

        @break_react_loop(2)
        async def fake_search(config=None, **kw):
            calls["n"] += 1
            return "找到 0 行"

        _run(fake_search)
        _run(fake_search)
        _run(fake_search)  # trip
        _run(fake_search)  # still blocked (latch)
        assert calls["n"] == 2

    def test_success_resets_streak(self):
        @break_react_loop(2)
        async def fake_search(config=None, **kw):
            return kw.get("reply", "未找到")

        _run(fake_search)                 # 无结果 → streak 1
        assert "OK" in _run(fake_search, reply="OK result")  # 成功 → streak 清零
        _run(fake_search)                 # 无结果 → streak 1（重新计）
        assert "未找到" in _run(fake_search)  # 第 2 次无结果仍放行（未熔断）

    def test_thread_isolation(self):
        """不同 thread_id 的计数互不影响"""
        @break_react_loop(1)
        async def fake_search(config=None, **kw):
            return "未找到实体"

        cfg_a = {"configurable": {"thread_id": "thread-A"}}
        cfg_b = {"configurable": {"thread_id": "thread-B"}}

        _run(fake_search, config=cfg_a)  # A: streak 1
        assert "系统强制干预" in _run(fake_search, config=cfg_a)  # A 熔断
        assert "未找到" in _run(fake_search, config=cfg_b)  # B 不受影响

    def test_new_turn_resets_latch_for_same_thread(self):
        """同一会话的新问题不能继承上一问已经熔断的工具状态。"""
        calls = {"n": 0}

        @break_react_loop(1)
        async def fake_search(config=None, **kw):
            calls["n"] += 1
            return "未找到实体"

        first_turn = {"configurable": {"thread_id": "thread-A", "turn_id": "turn-1"}}
        second_turn = {"configurable": {"thread_id": "thread-A", "turn_id": "turn-2"}}

        _run(fake_search, config=first_turn)
        assert "系统强制干预" in _run(fake_search, config=first_turn)
        assert "未找到" in _run(fake_search, config=second_turn)
        assert calls["n"] == 2


def _run(coro_func, **kwargs):
    """同步辅助：跑一次 async 工具调用（每次独立事件循环）"""
    import asyncio
    return asyncio.run(coro_func(**kwargs))
