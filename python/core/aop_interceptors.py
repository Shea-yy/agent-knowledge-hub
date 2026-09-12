"""
AOP 安全拦截器 — 企业级 Agent 系统的中央管控基建层

所有需要对接 LLM 的 Agent 工具，统一从这里 import 并打上对应切面。
安全策略一处定义，全局生效——Agent 业务代码不掺杂任何安全逻辑。

切面清单:
  1. require_readonly_cypher — 防注入/防删库: LLM 只能生成只读 Cypher
  2. break_react_loop        — 物理熔断: 工具连续无结果 N 次后强制干预，
                               兜底 ReAct 死循环（软约束失效时的最后防线）

设计原则:
  - 拦截 + 干预，而非抛异常——工具返回干预消息，LLM 读到后自我修正
  - 状态按 (thread_id, turn_id, tool_name) 维度隔离：会话记忆可跨轮保留，
    但一次 ReAct 执行的熔断状态绝不污染下一轮
"""

from __future__ import annotations

import json
import re
from functools import wraps

from cachetools import TTLCache
from loguru import logger

log = logger.bind(module="aop")


# ═══════════════════════════════════════════════════════════════
# 切面 1: Cypher 只读拦截
# ═══════════════════════════════════════════════════════════════

# 面向 LLM 的自由 Cypher 工具只开放一个可审计子集：MATCH ... RETURN ... LIMIT。
# CALL/SHOW/PROFILE 虽不一定写数据，但能够调用过程、暴露系统元数据或执行高代价查询，
# 不应作为未受信任模型的能力。
_READ_ONLY_CLAUSES = ("MATCH",)
CYPHER_MAX_LENGTH = 2_000
CYPHER_MAX_ROWS = 50

# 危险写操作关键字（在剥掉字符串字面量后检查）
_WRITE_KEYWORDS = re.compile(
    r"\b(DELETE|DETACH|SET|REMOVE|CREATE|MERGE|DROP)\b", re.IGNORECASE,
)


def _cypher_code(cypher: str) -> str:
    """移除注释和字面量后保留 Cypher 结构，用于关键字安全检查。"""
    code: list[str] = []
    index = 0
    quote: str | None = None
    length = len(cypher)

    while index < length:
        char = cypher[index]
        next_char = cypher[index + 1] if index + 1 < length else ""

        if quote:
            # Cypher 使用成对引号表示字符串中的引号；反斜杠转义也一并跳过。
            if char == "\\" and index + 1 < length:
                index += 2
                continue
            if char == quote and next_char == quote and quote != "`":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue

        if char in ("'", '"', "`"):
            quote = char
            code.append(" ")
            index += 1
            continue
        if char == "/" and next_char == "/":
            newline = cypher.find("\n", index + 2)
            index = length if newline == -1 else newline + 1
            code.append(" ")
            continue
        if char == "/" and next_char == "*":
            end = cypher.find("*/", index + 2)
            index = length if end == -1 else end + 2
            code.append(" ")
            continue

        code.append(char)
        index += 1

    return "".join(code)


def _first_clause(cypher: str) -> str:
    """提取第一条语句的首个关键字（去除注释、字面量和空白）。"""
    tokens = _cypher_code(cypher).strip().split()
    return tokens[0].upper() if tokens else ""


def _contains_write_op(cypher: str) -> bool:
    """
    检查是否含写操作关键字（先剥掉字符串字面量）。

    剥字面量是关键——全文搜危险词会误伤:
      MATCH (e) WHERE e.description CONTAINS 'delete' RETURN e
    剥掉 'delete' 后剩余部分无危险词，正确放行；
    而 MATCH (n) DELETE n 的 DELETE 是语法关键字，剥字面量后仍在，正确拦截。
    """
    return bool(_WRITE_KEYWORDS.search(_cypher_code(cypher)))


def _has_bounded_limit(cypher: str) -> bool:
    """要求显式、常量且不超过上限的 LIMIT，避免工具查询/返回无界数据。"""
    limits = [int(value) for value in re.findall(r"\bLIMIT\s+(\d+)\b", _cypher_code(cypher), re.I)]
    return bool(limits) and all(0 < value <= CYPHER_MAX_ROWS for value in limits)


def require_readonly_cypher(func):
    """
    拦截 LLM 生成的 Cypher：只放行只读语句。

    四层防御:
      1. 首关键字白名单 — 只允许 MATCH 开头，拒绝 CALL/SHOW/PROFILE 等越权能力
      2. 字面量剥离后的危险关键字检查 — 拦截 "MATCH (n) DELETE n"
         这类首关键字合法但嵌有写操作的语句，且不误伤字符串字面量
      3. 分号拦截 — 防多语句走私
      4. LIMIT 上限 — 要求显式 LIMIT 1..50，控制结果量和图遍历后返回量
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        cypher = kwargs.get("cypher") or (args[0] if args else "")

        if not isinstance(cypher, str) or not cypher.strip():
            return "系统安全拦截：Cypher 查询不能为空。"
        if len(cypher) > CYPHER_MAX_LENGTH:
            log.warning("拦截过长 Cypher ({} chars)", len(cypher))
            return (
                f"系统安全拦截：Cypher 长度不能超过 {CYPHER_MAX_LENGTH} 个字符。"
                "请缩小查询范围。"
            )
        if ";" in _cypher_code(cypher):
            log.warning("拦截多语句 Cypher: {}", cypher[:100])
            return (
                "系统安全拦截：检测到多语句查询（含分号），已被拒绝。"
                "请每次只提交一条 MATCH 查询。"
            )
        if _first_clause(cypher) not in _READ_ONLY_CLAUSES:
            log.warning("拦截非只读 Cypher: {}", cypher[:100])
            return (
                "系统安全拦截：只允许 MATCH 开头的只读 Cypher 查询。"
                "CALL、SHOW、PROFILE 和写操作均不可用。"
            )
        if _contains_write_op(cypher):
            log.warning("拦截嵌写操作 Cypher: {}", cypher[:100])
            return (
                "系统安全拦截：查询语句中包含写操作关键字（DELETE/SET/MERGE 等），"
                "已被拒绝。知识库查询只允许纯读操作。"
            )
        if not _has_bounded_limit(cypher):
            log.warning("拦截未受限 Cypher: {}", cypher[:100])
            return (
                f"系统安全拦截：查询必须包含 LIMIT 1..{CYPHER_MAX_ROWS}，"
                "以限制图查询的返回量。"
            )
        return await func(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════
# 切面 2: 无结果物理熔断
# ═══════════════════════════════════════════════════════════════

# (thread_id, turn_id, tool_name) → 连续无结果次数
# TTLCache: 最多 10000 个执行轮次状态，单条存活 1 小时自动蒸发。
# 不用 defaultdict——无界字典在生产中随 thread_id 增长无限膨胀 → OOM。
# 注: 内存态在多副本部署下不共享，企业级应迁移 Redis（见 CHANGES.md roadmap）。
_streak: TTLCache = TTLCache(maxsize=10000, ttl=3600)

# 纯文本旧工具的明确空结果；不能对整个成功内容做子串匹配。
_NO_RESULT_MESSAGES = frozenset(("未找到", "未找到实体", "未找到相关文档", "找到 0 行"))


def _is_no_result(result: object) -> bool:
    """只依据工具的结构化状态字段或完整的旧式状态消息判断空结果。"""
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return result.strip() in _NO_RESULT_MESSAGES

    if not isinstance(data, dict):
        return False
    if data.get("error"):
        return True

    message = data.get("message")
    if not isinstance(message, str):
        return False
    normalized = message.strip()
    return (
        normalized in _NO_RESULT_MESSAGES
        or normalized.startswith("未找到实体")
        or "找到 0 行" in normalized
    )


def break_react_loop(max_calls: int = 2):
    """
    熔断器：同一次 ReAct 执行内，工具连续无结果 max_calls 次后强制干预。

    设计要点:
      - 维度: (thread_id, turn_id, tool_name)，既隔离不同对话，也不让上一问的
        熔断 latch 影响同一会话的下一问
      - 计数: 只计"连续无结果"（streak）
      - 触发: latch 保持——熔断后该线程内不再执行该工具，持续返回干预指令
      - 内存: 成功调用即 pop 释放；TTLCache 兜底淘汰超时线程

    注: 工具函数需声明 config: RunnableConfig 参数，
        运行时由 LangChain 自动注入（含 thread_id、turn_id）。
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tool_name = func.__name__
            cfg = kwargs.get("config")
            configurable = cfg.get("configurable", {}) if cfg is not None else {}
            thread_id = str(configurable.get("thread_id") or "default")
            # 兼容旧调用方未传 turn_id 的情形：此类调用仍按 thread 聚合，
            # 新版 QAAgent 则会为每个用户问题生成新的 turn_id。
            turn_id = str(configurable.get("turn_id") or thread_id)
            key = (thread_id, turn_id, tool_name)

            current = _streak.get(key, 0)
            if current >= max_calls:
                log.warning("熔断触发: 线程={}, 轮次={}, 工具={}, 连续无结果={}",
                            thread_id, turn_id, tool_name, current)
                return (
                    f"系统强制干预：工具 {tool_name} 已连续 {current} 次无结果，"
                    f"请立即停止调用该工具，切换其他工具，或基于已有信息直接作答。"
                )

            result = await func(*args, **kwargs)
            if _is_no_result(result):
                _streak[key] = current + 1
                log.debug("工具 {} 连续无结果 +1 (streak={})", tool_name, _streak[key])
            else:
                _streak.pop(key, None)  # 成功：释放内存，而非只归零
            return result
        return wrapper
    return decorator
