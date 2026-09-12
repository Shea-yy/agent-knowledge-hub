"""可重复运行的 QA 金标评测：只比较可审计的输出契约，不使用 LLM-as-a-judge。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


class QAResultLike(Protocol):
    """评测器需要的最小 QA 输出接口，避免耦合具体 Agent 实现。"""

    answer: str
    contexts: Sequence[Any]
    intent: Any
    confidence: float


class QAAgentLike(Protocol):
    async def answer(self, question: str, thread_id: str | None = None) -> QAResultLike: ...


@dataclass(frozen=True)
class QAGoldCase:
    """一个人工维护的、可审计的问答金标样本。"""

    case_id: str
    question: str
    required_answer_terms: tuple[str, ...]
    required_sources: tuple[str, ...]
    required_citations: tuple[str, ...]
    expected_intent: str
    min_confidence: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "QAGoldCase":
        required = {
            "id", "question", "required_answer_terms", "required_sources",
            "required_citations", "expected_intent", "min_confidence",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"金标样本缺少字段: {', '.join(sorted(missing))}")

        case_id = str(data["id"]).strip()
        question = str(data["question"]).strip()
        if not case_id or not question:
            raise ValueError("金标样本的 id 和 question 不能为空")

        def strings(key: str) -> tuple[str, ...]:
            value = data[key]
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError(f"金标样本 {case_id} 的 {key} 必须是非空字符串列表")
            return tuple(item.strip() for item in value)

        try:
            min_confidence = float(data["min_confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"金标样本 {case_id} 的 min_confidence 非法") from exc
        if not 0 <= min_confidence <= 1:
            raise ValueError(f"金标样本 {case_id} 的 min_confidence 必须在 0 到 1 之间")

        return cls(
            case_id=case_id,
            question=question,
            required_answer_terms=strings("required_answer_terms"),
            required_sources=strings("required_sources"),
            required_citations=strings("required_citations"),
            expected_intent=str(data["expected_intent"]).strip(),
            min_confidence=min_confidence,
        )


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    answer_term_recall: float
    source_recall: float
    citation_recall: float
    intent_match: bool
    confidence_passed: bool
    passed: bool
    missing_answer_terms: tuple[str, ...]
    missing_sources: tuple[str, ...]
    missing_citations: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    passed: int
    pass_rate: float
    average_answer_term_recall: float
    average_source_recall: float
    average_citation_recall: float
    cases: tuple[CaseEvaluation, ...]


def load_gold_cases(path: str | Path) -> tuple[QAGoldCase, ...]:
    """读取 JSON 金标集，并在加载阶段拒绝重复 case id 或损坏结构。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 QA 金标集: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("QA 金标集必须是非空 JSON 数组")

    cases = tuple(QAGoldCase.from_mapping(item) for item in raw if isinstance(item, dict))
    if len(cases) != len(raw):
        raise ValueError("QA 金标集的每项都必须是对象")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("QA 金标集包含重复 id")
    return cases


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _recall(expected: Iterable[str], actual: str | set[str]) -> tuple[float, tuple[str, ...]]:
    expected_values = tuple(expected)
    if isinstance(actual, str):
        normalized_actual = _normalise(actual)
        missing = tuple(item for item in expected_values if _normalise(item) not in normalized_actual)
    else:
        normalized_actual = {_normalise(item) for item in actual}
        missing = tuple(item for item in expected_values if _normalise(item) not in normalized_actual)
    return (len(expected_values) - len(missing)) / len(expected_values), missing


def evaluate_case(case: QAGoldCase, result: QAResultLike) -> CaseEvaluation:
    """从答案和结构化 contexts 计算可复现的通过/失败原因。"""
    answer = result.answer or ""
    answer_recall, missing_terms = _recall(case.required_answer_terms, answer)
    source_values = {
        str(getattr(context, "source", ""))
        for context in result.contexts
        if getattr(context, "source", "")
    }
    source_recall, missing_sources = _recall(case.required_sources, source_values)
    citation_markers = [f"[来源: {citation}]" for citation in case.required_citations]
    citation_recall, missing_markers = _recall(citation_markers, answer)
    missing_citations = tuple(
        marker.removeprefix("[来源: ").removesuffix("]") for marker in missing_markers
    )
    actual_intent = str(getattr(result.intent, "value", result.intent))
    intent_match = actual_intent == case.expected_intent
    confidence_passed = result.confidence >= case.min_confidence
    passed = (
        answer_recall == 1.0
        and source_recall == 1.0
        and citation_recall == 1.0
        and intent_match
        and confidence_passed
    )
    return CaseEvaluation(
        case_id=case.case_id,
        answer_term_recall=answer_recall,
        source_recall=source_recall,
        citation_recall=citation_recall,
        intent_match=intent_match,
        confidence_passed=confidence_passed,
        passed=passed,
        missing_answer_terms=missing_terms,
        missing_sources=missing_sources,
        missing_citations=missing_citations,
    )


def summarise(evaluations: Sequence[CaseEvaluation]) -> EvaluationSummary:
    """汇总一批 case；空输入是调用错误，避免把 0/0 伪装成好成绩。"""
    if not evaluations:
        raise ValueError("至少需要一个评测结果")
    total = len(evaluations)
    return EvaluationSummary(
        total=total,
        passed=sum(item.passed for item in evaluations),
        pass_rate=sum(item.passed for item in evaluations) / total,
        average_answer_term_recall=sum(item.answer_term_recall for item in evaluations) / total,
        average_source_recall=sum(item.source_recall for item in evaluations) / total,
        average_citation_recall=sum(item.citation_recall for item in evaluations) / total,
        cases=tuple(evaluations),
    )


async def evaluate_agent(
    agent: QAAgentLike,
    cases: Sequence[QAGoldCase],
    *,
    thread_prefix: str = "gold-eval",
) -> EvaluationSummary:
    """运行实际 Agent 的可选入口；测试中可传入脚本化 Agent 保持完全确定性。"""
    evaluations = []
    for case in cases:
        result = await agent.answer(case.question, thread_id=f"{thread_prefix}-{case.case_id}")
        evaluations.append(evaluate_case(case, result))
    return summarise(evaluations)
