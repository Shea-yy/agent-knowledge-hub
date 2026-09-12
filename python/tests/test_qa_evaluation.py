"""QA 金标评测器的纯逻辑测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.qa_agent import QAResult, QueryIntent, RetrievedContext
from core.qa_evaluation import evaluate_case, load_gold_cases, summarise


GOLD_CASES_PATH = Path(__file__).with_name("qa_gold_cases.json")


def _result(
    *,
    answer: str,
    source: str,
    intent: QueryIntent = QueryIntent.FACTOID,
    confidence: float = 0.9,
) -> QAResult:
    return QAResult(
        question="测试问题",
        answer=answer,
        contexts=[RetrievedContext(
            content="可追溯的原文片段",
            source=source,
            score=confidence,
            retrieval_type="vector",
        )],
        intent=intent,
        confidence=confidence,
    )


def test_load_gold_cases_has_unique_auditable_cases():
    cases = load_gold_cases(GOLD_CASES_PATH)

    assert len(cases) == 3
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.required_answer_terms and case.required_sources for case in cases)


def test_evaluate_case_requires_answer_source_citation_intent_and_confidence():
    case = load_gold_cases(GOLD_CASES_PATH)[0]
    result = _result(
        answer="张三在腾讯工作。[来源: 员工名录.csv]",
        source="员工名录.csv",
    )

    evaluation = evaluate_case(case, result)

    assert evaluation.passed
    assert evaluation.answer_term_recall == 1.0
    assert evaluation.source_recall == 1.0
    assert evaluation.citation_recall == 1.0


def test_evaluate_case_exposes_precise_regression_reason():
    case = load_gold_cases(GOLD_CASES_PATH)[0]
    result = _result(answer="张三在其他公司工作。", source="错误来源.md", confidence=0.5)

    evaluation = evaluate_case(case, result)

    assert not evaluation.passed
    assert "腾讯" in evaluation.missing_answer_terms
    assert evaluation.missing_sources == ("员工名录.csv",)
    assert evaluation.missing_citations == ("员工名录.csv",)
    assert not evaluation.confidence_passed


def test_rejects_invalid_gold_dataset(tmp_path):
    invalid_cases = tmp_path / "invalid.json"
    invalid_cases.write_text('[{"id": "only-id"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="缺少字段"):
        load_gold_cases(invalid_cases)


def test_summary_rejects_empty_evaluation_set():
    with pytest.raises(ValueError, match="至少需要一个"):
        summarise([])
