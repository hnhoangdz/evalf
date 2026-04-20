import asyncio

import pytest

from evalf.metrics.c4 import C4Metric
from evalf.metrics.c4.schema import C4Assessment, C4SummaryReason, CriterionAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_c4_metric_skips_when_required_inputs_are_missing() -> None:
    metric = C4Metric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["actual_output", "expected_output", "question"]


def test_c4_metric_averages_four_criteria_for_single_attempt() -> None:
    metric = C4Metric(threshold=0.7)
    llm = SequenceLLM(
        [
            C4Assessment(
                alignment_integrity=CriterionAssessment(
                    score=1.0,
                    reasoning="Bám đúng câu hỏi và không nhầm thực thể.",
                ),
                accuracy_consistency=CriterionAssessment(
                    score=0.75,
                    reasoning="Đúng phần lớn nội dung nhưng còn thiếu một chi tiết nhỏ.",
                ),
                safety_sovereignty_tone=CriterionAssessment(
                    score=1.0,
                    reasoning="Giọng điệu an toàn và phù hợp.",
                ),
                completeness_coverage=CriterionAssessment(
                    score=0.5,
                    reasoning="Thiếu một bước hướng dẫn nên chưa đủ đầy.",
                ),
            )
        ]
    )

    result = metric.measure(
        EvalCase(
            question="VinFast VF 8 có hỗ trợ OTA không?",
            expected_output="VF 8 hỗ trợ OTA và người dùng có thể kiểm tra phiên bản trong phần cài đặt.",
            actual_output="VF 8 hỗ trợ OTA, bạn có thể kiểm tra trong phần cài đặt xe.",
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == pytest.approx(0.8125)
    assert result.reason is not None
    assert "alignment_integrity=1.00" in result.reason


def test_c4_metric_async_strict_mode_can_generate_summary_reason() -> None:
    metric = C4Metric(threshold=0.8, need_summary_reason=True, strict_mode=True)
    llm = SequenceLLM(
        [
            C4Assessment(
                alignment_integrity=CriterionAssessment(score=0.75, reasoning="Mostly aligned."),
                accuracy_consistency=CriterionAssessment(score=0.75, reasoning="Mostly accurate."),
                safety_sovereignty_tone=CriterionAssessment(score=1.0, reasoning="Safe tone."),
                completeness_coverage=CriterionAssessment(score=0.5, reasoning="Missing one detail."),
            ),
            C4SummaryReason(reason="The answer misses a key detail, so strict mode clamps the score."),
        ]
    )

    result = asyncio.run(
        metric.a_measure(
            EvalCase(
                question="How do I check the software version on a VinFast vehicle?",
                expected_output="Open system settings to review the installed software version.",
                actual_output="Check settings, but the answer misses the system-settings detail.",
            ),
            llm,
        )
    )

    assert result.status == "failed"
    assert result.score == 0.0
    assert result.reason == "The answer misses a key detail, so strict mode clamps the score."


def test_c4_metric_async_can_omit_reason_output() -> None:
    metric = C4Metric(threshold=0.7, include_reason=False)
    llm = SequenceLLM(
        [
            C4Assessment(
                alignment_integrity=CriterionAssessment(score=1.0, reasoning="On topic."),
                accuracy_consistency=CriterionAssessment(score=1.0, reasoning="Accurate."),
                safety_sovereignty_tone=CriterionAssessment(score=1.0, reasoning="Safe."),
                completeness_coverage=CriterionAssessment(score=0.75, reasoning="Mostly complete."),
            )
        ]
    )

    result = asyncio.run(
        metric.a_measure(
            EvalCase(
                question="Does the vehicle support OTA updates?",
                expected_output="Yes, the vehicle supports OTA updates.",
                actual_output="Yes, it supports OTA updates.",
            ),
            llm,
        )
    )

    assert result.status == "passed"
    assert result.trial_results[0].reason is None
