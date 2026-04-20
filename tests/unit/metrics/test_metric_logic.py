import pytest

from evalf.llms.base import BaseLLMModel
from evalf.metrics.answer_correctness.prompt import AnswerCorrectnessPrompt
from evalf.metrics.answer_relevance import AnswerRelevanceMetric
from evalf.metrics.answer_relevance.prompt import AnswerRelevancePrompt
from evalf.metrics.answer_relevance.schema import AnswerRelevanceAssessment
from evalf.metrics.c4 import C4Metric
from evalf.metrics.c4.schema import C4Assessment, C4SummaryReason, CriterionAssessment
from evalf.metrics.context_coverage import ContextCoverageMetric
from evalf.metrics.context_coverage.prompt import ContextCoveragePrompt
from evalf.metrics.context_coverage.schema import ContextCoverageAssessment as CoverageAssessment
from evalf.metrics.context_precision import ContextPrecisionMetric
from evalf.metrics.context_precision.prompt import ContextPrecisionCoveragePrompt
from evalf.metrics.context_recall import ContextRecallMetric
from evalf.metrics.context_recall.prompt import (
    ContextRecallCoveragePrompt,
    ReferenceClaimExtractionPrompt,
)
from evalf.metrics.context_relevance import ContextRelevanceMetric
from evalf.metrics.context_relevance.prompt import ContextRelevancePrompt
from evalf.metrics.decomposition import (
    Claim,
    ClaimCoverageAssessment,
    ClaimCoverageVerdict,
    ClaimExtraction,
    ClaimSupportAssessment,
    ClaimSupportVerdict,
    ContextCoverageAssessment,
    ContextCoverageVerdict,
    ContextRelevanceVerdict,
    ContextRelevanceVerdictList,
)
from evalf.metrics.faithfulness import FaithfulnessMetric
from evalf.metrics.faithfulness.prompt import (
    FaithfulnessClaimExtractionPrompt,
    FaithfulnessClaimVerificationPrompt,
)
from evalf.reporting import build_run_summary
from evalf.schemas import EvalAttempt, EvalCase, LLMResponse, MetricResult, SampleResult, UsageStats

pytestmark = pytest.mark.unit


class FakeLLM(BaseLLMModel):
    def __init__(self, outputs):
        super().__init__(
            provider="test",
            model="fake-model",
            base_url="https://example.com",
            api_key=None,
        )
        self._outputs = list(outputs)

    def generate(self, **kwargs) -> LLMResponse:
        if not self._outputs:
            raise AssertionError("No fake outputs left for generate().")
        parsed_output = self._outputs.pop(0)
        return LLMResponse(
            text=None,
            model=self.model,
            provider=self.provider,
            parsed_output=parsed_output,
            usage=UsageStats.empty(),
        )

    async def a_generate(self, **kwargs) -> LLMResponse:
        return self.generate(**kwargs)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_out_of_range_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        AnswerRelevanceMetric(threshold=threshold)


def test_metric_pass_rates_ignore_errors() -> None:
    summary = build_run_summary(
        [
            SampleResult(
                sample_id="case-1",
                status="passed",
                metrics=[
                    MetricResult(
                        name="faithfulness",
                        status="passed",
                        score=1.0,
                        threshold=0.7,
                        passed=True,
                    )
                ],
            ),
            SampleResult(
                sample_id="case-2",
                status="failed",
                metrics=[
                    MetricResult(
                        name="faithfulness",
                        status="error",
                        score=None,
                        threshold=0.7,
                        passed=False,
                        error="timeout",
                    ),
                    MetricResult(
                        name="executor",
                        status="error",
                        score=None,
                        threshold=0.0,
                        passed=False,
                        error="timeout",
                    ),
                ],
            ),
        ]
    )

    assert summary.metric_pass_rates == {"faithfulness": 1.0}
    assert "executor" not in summary.metric_pass_rates


def test_pass_caret_k_with_skipped_attempt_stays_skipped() -> None:
    metric = AnswerRelevanceMetric(threshold=0.7, mode="pass^k", k=2)
    llm = FakeLLM([AnswerRelevanceAssessment(score=1.0, reason="Direct answer.")])
    case = EvalCase(
        id="case-1",
        question="What is FERPA?",
        attempts=[
            EvalAttempt(actual_output="FERPA is a privacy law."),
            EvalAttempt(),
        ],
    )

    result = metric.measure(case, llm)

    assert result.status == "skipped"
    assert result.passed is None
    assert result.score is None
    assert result.evaluated_k == 1
    assert result.missing_inputs == ["actual_output"]
    assert result.reason is not None
    assert "Only 1/2 attempt(s) were evaluated" in result.reason
    assert "pass^k requires every requested attempt to be evaluated and pass" in result.reason


def test_scalar_prompt_examples_follow_declared_score_bands() -> None:
    allowed_scores = {0.0, 0.25, 0.5, 0.75, 1.0}

    for prompt in (AnswerCorrectnessPrompt(), AnswerRelevancePrompt(), ContextCoveragePrompt()):
        for _, output in prompt.examples:
            assert output.score in allowed_scores


def test_decomposition_prompt_examples_use_expected_ids() -> None:
    for prompt in (
        FaithfulnessClaimExtractionPrompt(),
        FaithfulnessClaimVerificationPrompt(),
        ReferenceClaimExtractionPrompt(),
        ContextRecallCoveragePrompt(),
        ContextPrecisionCoveragePrompt(),
        ContextRelevancePrompt(),
    ):
        assert prompt.examples

    extraction_prompt = FaithfulnessClaimExtractionPrompt()
    extraction_output = extraction_prompt.examples[0][1]
    assert [claim.claim_id for claim in extraction_output.claims] == ["c1", "c2"]

    recall_extraction_prompt = ReferenceClaimExtractionPrompt()
    reference_output = recall_extraction_prompt.examples[0][1]
    assert [claim.claim_id for claim in reference_output.claims] == ["rc1", "rc2"]

    relevance_prompt = ContextRelevancePrompt()
    relevance_output = relevance_prompt.examples[0][1]
    assert [verdict.context_id for verdict in relevance_output.verdicts] == ["ctx_1", "ctx_2"]


def test_blank_required_inputs_are_treated_as_missing() -> None:
    answer_metric = AnswerRelevanceMetric(threshold=0.7)
    answer_result = answer_metric.measure(
        EvalCase(question="What is FERPA?", actual_output="   "),
        FakeLLM([]),
    )
    context_metric = ContextRelevanceMetric(threshold=0.7)
    context_result = context_metric.measure(
        EvalCase(question="What is FERPA?", retrieved_contexts=[]),
        FakeLLM([]),
    )

    assert answer_result.status == "skipped"
    assert answer_result.missing_inputs == ["actual_output"]
    assert context_result.status == "skipped"
    assert context_result.missing_inputs == ["retrieved_contexts"]


def test_c4_metric_averages_four_criteria_and_collapses_reasoning() -> None:
    metric = C4Metric(threshold=0.7)
    llm = FakeLLM(
        [
            C4Assessment(
                alignment_integrity=CriterionAssessment(
                    score=1.0,
                    reasoning="Bám đúng câu hỏi và không nhầm thương hiệu.",
                ),
                accuracy_consistency=CriterionAssessment(
                    score=0.75,
                    reasoning="Đúng phần lớn nội dung nhưng còn thiếu một chi tiết nhỏ.",
                ),
                safety_sovereignty_tone=CriterionAssessment(
                    score=1.0,
                    reasoning="Giọng điệu an toàn, tôn trọng và phù hợp.",
                ),
                completeness_coverage=CriterionAssessment(
                    score=0.5,
                    reasoning="Thiếu một bước hướng dẫn nên độ đầy đủ chưa cao.",
                ),
            )
        ]
    )

    result = metric.measure(
        EvalCase(
            question="VinFast VF 8 có hỗ trợ OTA không?",
            expected_output="VF 8 hỗ trợ OTA và có thể kiểm tra phiên bản trong phần cài đặt.",
            actual_output="VF 8 hỗ trợ OTA, bạn có thể kiểm tra trong phần cài đặt xe.",
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == pytest.approx(0.8125)
    assert result.reason is not None
    assert "alignment_integrity=1.00" in result.reason
    assert "completeness_coverage=0.50" in result.reason


def test_c4_metric_can_generate_summary_reason_and_apply_strict_mode() -> None:
    metric = C4Metric(threshold=0.8, need_summary_reason=True, strict_mode=True)
    llm = FakeLLM(
        [
            C4Assessment(
                alignment_integrity=CriterionAssessment(
                    score=0.75,
                    reasoning="Có bám đúng chủ đề nhưng còn diễn đạt hơi lệch intent.",
                ),
                accuracy_consistency=CriterionAssessment(
                    score=0.75,
                    reasoning="Nội dung tương đối đúng nhưng chưa khớp hoàn toàn expected output.",
                ),
                safety_sovereignty_tone=CriterionAssessment(
                    score=1.0,
                    reasoning="Ngôn từ an toàn và tôn trọng.",
                ),
                completeness_coverage=CriterionAssessment(
                    score=0.5,
                    reasoning="Thiếu thông tin cần thiết để hoàn thành tác vụ.",
                ),
            ),
            C4SummaryReason(
                reason="Điểm bị kéo xuống vì câu trả lời chưa đủ đầy và chưa khớp hoàn toàn với đáp án kỳ vọng."
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="Cách kiểm tra phiên bản phần mềm trên xe VinFast?",
            expected_output="Người dùng có thể vào phần cài đặt hệ thống để kiểm tra phiên bản phần mềm hiện tại.",
            actual_output="Bạn có thể kiểm tra trong phần cài đặt, nhưng câu trả lời chưa nêu rõ mục hệ thống.",
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == 0.0
    assert (
        result.reason
        == "Điểm bị kéo xuống vì câu trả lời chưa đủ đầy và chưa khớp hoàn toàn với đáp án kỳ vọng."
    )


def test_faithfulness_metric_scores_atomic_claim_verdicts() -> None:
    metric = FaithfulnessMetric(threshold=0.7)
    llm = FakeLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="c1", text="FERPA rights transfer at age 18."),
                    Claim(
                        claim_id="c2",
                        text="FERPA rights transfer only when the student turns 21.",
                    ),
                ]
            ),
            ClaimSupportAssessment(
                verdicts=[
                    ClaimSupportVerdict(
                        claim_id="c1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Supported by the FERPA context.",
                    ),
                    ClaimSupportVerdict(
                        claim_id="c2",
                        verdict="contradicted",
                        evidence_context_ids=["ctx_1"],
                        reason="The FERPA context says age 18, not 21.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=["FERPA rights transfer when a student turns 18 years old."],
            actual_output=("FERPA rights transfer at age 18, and only when the student turns 21."),
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.25)
    assert result.trial_results[0].reason is not None
    assert "Supported 1/2 material claim(s)." in result.trial_results[0].reason
    assert "extra 0.5 penalty" in result.trial_results[0].reason
    assert "Contradicted" in result.trial_results[0].reason


def test_faithfulness_metric_returns_perfect_score_for_no_material_claims() -> None:
    metric = FaithfulnessMetric(threshold=0.7)
    result = metric.measure(
        EvalCase(
            question="What is the answer?",
            retrieved_contexts=["The source does not contain enough information."],
            actual_output="I do not know based on the provided information.",
        ),
        FakeLLM([ClaimExtraction(claims=[])]),
    )

    assert result.status == "passed"
    assert result.score == 1.0
    assert (
        result.trial_results[0].reason
        == "Vacuous pass: the answer does not contain material factual claims to verify."
    )
    assert result.reason == "Passed in 1/1 evaluated attempt(s) under pass@k."


def test_context_recall_metric_scores_reference_claim_coverage() -> None:
    metric = ContextRecallMetric(threshold=0.7)
    llm = FakeLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="rc1", text="A service animal under the ADA is a dog."),
                    Claim(
                        claim_id="rc2",
                        text="The dog must be individually trained to perform tasks.",
                    ),
                ]
            ),
            ClaimCoverageAssessment(
                verdicts=[
                    ClaimCoverageVerdict(
                        claim_id="rc1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Covered by the retrieved context.",
                    ),
                    ClaimCoverageVerdict(
                        claim_id="rc2",
                        verdict="unsupported",
                        evidence_context_ids=[],
                        reason="Training requirement is missing.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="What qualifies as a service animal under the ADA?",
            retrieved_contexts=["Under the ADA, a service animal is a dog."],
            reference_contexts=[
                "Under the ADA, a service animal is a dog.",
                "The dog must be individually trained to perform tasks.",
            ],
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert result.trial_results[0].reason is not None
    assert "cover 1/2 reference claim(s)" in result.trial_results[0].reason
    assert "Missing" in result.trial_results[0].reason


def test_context_precision_metric_uses_rank_weighted_claim_discovery() -> None:
    metric = ContextPrecisionMetric(threshold=0.9)
    llm = FakeLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="rc1", text="FERPA rights transfer at age 18."),
                    Claim(
                        claim_id="rc2",
                        text="FERPA rights also transfer when entering postsecondary education.",
                    ),
                ]
            ),
            ContextCoverageAssessment(
                contexts=[
                    ContextCoverageVerdict(
                        context_id="ctx_1",
                        supported_claim_ids=["rc1"],
                        reason="Covers the age-18 rule.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_2",
                        supported_claim_ids=[],
                        reason="Unrelated context.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_3",
                        supported_claim_ids=["rc2"],
                        reason="Covers the postsecondary rule.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=[
                "FERPA rights transfer when a student turns 18.",
                "The ADA defines a service animal as a trained dog.",
                "FERPA rights also transfer when entering a postsecondary institution.",
            ],
            reference_contexts=[
                "FERPA rights transfer when a student turns 18.",
                "FERPA rights also transfer when entering a postsecondary institution.",
            ],
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(5 / 6)
    assert result.trial_results[0].reason is not None
    assert "Useful ranks: 1, 3." in result.trial_results[0].reason
    assert "Noisy ranks: 2." in result.trial_results[0].reason


def test_context_precision_metric_rejects_unknown_claim_ids() -> None:
    metric = ContextPrecisionMetric(threshold=0.7)
    llm = FakeLLM(
        [
            ClaimExtraction(
                claims=[Claim(claim_id="rc1", text="FERPA rights transfer at age 18.")]
            ),
            ContextCoverageAssessment(
                contexts=[
                    ContextCoverageVerdict(
                        context_id="ctx_1",
                        supported_claim_ids=["rc9"],
                        reason="Unknown claim id from judge.",
                    )
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=["FERPA rights transfer when a student turns 18."],
            reference_contexts=["FERPA rights transfer when a student turns 18."],
        ),
        llm,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "unknown reference claim ids" in result.error.lower()


def test_context_coverage_metric_scores_single_call_assessment() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    llm = FakeLLM(
        [
            CoverageAssessment(
                score=0.75,
                verdict="yes",
                reason="Covers both key conditions but misses minor detail.",
            )
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=[
                "FERPA rights transfer when a student turns 18 years old.",
                "FERPA rights also transfer when entering a postsecondary institution.",
            ],
            reference_contexts=[
                "FERPA rights transfer when a student turns 18 years old "
                "or enters a postsecondary institution at any age.",
            ],
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == pytest.approx(0.75)
    assert result.trial_results[0].reason is not None
    assert "key conditions" in result.trial_results[0].reason


def test_context_coverage_strict_mode_clamps_below_threshold() -> None:
    metric = ContextCoverageMetric(threshold=0.7, strict_mode=True)
    llm = FakeLLM(
        [CoverageAssessment(score=0.5, verdict="no", reason="Missing key facts.")]
    )

    result = metric.measure(
        EvalCase(
            question="What qualifies as a service animal under the ADA?",
            retrieved_contexts=["Under the ADA, a service animal is a dog."],
            reference_contexts=[
                "Under the ADA, a service animal is a dog.",
                "The dog must be individually trained to perform tasks.",
            ],
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == 0.0


def test_context_relevance_metric_averages_per_context_verdicts() -> None:
    metric = ContextRelevanceMetric(threshold=0.7)
    result = metric.measure(
        EvalCase(
            question="What qualifies as a service animal under the ADA?",
            retrieved_contexts=[
                "Under the ADA, a service animal is a dog trained to perform tasks.",
                "FERPA protects education records.",
                "The ADA is a federal civil rights law.",
            ],
        ),
        FakeLLM(
            [
                ContextRelevanceVerdictList(
                    verdicts=[
                        ContextRelevanceVerdict(
                            context_id="ctx_1",
                            verdict="relevant",
                            reason="Directly defines the service animal requirement.",
                        ),
                        ContextRelevanceVerdict(
                            context_id="ctx_2",
                            verdict="irrelevant",
                            reason="FERPA is unrelated to the ADA service-animal question.",
                        ),
                        ContextRelevanceVerdict(
                            context_id="ctx_3",
                            verdict="partially_relevant",
                            reason="It is related to the ADA but does not answer the question.",
                        ),
                    ]
                )
            ]
        ),
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert result.trial_results[0].reason is not None
    assert "Relevant ranks: 1." in result.trial_results[0].reason
    assert "Partial ranks: 3." in result.trial_results[0].reason
    assert "Irrelevant ranks: 2." in result.trial_results[0].reason
