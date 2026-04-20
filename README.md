# evalf

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)

Evaluate RAG systematically with LLM-as-a-Judge from the CLI or Python.

`evalf` scores answers against expected outputs and contexts, aggregates results across samples, and writes JSON or Markdown reports with scores, pass/fail status, token usage, latency, and estimated cost.

## How it works

`evalf` takes one sample or a dataset of samples, sends structured judging prompts to an OpenAI-compatible model, and computes metric scores such as:

- answer correctness
- answer relevance
- faithfulness
- context coverage
- context relevance
- context precision
- context recall
- `c4`, a one-call composite metric that scores:
  - `alignment_integrity`
  - `accuracy_consistency`
  - `safety_sovereignty_tone`
  - `completeness_coverage`

It supports single-attempt and multi-attempt evaluation through `pass@k` and `pass^k`.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) if you do not already have it.

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Install `evalf`:

```bash
uv tool install evalf
# or
pip install evalf
# or inside an application
uv add evalf
```

If you prefer to work from a source checkout instead of installing from PyPI:

```bash
git clone https://github.com/hnhoangdz/evalf.git
cd evalf
uv sync --extra dev
```

3. Configure your judge model:

```bash
cp .env.example .env.local
```

Example:

```env
EVALF_PROVIDER=openai
EVALF_MODEL=gpt-4.1-mini
EVALF_BASE_URL=https://api.openai.com/v1
EVALF_API_KEY=your-api-key-here
EVALF_CONCURRENCY=4
EVALF_REQUEST_TIMEOUT_SECONDS=60
EVALF_PER_SAMPLE_TIMEOUT_SECONDS=120
EVALF_MAX_RETRIES=3
```

`evalf` loads `.env.local` first, then `.env`.

4. Check available metrics:

```bash
evalf list-metrics
```

## Usage

### Evaluate a single sample

```bash
evalf run \
  --provider openai \
  --model gpt-4.1-mini \
  --request-timeout-seconds 60 \
  --per-sample-timeout-seconds 120 \
  --question "Under FERPA, when do rights transfer from parents to a student?" \
  --retrieved-context "When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student." \
  --actual-output "Under FERPA, rights transfer when a student turns 18 or enters a postsecondary institution at any age." \
  --expected-output "Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age." \
  --metrics faithfulness,answer_correctness,answer_relevance \
  --threshold 0.7
```

### Evaluate inline JSON

```bash
evalf run \
  --sample-json '{"id":"case-1","question":"Under FERPA, when do rights transfer from parents to a student?","retrieved_contexts":["When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."],"reference_contexts":["When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."],"actual_output":"Under FERPA, rights transfer when a student turns 18 or enters a postsecondary institution at any age.","expected_output":"Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."}'
```

### Evaluate a file

```bash
evalf run \
  --input examples/rag_eval.jsonl \
  --metrics faithfulness,answer_correctness,context_precision,context_recall \
  --threshold 0.8 \
  --output .evalf/report.json
```

If `--output` has no suffix, `evalf` writes `<path>.json`.

### Evaluate with `c4`

```bash
evalf run \
  --input examples/rag_eval.jsonl \
  --metrics c4 \
  --threshold 0.7
```

Optional C4 flags:

- `--c4-summary-reason` to request a synthesized overall reason
- `--no-c4-include-reason` to omit criterion-level reasoning from the final report
- `--c4-strict-mode` to clamp below-threshold C4 scores to `0.0`

### Evaluate multi-attempt samples

Use `pass@k` when a sample passes if any attempt passes. Use `pass^k` when all evaluated attempts must pass.

```bash
evalf run \
  --input examples/rag_eval_attempts.json \
  --metrics faithfulness,answer_correctness \
  --metric-mode pass@k \
  --k 3
```

## Python API

```python
from evalf import EvalCase, Evaluator
from evalf.metrics import AnswerCorrectnessMetric, FaithfulnessMetric
from evalf.llms import build_llm

judge = build_llm()
evaluator = Evaluator(judge=judge, concurrency=4)

report = evaluator.evaluate(
    cases=[
        EvalCase(
            id="case-1",
            question="Under FERPA, when do rights transfer from parents to a student?",
            retrieved_contexts=[
                "When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."
            ],
            reference_contexts=[
                "When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."
            ],
            actual_output="Under FERPA, rights transfer when a student turns 18 or enters a postsecondary institution at any age.",
            expected_output="Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age.",
        )
    ],
    metrics=[
        FaithfulnessMetric(threshold=0.8),
        AnswerCorrectnessMetric(threshold=0.8),
    ],
)

print(report.summary.total_cost_usd)
print(report.samples[0].status)
```

If your environment is already configured, you can skip `build_llm()`:

```python
from evalf import EvalCase, Evaluator
from evalf.metrics import AnswerCorrectnessMetric, FaithfulnessMetric

evaluator = Evaluator(concurrency=4)
```

If you pass a custom `judge` into `Evaluator`, you own that client's lifecycle. `evalf` only auto-closes judges that it creates itself.

You can also register project-specific metrics without monkey-patching internals:

```python
from evalf.metrics import BaseMetric, build_metrics, register_metric


class MyMetric(BaseMetric):
    ...


register_metric("my_metric", MyMetric)
metrics = build_metrics(["faithfulness", "my_metric"])
```

## Input format

Each sample can include:

- `id`
- `question`
- `retrieved_contexts`
- `reference_contexts`
- `actual_output`
- `expected_output`
- `attempts`

Example `examples/rag_eval.jsonl`:

```json
{"id":"case-1","question":"Under FERPA, when do rights transfer from parents to a student?","retrieved_contexts":["When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."],"reference_contexts":["When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."],"actual_output":"Under FERPA, rights transfer when a student turns 18 or enters a postsecondary institution at any age.","expected_output":"Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."}
{"id":"case-2","question":"Under FERPA, when do rights transfer from parents to a student?","retrieved_contexts":["When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."],"reference_contexts":["When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."],"actual_output":"Under FERPA, rights transfer only when a student turns 21 years old.","expected_output":"Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."}
```

Example `examples/rag_eval_attempts.json`:

```json
[
  {
    "id": "case-3",
    "question": "Under FERPA, when do rights transfer from parents to a student?",
    "retrieved_contexts": [
      "When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."
    ],
    "reference_contexts": [
      "When a student turns 18 years old, or enters a postsecondary institution at any age, the rights under FERPA transfer from the parents to the student."
    ],
    "expected_output": "Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age.",
    "attempts": [
      {"actual_output": "Under FERPA, rights transfer only when a student turns 21 years old."},
      {"actual_output": "Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."},
      {"actual_output": "FERPA rights move from the parent to the student at age 18 or when the student enters a postsecondary institution at any age."}
    ]
  }
]
```

## Output

`evalf` writes either JSON or Markdown reports.

Example:

```json
{
  "run_id": "run_123456789abc",
  "summary": {
    "total_samples": 2,
    "passed_samples": 1,
    "failed_samples": 1,
    "skipped_samples": 0,
    "total_input_tokens": 1864,
    "total_output_tokens": 622,
    "total_tokens": 2486,
    "total_cost_usd": 0.00174,
    "avg_latency_ms_per_sample": 1421.6,
    "metric_pass_rates": {
      "answer_correctness": 0.5,
      "faithfulness": 0.5
    }
  },
  "samples": [
    {
      "sample_id": "case-1",
      "status": "passed"
    },
    {
      "sample_id": "case-2",
      "status": "failed"
    }
  ]
}
```

## Notebooks

- `notebooks/01_llms.ipynb`: runtime settings, `build_llm`, structured output, and client cleanup
- `notebooks/02_metrics.ipynb`: all built-in metrics with `ReplayJudge`, including C4 composite metric
- `notebooks/03_evaluate_file.ipynb`: load a dataset, run evaluation, inspect the report, and export JSON/Markdown
- `notebooks/04_custom_metrics.ipynb`: create, register, and run a custom metric alongside built-in ones
- `notebooks/05_cli_guide.ipynb`: CLI quick reference with examples for every major feature

## Example Sources

The example facts in this README and in the built-in prompt examples are based on official sources:

- FERPA: https://studentprivacy.ed.gov/faq/what-ferpa
- CDC flu vaccine safety: https://www.cdc.gov/vaccine-safety/vaccines/flu.html
- ADA service animals: https://www.ada.gov/resources/service-animals-2010-requirements/
- EEOC filing deadlines: https://www.eeoc.gov/time-limits-filing-charge

## Notes

- Cost is estimated in USD from the model pricing registry when token usage is available.
- If a response does not include `usage`, `cost_usd` remains `null`.
- Gemini uses a built-in OpenAI-compatible base URL.
- Claude requires an explicit OpenAI-compatible `base_url`; the native Anthropic API endpoint is not supported by this transport.
- `faithfulness`, `context_precision`, and `context_recall` return `1.0` when no material claims are extracted; `evalf` treats those cases as vacuous success rather than hard failure.
- `context_coverage` supports `strict_mode`: when enabled and the score falls below the threshold, the score is clamped to `0.0`.
- `faithfulness` penalizes contradicted claims more heavily than unsupported claims by subtracting an extra `0.5` weight per contradicted claim before normalizing by total claims.
- `k` is capped at `5`.
