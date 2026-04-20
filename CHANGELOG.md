# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- New `context_coverage` metric: single-call LLM judge that scores whether retrieved contexts cover enough key information from reference contexts to answer the question. Supports `strict_mode` to clamp below-threshold scores to `0.0`.

## [1.0.0] - 2026-04-18

### Added

- Initial public release of the `evalf` RAG evaluation library.
- CLI (`evalf run`, `evalf list-metrics`) and Python API (`Evaluator`, `evaluate`, `a_evaluate`).
- Eight built-in metrics: faithfulness, answer_correctness, answer_relevance, context_coverage, context_relevance, context_precision, context_recall, c4.
- Multi-attempt evaluation with `pass@k` and `pass^k` aggregation.
- Static model pricing registry for USD cost estimation.
- JSON and Markdown report output.

### Added

- Standard open-source repository docs and the `py.typed` marker.
- Custom metric registration support through `register_metric()`.

### Changed

- Public package metadata now targets the 1.0 release.
- Claude now requires an explicit OpenAI-compatible base URL.
- README installation guidance now includes PyPI usage.

### Fixed

- CLI configuration and input errors now surface as clean user-facing messages.
- `Evaluator` ownership semantics no longer auto-close caller-managed judges.
- Prompt rendering now uses per-instance examples instead of shared mutable defaults.
- Sample ids are now stable across success and error paths.
