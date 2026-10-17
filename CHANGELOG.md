# Changelog

All notable changes to this project will be documented in this file.

This project follows the Keep a Changelog format and aims to stay close to Semantic Versioning.

## [Unreleased]

### Changed
- Repository history was rewritten on 2026-10-17 to correct commit authorship. Commit ids recorded in the
  `git_commit` column of `experiments/registry.csv` before that date changed for two registry runs; the registry
  rows are left untouched to keep their hash chain valid. Mapping (old -> new):
  `8b61872` -> `1a485c0` (merge of PR #29), `e6ef9ff` -> `b7d4742` (merge of PR #30). The three v1.0 runs
  (`9f15248`, `d067ef1`, `9ce1024`) are unaffected.

### Added
- Drawdown analysis: `drawdown_periods`, `max_recovery_time` and `drawdown_suggestions` in `src/drawdown.py`, with tests.
- Research protocol v1.1: confirmation rule written into the protocol, changelog, registry and benchmark sections.
- Append-only, SHA-256 hash-chained research registry with unique `execution_id`, separate `decision` records,
  automatic non-destructive migration of v1.0 rows and `experiments/verify_registry.py` (run in CI).
- Buy-and-hold OOS benchmark (`src/benchmark.py`) recorded in the registry for every new run.
- Case study `docs/CASE_STUDY.md`.
- README refreshed for portfolio presentation.
- CONTRIBUTING.md added.
- GitHub issue templates and pull request template added under `.github/`.

### Changed
- `update_registry_decision` removed; verdicts are appended with `append_decision`.
- Registry rows of runs 2026-09-25..27 are kept as `legacy_run` under protocol v1.0.

## [0.1.0] - 2026-07-25

### Added
- Initial public project structure and core quantitative trading modules.
- Documentation for repository usage and contribution workflow.

### Changed
- Repository documentation polished for GitHub presentation.
