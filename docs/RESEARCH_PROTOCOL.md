# Strategy Research Protocol

**Status:** Frozen research protocol  
**Version:** 1.0  
**Scope:** All strategy experiments conducted in TradingLab after this protocol is adopted.

## Purpose

This document defines the rules for evaluating trading strategies in TradingLab. The objective is to separate research decisions from observed results and to make experiments reproducible, comparable, and resistant to data leakage, overfitting, and post-hoc parameter selection.

Once an experiment starts, these rules must not be changed because of its observed results. A material protocol change creates a new protocol version and requires the affected experiment to be rerun.

## 1. Dataset

Every experiment must record the exact dataset used, including:

- provider and retrieval method;
- symbols/tickers;
- start and end dates;
- frequency;
- timezone and timestamp convention;
- adjustment policy;
- missing-data handling;
- dataset checksum or equivalent provenance identifier where available;
- retrieval timestamp/metadata.

The experiment artifact must preserve the actual dataset used for the run whenever technically possible.

No experiment may silently replace, extend, truncate, or revise its dataset after seeing results.

## 2. Investable Universe

The investable universe must be explicitly defined before the experiment.

The experiment must document:

- eligible instruments;
- inclusion/exclusion rules;
- rebalancing or membership rules;
- treatment of delisted/unavailable instruments when relevant;
- whether the universe is static or time-varying.

Universe construction must not use information that would have been unavailable at the historical decision time. Survivorship bias must be identified and, where applicable, measured or mitigated.

## 3. Periods and Temporal Ordering

Every experiment must explicitly define:

`TRAIN → VALIDATION → FROZEN MODEL/PARAMETERS → OOS TEST`

Temporal ordering is mandatory. Future observations must never influence signals, features, parameter selection, sizing rules, or model decisions used in an earlier period.

The final OOS period must remain untouched until all decisions that affect the tested strategy have been frozen.

Dataset availability beyond the final OOS date does not imply that those observations belong to the OOS sample; the actual tested dates must be reported explicitly.

## 4. Walk-Forward Evaluation

Strategies must be evaluated with the project's walk-forward framework unless a documented exception is approved before the experiment.

Each window must record:

- train period;
- validation period;
- selected parameters/model;
- OOS period;
- OOS trades/observations;
- performance metrics;
- costs and slippage;
- provenance.

Windows must be evaluated in chronological order. OOS results from one window must not be used to select parameters for another window.

The aggregate OOS result must be calculated from the window-level outputs using the framework's predefined aggregation rules.

## 5. Parameter Selection

Parameter grids must be declared before running the experiment.

Selection must occur exclusively on TRAIN/VALIDATION data according to the predefined selection rule. The selected parameters are then frozen before OOS evaluation.

The OOS period must never be used to:

- select parameters;
- choose among strategies;
- alter thresholds;
- change position sizing;
- change entry/exit rules;
- choose the cost model;
- decide which windows to report.

If OOS results cause any of these decisions to change, the resulting test is a new experiment and the previous OOS result must remain recorded as the original test.

## 6. Transaction Cost Model

Every strategy must specify transaction costs before the OOS run.

The baseline TradingLab research assumption is:

- **5 bps transaction cost**;
- applied according to the execution/cost semantics implemented by the frozen engine.

Any alternative cost model must be declared before the experiment and recorded in metadata.

Gross and net results must both be retained whenever the engine supports them. Cost drag must be reported explicitly.

## 7. Slippage

The baseline research assumption is:

- **2 bps slippage**.

Slippage must be applied consistently according to the frozen execution model and must not be tuned after observing performance.

A strategy that only works under unrealistically favorable execution assumptions must not be promoted to candidate status.

Where useful, additional counterfactual cost/slippage scenarios may be reported, but they must be labeled as predefined scenarios rather than selected after observing their results.

## 8. Position Sizing and Exposure

Position sizing must be deterministic, documented, and identical between validation and OOS except for parameters legitimately selected and frozen by the protocol.

The experiment must record the exposure contract, including as applicable:

- long/short direction;
- target position or weight definition;
- leverage constraints;
- maximum exposure;
- compounding convention;
- cash treatment;
- rebalance/execution timing.

Sizing rules must not depend on future returns or future realized volatility unless the information is explicitly available at the decision timestamp.

## 9. Bootstrap

Statistical uncertainty must be assessed using the predefined bootstrap methodology in the framework.

The current baseline methodology uses a moving-block bootstrap for OOS daily observations, preserving local temporal dependence. The experiment must record:

- bootstrap method;
- block length;
- number of replications;
- random seed where applicable;
- confidence level;
- metric being bootstrapped.

Bootstrap results are uncertainty estimates, not proof of future profitability.

## 10. Probabilistic Sharpe Ratio (PSR)

PSR must be reported for the aggregate OOS result where the required inputs are available.

PSR must account for the observed Sharpe, sample size, skewness, and excess kurtosis according to the implementation used by the frozen framework.

PSR must not be presented as the probability that a strategy will make money in the future. It is a statistical diagnostic concerning the observed Sharpe relative to a specified benchmark under the model's assumptions.

## 11. Deflated Sharpe Ratio (DSR)

DSR must be used to assess whether selected validation performance remains convincing after accounting for the number of candidate trials represented in the experiment.

The experiment must record the nominal number of candidate configurations/trials used for selection.

If the number of statistically independent trials cannot be established, the artifact must state that limitation. DSR must then be treated as a diagnostic rather than an exact correction for the entire research process.

## 12. Multiple Testing

Multiple testing must be considered at the parameter-selection level and, where identifiable, at the strategy-research level.

The following are prohibited:

- repeatedly changing a parameter grid after seeing results and reporting only the successful run;
- trying multiple strategies and presenting only the best result without recording the search process;
- changing the evaluation period after seeing performance;
- changing cost/slippage assumptions to obtain a preferred result;
- silently discarding failed windows or experiments.

All material trials used to make a research decision should be recorded in the experiment ledger or an equivalent research log.

The framework's DSR correction does **not** automatically account for every undocumented research decision made outside the experiment. This limitation must be stated when relevant.

## 13. Acceptance Criteria

A strategy is not accepted merely because its backtest is profitable.

A candidate must satisfy the following minimum conditions unless an experiment-specific threshold is declared before testing:

1. No known data leakage or look-ahead bias.
2. Reproducible dataset and provenance.
3. Correct execution, exposure, sizing, cost, and slippage semantics.
4. Positive **net** OOS performance is required for candidate consideration.
5. The aggregate OOS result must not rely exclusively on a single exceptional window.
6. OOS performance must remain economically meaningful after the predefined cost/slippage assumptions.
7. Bootstrap uncertainty must be reported; a confidence interval that includes zero is evidence against claiming a statistically established positive edge, though not proof of no edge.
8. PSR/DSR and multiple-testing diagnostics must be reported where applicable.
9. Validation performance must not be treated as OOS evidence.
10. Any material limitation must be documented before promoting the strategy.

These criteria are gates for promotion, not guarantees of future profitability.

## 14. Failed Strategy Criteria

A strategy is classified as **FAILED** for the experiment when one or more of the following applies:

- known leakage or invalid temporal ordering is detected;
- the implementation violates the strategy contract;
- the OOS result is negative and does not meet the predefined candidate criteria;
- the strategy's apparent edge disappears after the predefined costs/slippage;
- the result is materially dependent on an isolated window and lacks sufficient robustness for candidate status;
- parameter selection cannot be reproduced without using OOS information;
- required provenance or experiment metadata is missing;
- a post-hoc modification materially changes the tested hypothesis or evaluation protocol.

A failed strategy is not deleted. Its artifact, parameters, results, and reason for failure remain part of the research record.

A failure means **the tested hypothesis did not satisfy the protocol for this experiment**. It does not prove that the underlying economic phenomenon can never work.

## 15. Candidate Strategy Criteria

A strategy may be classified as **CANDIDATE** only after satisfying the protocol and passing the predefined acceptance gates.

At minimum, candidate status requires:

- clean temporal separation;
- reproducible provenance;
- valid implementation;
- positive net OOS performance;
- no material evidence that performance is generated solely by one anomalous window;
- explicit cost/slippage survival;
- complete statistical diagnostics;
- documented multiple-testing/selection considerations;
- no unresolved critical methodological limitation.

Candidate status does **not** mean production-ready, live-trading-ready, or proven profitable in the future.

A candidate should subsequently undergo additional robustness checks, sensitivity analysis, and—where appropriate—a genuinely untouched holdout or paper-trading/live-simulation phase before real capital is considered.

## Immutable Experiment Rule

Once a strategy experiment has begun, its protocol, dataset definition, universe, periods, parameter grid, cost model, slippage model, sizing rules, and acceptance criteria must not be changed in response to observed results.

If a change is necessary, create a new protocol version or a new experiment identifier and preserve the original result unchanged.

The correct workflow is:

```text
HYPOTHESIS
    ↓
PROTOCOL + PARAMETERS FROZEN
    ↓
DATASET + UNIVERSE FROZEN
    ↓
TRAIN
    ↓
VALIDATION / PARAMETER SELECTION
    ↓
PARAMETERS FROZEN
    ↓
OOS TEST
    ↓
COSTS + SLIPPAGE
    ↓
BOOTSTRAP + PSR + DSR
    ↓
AUDIT
    ↓
FAILED / CANDIDATE
```

## Current Baseline

The existing SPY Momentum experiment is retained as the first research baseline. Its observed OOS result must not be used to alter this protocol retrospectively.

Future strategies must be evaluated against this frozen protocol so that comparisons remain meaningful.