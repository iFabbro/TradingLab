# Statistical inference policy

TradingLab treats statistical inference as part of the research protocol, not as a cosmetic report section.

## 1. Window-level summaries

IS, validation and OOS metrics are reported by walk-forward window. Window metrics are not pooled as if they were independent daily observations.

## 2. OOS inference

The primary OOS confidence intervals use a moving-block bootstrap on the concatenated, non-overlapping daily OOS return series. The block length is explicit in the experiment configuration. This is preferable to an IID bootstrap when short-range serial dependence may be present.

The report also keeps the older window-level IID bootstrap as a descriptive diagnostic. It is not the primary daily inference.

## 3. PSR

The Probabilistic Sharpe Ratio adjusts finite-sample Sharpe inference for skewness and kurtosis. The implementation follows the Bailey/López de Prado framework and converts the annualised Sharpe to the sampling frequency before applying the finite-sample correction.

## 4. DSR and multiple testing

Every parameter configuration actually evaluated is written to `research_trial_ledger.csv`. The DSR uses the declared nominal number of candidates per validation window. It does not pretend that parameter-grid trials are independent when they are not.

The report therefore records the assumption:

> nominal independent trials under the zero-Sharpe IID null

The DSR is used for validation selection. The untouched OOS set is not retroactively treated as another parameter-selection contest. OOS is instead reported with PSR and dependent-data bootstrap intervals.

This separation is intentional: applying a post-hoc multiple-testing correction to an untouched confirmatory OOS set would mix the selection and confirmation stages.

## 5. What the framework does not claim

No finite-sample procedure can make a backtest mathematically infallible. The effective number of independent research trials is generally not identifiable from a parameter grid alone, and the validity of a block bootstrap depends on the block-length choice and the dependence structure.

The framework therefore reports assumptions, trial counts, seeds, block lengths, data provenance and code commit rather than hiding them behind a single score.

## References

- Bailey, D. H. and López de Prado, M. (2014), *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
- Bailey, D. H. and López de Prado, M. (2012/2013), *The Sharpe Ratio Efficient Frontier*.
- Künsch, H. R. (1989), *The Jackknife and the Bootstrap for General Stationary Observations*.
