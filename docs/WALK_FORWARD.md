# Walk-forward evaluation

TradingLab now provides a chronological expanding-window evaluator in `src/walk_forward.py`.

Each window is split into three strictly ordered segments:

```text
TRAIN ─────────► VALIDATION ─────► TEST
fit/select       tune/select       untouched evaluation
```

The strategy factory receives **training data only**. The resulting strategy instance is then frozen and evaluated on validation and test data. The test slice is never passed to the factory.

## Example

```python
from src.walk_forward import WalkForwardEvaluator


evaluator = WalkForwardEvaluator(
    train_size=756,
    validation_size=252,
    test_size=252,
    step_size=252,
)

result = evaluator.evaluate(
    prices,
    strategy_factory=lambda train: build_strategy_from_training_data(train),
)

print(result.windows)
print(result.validation_metrics)
print(result.test_metrics)
```

## Interpretation

Validation results can be used for model/parameter selection. Test results are reserved for final out-of-sample assessment and should not be used to tune the strategy.

The current evaluator deliberately does **not** perform parameter optimization itself. The caller owns the training/validation selection logic and must return a frozen strategy. This makes the leakage boundary explicit instead of hiding optimization inside the evaluator.

## Limitations

This is a time-series evaluation framework, not a complete institutional research stack. It does not by itself solve survivorship bias, point-in-time universe construction, corporate actions, market impact, borrow constraints, or data-quality problems. Those must be supplied by the data and execution model.
