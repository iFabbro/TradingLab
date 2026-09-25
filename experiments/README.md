# Reproducible experiments

Experiments use an explicitly configured external market-data provider. The runner stores the exact downloaded dataset and metadata for each run.

Example:

```bash
python experiments/run_experiment.py \
  --symbol spy \
  --start 2015-01-01 \
  --end 2025-01-01
```

Each run creates:

```text
experiments/results/<run_id>/
├── dataset.csv
└── metadata.json
```

`metadata.json` records the requested period, provider, actual data range, row count, schema and SHA-256 digest. The runner never silently substitutes synthetic or fallback data.

The current runner is the **data acquisition/reproducibility layer**. The full walk-forward strategy evaluation should consume the saved `dataset.csv` and use its metadata as the provenance record for the subsequent IS/validation/OOS report.

Do not commit large datasets or credentials. For portfolio demonstrations, commit metadata and small reproducible fixtures where appropriate.
