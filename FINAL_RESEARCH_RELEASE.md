# SafeGrip-Open v0.5.0 — Final research release

This release is designed to make invalid or degenerate results fail loudly rather than look competitive.

## Research-critical invariants

1. **Published LiRA conversion** — documented Table-2 offsets/resolutions are applied to encoded CAN channels before physics or learning.
2. **No mandatory-signal fabrication** — missing `speed`, `ax`, or `ay` aborts preprocessing.
3. **Segment-safe data handling** — reference-trace changes and large time gaps create separate segments; windows, interpolation and resampling cannot cross them.
4. **Leakage-safe split order** — spatial split is assigned before imputation/resampling; scalers use training only; tuning uses validation only; test is locked.
5. **Fixed physics observation window** — the lower endpoint at an evaluation time is the maximum mechanics lower bound over the configured trailing window, independent of the comparator's neural history length.
6. **Calibration isolation** — calibration labels determine the one-sided lower-bound correction but are not used in gradient training.
7. **Probabilistic consistency** — Gaussian NLL and raw 95% intervals use the raw network mean; physics projection is reported separately as post-processing.
8. **Anti-collapse gates** — the final prediction must vary, beat the train-mean predictor, have positive R2, avoid >95% projection correction, contain enough endpoints, and achieve calibrated-lower coverage consistent with the nominal alpha up to finite-sample tolerance.
9. **Reproducibility** — every benchmark writes processed-data SHA-256, configuration, environment versions, selected hyperparameters, seed-level metrics and exact endpoint IDs.
10. **Paper packaging is gated** — a paper ZIP is created only when the main benchmark, five-seed ablation, tuning, audits and health checks all pass.

## Recommended workflow

```bash
# Development / real-data validation
bash scripts/run_kaggle_trust.sh

# Full paper experiment (requires paper extras and substantially more compute)
bash scripts/run_paper.sh
```

The full paper run performs validation-only tuning, five-seed evaluation, projection parity controls, five-seed ablations, excitation analysis and robustness analysis. `RUN_EXTENDED=1` additionally enables scarcity/cross-route studies.

## Interpretation rule

Do not report a run as a paper result unless `results/lira_paper/paper_readiness.json` says `PAPER_READY`. Even then, the file certifies only the implementation/reproducibility gates; scientific claims still require appropriate interpretation, limitations and comparison with current literature.
