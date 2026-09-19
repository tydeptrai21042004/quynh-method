# SafeGrip-PNTR small validation experiment

This report records a **development smoke experiment**, not final paper evidence.
It is intentionally small so that proposal failures are detected before running the full LiRA benchmark.

## Validation status

- Repository test suite: **101 passed, 0 failed**.
- Two warnings are unrelated to PNTR correctness (pandas datetime-format inference and PyTorch `padding=same`).
- New unit tests verify that the physics-residual anchor cannot fall below the mechanics lower bound or exceed `mu_upper`.

## Controlled synthetic experiment

Dataset: repository synthetic generator, `n=1600`, seed `321`.
Evaluation: one seed, 5 PNTR training epochs, hidden size 64, batch size 128.
Comparator: same-encoder direct GRU and the adapted Lampe et al. LSTM baseline on the same locked endpoints.

| Method | MAE | RMSE | R2 | unsafe overestimate rate (>0.05) |
|---|---:|---:|---:|---:|
| SafeGrip-PNTR | 0.111597 | 0.137341 | 0.660090 | 0.589286 |
| Direct GRU control | 0.241640 | 0.242632 | -0.060855 | 0.607143 |
| Lampe 2023 LSTM (3-epoch quick baseline) | 0.482570 | 0.530089 | -4.063580 | 1.000000 |

On this particular small split, the physics-residual parameterization clearly improves the same-encoder direct GRU. The response trust-region stage made no additional correction (`correction_rate = 0`), so the reported PNTR result equals the physics-residual anchor. This is desirable fail-safe behavior, but it also means the response-refinement component is **not yet empirically validated** by this smoke experiment.

## Earlier diagnostic on a second synthetic split

On `n=2400`, seed `123`, 8-epoch standalone anchor diagnostic:

- direct-mu GRU RMSE: **0.10934**
- physics-slack GRU RMSE: **0.08603**
- unsafe-overestimate rate: **0.52083 -> 0.29167**

This second split independently supports the physics-residual anchor design. It still does not replace a multi-seed full-data LiRA experiment.

## Interpretation

The redesign fixes the previous FRC failure mode in two ways:

1. Physics is now embedded in the neural output through `mu = lower + nonnegative_slack`, rather than being an unconstrained global inverse replacement.
2. The response-based PNTR correction has an exact neural fallback; when response evidence is weak, it returns the anchor instead of degrading it.

The next publication-grade check must use the full LiRA dataset, multiple evaluation seeds, and the complete literature baseline set. The response-refinement component should only be claimed as beneficial if it produces positive correction rates and improves validation/test metrics without test-label tuning.
