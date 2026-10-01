# Real-data + semantic-reference revision summary

## What changed

1. **One proposal, no dataset-specific proposal branches**
   - UniversalSafeGrip keeps the same tokenizer, latent backbone and query decoder.
   - Final estimator remains one equation: `y_hat = P_q(X) + R_theta(X,q)`.
   - `P_q` is selected through a physical-target-type registry, not dataset names.
   - State: latest matching measured state.
   - Road friction: horizontal specific-force demand `sqrt(ax_rms^2+ay_rms^2)/g`.
   - Displacement: integral of measured body speed.
   - Unsupported query: zero reference.

2. **UC3M/U6ICRX real task fixed**
   - Parses the documented time/e1y/e2y/e3x workbook layout.
   - Uses the real file-level slip-angle condition (0/6/13 deg).
   - No unavailable force labels are synthesized.
   - Mendoza/Yunta local paper reproductions are evaluated only on their compatible slip-angle output.

3. **IO-VNBD real download fixed**
   - Fetches the synchronized Git-LFS payload instead of accepting source-archive pointer stubs.
   - Rejects/repairs stale `.complete` markers from old pointer-only downloads.
   - `prepare` writes a lightweight manifest instead of concatenating the full large archive.
   - Parses the documented 29-column vehicle stream.
   - GPS latitude/longitude form the real displacement target; measured wheel/body channels remain inputs.

4. **Repeated-seed evidence added**
   - `scripts/KAGGLE_ALL_REAL_DATASETS_MULTI_SEED.py`
   - Default seeds: 3101, 3102, 3103.
   - Independent resumable per-seed runs.
   - Aggregates MAE/RMSE/R2 mean, std and count.

## Validation

```text
165 passed, 14 warnings
```

Both Kaggle runners compile successfully.

A frozen real-LiRA checkpoint diagnostic (not a final retrained result) changed
from near-constant output `R2=-0.5621, RMSE=0.04104` to
`R2=0.0538, RMSE=0.03194` when the semantic friction reference was applied to
the same decoder weights. Final performance claims must use the repeated-seed
retrained run.
