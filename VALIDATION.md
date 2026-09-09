# Validation status

Packaging validation performed on 2026-09-08.

## Automated checks

- `python -m compileall -q src scripts tests`: **PASS**
- `pytest -q`: **23/23 PASS**
- `bash -n scripts/*.sh`: **PASS**
- `bash scripts/smoke_test.sh`: **PASS**
  - synthetic dataset preparation;
  - literature-backed quick baselines;
  - SafeGrip proposal;
  - corrected trip-safe sequence construction;
  - unit tests.

PyTorch emits one non-failing warning for `padding='same'` with an even CNN kernel in the Todorovic architecture-faithful comparator. This is a framework performance warning, not a failed correctness check.

## Tuning-path checks

- `safegrip tune --dataset synthetic --trials 1 --epochs 1 --no-test`: **PASS**
  - validation-RMSE objective completed;
  - `test_used_during_search: false` retained.
- `safegrip tune-baselines --dataset synthetic --models lampe2023_gru --trials 1 --epochs 1`: **PASS**
  - baseline validation-only tuning completed.

## New experiment-path checks

Using the synthetic plumbing dataset only:

- excitation-stratified analysis: **PASS**;
- physical-assumption / `mu_upper` robustness analysis: **PASS**;
- 10/25/50/75/100% data-scarcity analysis: **PASS**;
- five-seed paper-mode ablation aggregation: **PASS**, `n_seeds = 5` for each tested variant;
- force-validation helper: covered by unit test and **PASS**.

Synthetic outputs are not research evidence and are not bundled as paper results.

## New regression tests

The suite now explicitly checks:

- temporal windows cannot cross trip boundaries;
- feature interpolation cannot cross split boundaries;
- official LiRA `task_7505_*` asynchronous sensor streams are synchronized before GPS/reference alignment;
- explicit LiRA route/direction parsing does not invent missing metadata;
- monotonic reference matching;
- source-column unit conversion (`km/h`, `[g]`);
- projection dominance for point estimates;
- projected uncertainty intervals never widen and preserve a truth already inside both sets;
- nested identified lower endpoints are non-decreasing;
- conformal calibration can only relax the physical lower endpoint;
- force-error robust utilization is no larger than nominal utilization.

## Real-data note

The corrected LiRA pipeline writes `lira_stream_assembly_report.json`, `lira_alignment_report.csv` and `lira_preprocessing_report.json`. The packaged source was validated offline with synthetic plumbing data; a full LiRA paper run still requires downloading the public dataset and should be executed with:

```bash
bash scripts/run_paper.sh
```

For the training-heavy scarcity and cross-route studies:

```bash
RUN_EXTENDED=1 bash scripts/run_paper.sh
```

Public repositories may change API metadata, guestbook requirements or authentication rules. The downloader reports those cases rather than bypassing upstream terms.


## v0.5.0 final audit

- `pytest -q`: **31/31 PASS** in the final audit environment.
- New regression tests cover discontinuous-segment window safety, fixed-window maximum lower bounds, mandatory-signal failure, and LiRA Table-2 decoding.
- `scripts/check_paper_readiness.py` is the authoritative automatic paper-output gate.

## v0.7.0 SafeGrip-v3 audit

The v0.7.0 test suite includes checks for:

- identified-set output parameterization (`lower <= prediction <= mu_upper`);
- monotone non-decreasing excitation reliability;
- explicit prior/evidence/reliability outputs from the v3 estimator;
- residual-scale positivity and initialization near the requested error scale;
- label-free excitation feature construction and causal recent-excitation memory;
- excitation memory and jerk calculations that reset at split/segment boundaries;
- preservation of `sg_excitation_score` in physical `[0,1]` scale after proposal feature scaling;
- same-segment relative-pair construction that never crosses trajectory boundaries;
- conditional vector force-balance behavior in the physics lower bound;
- disjoint lower-bound-calibration and predictive-UQ calibration roles.

Repository audit performed for this release: **43 tests passed**. The two remaining runtime warnings are non-fatal upstream/PyTorch warnings already surfaced by the test output. The package also installs successfully in offline verification mode with `pip install -e . --no-build-isolation --no-deps`, reports version `0.7.0`, and exposes the expected CLI commands.

An end-to-end synthetic smoke run completes the SafeGrip-v3 benchmark and controlled ablations and writes the new prior/evidence/reliability audit outputs. A same-budget synthetic regression check also showed the redesigned point core substantially lower RMSE than the v0.6 core; this is a software/regression diagnostic only and is not treated as paper evidence. Scientific-health status is never forced to PASS; low-quality or under-covered runs remain REVIEW by design.
