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
