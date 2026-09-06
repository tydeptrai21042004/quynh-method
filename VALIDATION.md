# Validation status

Final packaging validation performed on 2026-09-06.

- `python -m compileall -q src scripts tests`: PASS
- `pytest -q`: **11/11 PASS**
- `bash -n scripts/*.sh`: PASS
- `bash scripts/smoke_test.sh`: PASS
  - synthetic dataset preparation
  - literature-backed quick baselines
  - SafeGrip proposal
  - unit tests
- `safegrip tune --dataset synthetic --trials 1 --epochs 2 --no-test`: PASS
  - wrote `best_hparams.yaml`
  - confirmed `test_used_during_search: false`
- single-variant ablation smoke (`safegrip_no_projection`): PASS

The full real-data paper run is intentionally not bundled with downloaded datasets or generated results. It is started with:

```bash
bash scripts/run_paper.sh
```

Public repositories may change access rules, API metadata, guestbook requirements, or require authentication. The downloader reports these cases rather than bypassing upstream terms.
