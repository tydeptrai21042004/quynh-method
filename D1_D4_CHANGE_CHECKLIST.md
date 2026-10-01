# D1–D4 paper benchmark refactor — applied change checklist

## Implemented in this patch

- [x] Public dataset registry is closed to exactly D1 LiRA-CD, D2 UC3M Tire, D3 Deep Dynamics IAC, and D4 IO-VNBD.
- [x] Public paper-baseline registry is dataset-scoped and rejects cross-dataset combinations.
- [x] D2/Yunta2018 is explicitly marked `exact_dataset: false`; the code does not claim exact U6ICRX reproduction.
- [x] Removed Todorovic/Lampe from the final default benchmark configuration and public baseline allowlist.
- [x] Added download metadata for the four paper datasets.
- [x] Added exact U6ICRX Dataverse downloader and GitHub download routes for D3/D4.
- [x] Download `SOURCE.json` now records per-file SHA256 and byte size.
- [x] Added D1/D2/D3/D4 preparation dispatch.
- [x] Added dataset-specific SensorRecord adapters.
- [x] D2 adapter does not synthesize unavailable force targets.
- [x] D3 adapter separates history inputs from the final next-state target.
- [x] D4 wheel-speed channels remain angular when supplied in angular units.
- [x] Added dataset-specific physical query sets instead of one friction query universe.
- [x] Added displacement/state target ontology entries; the verified IO-VNBD experiment uses displacement as the common real target.
- [x] Fixed force utilization to `sqrt(Fx^2 + Fy^2) / max(abs(Fz), eps)`.
- [x] Added heteroscedastic Gaussian NLL so the Universal SafeGrip scale head receives gradients.
- [x] Empty/undersized ECR calibration now fails explicitly instead of silently returning a zero correction.
- [x] Added A0–A13 computation-level ablation configuration.
- [x] A1–A4 can independently remove physical/unit/rate/time metadata.
- [x] A5 disables spectral features in the tokenizer.
- [x] A6 uses a true masked mean-pooling backbone.
- [x] A7 disables whole-sensor dropout.
- [x] A9/A10 can disable the corresponding physics loss terms.
- [x] A11 disables learned scale output/loss contribution.
- [x] A12 optionally injects dataset-ID conditioning as a negative control only.
- [x] A13 replaces physical metadata with learned channel-ID embedding for the negative control.
- [x] CLI now exposes only D1–D4 datasets and dataset-scoped baseline metadata.
- [x] Added `universal-check --dataset ... --ablation ...` integration sanity check.
- [x] 56/56 D1–D4 × A0–A13 architecture checks pass.
- [x] Full test suite passes: 138 tests.

## Intentionally not faked / still requires dedicated reproduction work

- [ ] Mendoza2019 fuzzy/Pacejka local reproduction wrapper.
- [ ] Yunta2018 fuzzy-LFC local reproduction wrapper.
- [ ] Chrosniak2024 DDM wrapper around the official upstream evaluation workflow.
- [ ] Fang & Yu 2025 FTHD/EKF-FTHD wrapper around the official upstream workflow.
- [ ] Onyekpe2021 QGRU/GRU exact training/evaluation reproduction.
- [ ] Onyekpe2021 WhONet exact training/evaluation reproduction beyond the current paper-structured local reproduction.
- [ ] Exact paper split/group definitions must be locked after inspecting downloaded D2/D4 source tables and the upstream baseline scripts.
- [ ] Dataset-specific published metric runners (especially D3 ADE/FDE and D4 CRSE) should be implemented from the exact paper definitions rather than approximated.
- [ ] U3 missing-sensor sweeps, U4 held-out sensor combinations, U5 joint multi-dataset training, and U6 leave-one-dataset-out experiment runners still need the final real-data training loop.
- [ ] Final paper result CSVs and claim audit must be produced only after the above exact baseline/split definitions are frozen.

## Validation commands used

```bash
pytest -q
# 138 passed

python -m safegrip.cli datasets list
python -m safegrip.cli baselines --dataset uc3m_tire
python -m safegrip.cli universal-check --dataset io_vnbd --ablation channel_id_only
```

The benchmark deliberately refuses synthetic/generated data, fabricated targets, dummy comparators, and generic-model substitutions for paper baselines. This is a scientific-safety guard, not a silent fallback.
