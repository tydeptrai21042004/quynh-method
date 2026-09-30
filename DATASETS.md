# Final real-data registry

SafeGrip does **not** fabricate missing targets or synthesize data to merge incompatible tasks. The public paper registry is closed to the four measured datasets below.

| ID | Paper role | Target/task | Primary comparators |
|---|---|---|---|
| `lira_cd` | D1, SafeGrip-PFR-ECR friction benchmark | continuous road-friction regression | Du2023 InceptionTime, Todorovic2022 CNN, Lampe2023 GRU, Levenberg2023 STFT adaptation |
| `uc3m_tire` | D2, intelligent-tire mechanics | tire forces / slip angle | Mendoza-Petit2019, Yunta2018 |
| `deep_dynamics_iac` | D3, real racing-vehicle dynamics | velocity / yaw-rate prediction | Chrosniak2024 DDM, Fang-Yu2025 FTHD |
| `io_vnbd` | D4, vehicle navigation | displacement / orientation | Onyekpe2021 QGRU, Wang2023 Transformer |

## D1: LiRA-CD

D1 is the only dataset driven by the legacy end-to-end PFR-ECR benchmark command. The paper protocol prepares trajectory identities first, assigns whole trajectories to train/calibration/validation/test (`group_holdout`), then builds segment-safe temporal windows.

```bash
safegrip download --datasets lira_cd
safegrip prepare --dataset lira_cd
safegrip baselines --dataset lira_cd
safegrip benchmark --dataset lira_cd --preset trust --protocol controlled
```

The controlled D1 table contains exactly four registered paper-supported comparators. The 20-Hz LiRA preparation cannot reproduce Levenberg's reported high-frequency vibration band; that row is therefore labelled a **low-rate method-structure adaptation**, not a source-frequency reproduction.

## D2--D4

The comparator implementations for D2--D4 can be instantiated and smoke-trained with:

```bash
safegrip baseline-check --dataset uc3m_tire --train-step
safegrip baseline-check --dataset deep_dynamics_iac --train-step
safegrip baseline-check --dataset io_vnbd --train-step
```

The current `benchmark` command intentionally rejects D2--D4 because that command expects the D1 LiRA table layout. Their final evaluation belongs to the universal prepared-record pipeline, where each dataset keeps its native targets and no missing labels are fabricated.

## Download all registered sources

```bash
safegrip datasets
safegrip download --datasets all
```

Every adapter must preserve source provenance and fail transparently when required real payloads or target channels are unavailable.
