# Open dataset support

SafeGrip does not merge datasets with incompatible targets into one table. Each source has a defined scientific role.

| ID | Source | Auto-download | Direct friction label? | Main role |
|---|---|---|---|---|
| `lira` | LiRA-CD platoon friction test | Figshare API | external VIAFRIK road-friction reference | primary friction benchmark |
| `kuleuven` | LMSD Concept Car | Dataverse API | no peak-friction label; WFT force ground truth | validate virtual force/physics layer |
| `kit` | KIT inner-drum dry-asphalt tire data | RADAR runtime discovery | measured forces, not road-reference time series | tire-force/utilization mechanics |
| `deep_dynamics` | Deep Dynamics / IAC | GitHub archive | no | high-dynamics/domain shift |
| `comma2k19` | comma2k19 | GitHub repo/example | no | unlabeled CAN/IMU pretraining/domain data |
| `extreme_road` | Extreme Road Image Dataset | GitHub archive | road class, no numeric ego-tire mu | optional multimodal road prior |
| `bicycle_tire` | Bicycle Tyre Data | Zenodo API | force/torque, bicycle test rig | auxiliary mechanics validation |
| `mendeley_friction` | tire-pavement friction coefficient data | Mendeley public endpoint discovery | friction coefficient | external friction/speed/surface sanity check |

## Primary dataset: LiRA-CD

The platoon test combines normal vehicle/AutoPi/CAN data from a Renault Zoe with a VIAFRIK reference vehicle driving the same wet road. The target in this repository is named `mu_ref` deliberately: it is a standardized external road reference, not the exact instantaneous peak coefficient of the ego tire.

The downloader queries Figshare article metadata at runtime and downloads the current files. The corrected preprocessor aligns each vehicle trip against a route/direction-consistent reference trace when those identifiers are explicit, then applies distance, heading and monotonic-progress checks. Spatial split assignment occurs before interpolation/resampling, and all filling is restricted to one `(trip_id, split)` block. GPS and route metadata remain excluded from learned inputs.

## KU Leuven LMSD Concept Car

This real-vehicle dataset includes automotive onboard sensing plus two front Kistler RoaDyn S635 wheel-force transducers and a Correvit optical velocity reference. SafeGrip uses it as auxiliary ground truth for the force-estimation/physics part of the method.

The Dataverse API is public, but repository guestbook/terms may require user acceptance. SafeGrip stops with a clear message instead of bypassing it.

## KIT tire data

The KIT dataset contains measured longitudinal/lateral force-transmission characteristics on dry asphalt from an inner-drum test bench plus a simulated vehicle slalom cycle. It is ideal for checking force utilization and friction-cone/ellipse assumptions, but it is not disguised as a production-CAN road-friction benchmark.

## Deep Dynamics / IAC

Downloaded from the authors' public GitHub repository. It is useful for high-excitation vehicle-dynamics/domain-shift experiments. The canonicalizer collects compatible vehicle tables without manufacturing friction labels.

## comma2k19

The full dataset is roughly 100 GB. `safegrip download --datasets comma2k19` intentionally downloads the public repository and bundled example so a normal paper setup does not unexpectedly consume ~100 GB. The full upstream dataset remains opt-in outside the default script.

## Extreme Road Image Dataset

Six extreme surface classes are available for a later multimodal extension. The current primary paper remains mechanics/sensor based, so these images are not silently included in the direct sensor-only benchmark.

## Bicycle Tyre Data

Zenodo provides lateral force and self-aligning torque measurements across load, pressure and camber. This is mechanically useful but is a bicycle rig; the repository preserves that distinction.

By default only lightweight/readme/YAML-compatible content is requested; use `--full` for all Zenodo files.

## Mendeley tire-pavement friction data

The public dataset contains a friction workbook measured across road conditions and vehicle speeds. Mendeley's public frontend/API can change; the downloader tries public API patterns and embedded public-file links and never attempts to bypass credentials.

## Commands

```bash
safegrip datasets
safegrip download --datasets lira kit
safegrip download --datasets all
safegrip download --datasets bicycle_tire --full
bash scripts/download_all_datasets.sh
```

Every successful dataset directory receives a `SOURCE.json` with DOI/license/role metadata and a `.complete` marker.
