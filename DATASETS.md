# Real dataset support

SafeGrip-PFR-ECR does **not** generate or benchmark simulated/synthetic datasets. Every dataset registered below is a real measured/public source. Datasets with incompatible targets are kept in separate scientific roles rather than being forced into one prediction table.

| ID | Source | Real measured target/data | Role |
|---|---|---|---|
| `lira` | LiRA-CD platoon friction test | VIAFRIK standardized road-friction reference + Renault Zoe CAN/AutoPi sensors | **Primary friction benchmark; auto-downloadable** |
| `mssp2023_friction` | Guo et al. MSSP 2023 friction dataset | real vehicle dynamics/video + road adhesion/friction coefficient | **Second primary friction benchmark when author payload is supplied** |
| `kuleuven` | KU Leuven LMSD Concept Car | Kistler RoaDyn wheel forces + vehicle sensing | real-vehicle force/physics validation |
| `kit` | KIT tire force transmission dataset | measured Fx/Fy/Fz characteristics on dry asphalt | tire-force/utilization mechanics validation |
| `mendeley_friction` | tire-pavement friction coefficient dataset | measured friction coefficient, road condition/speed fields when available | external friction-reference validation |
| `deep_dynamics` | Deep Dynamics / IAC | real high-dynamics vehicle signals | domain-shift/vehicle-dynamics auxiliary data |
| `comma2k19` | comma2k19 | production driving sensor streams | unlabeled temporal/domain data |
| `extreme_road` | Extreme Road Image Dataset | real road-surface image classes | optional multimodal road-condition prior |
| `bicycle_tire` | Bicycle Tyre Data | measured force/torque rig data | auxiliary tire-mechanics validation |

## Primary real benchmark 1: LiRA-CD

The LiRA platoon-friction subset contains a regular car driving behind the VIAFRIK reference vehicle on wet road. It combines real in-vehicle/CAN/AutoPi signals with a standardized road-friction reference. The repository aligns the car trajectory to the reference trace, performs leakage-safe temporal splitting, and uses the resulting continuous `mu_ref` target for SafeGrip-PFR-ECR and all paper-supported baselines.

```bash
safegrip download --datasets lira
safegrip prepare --dataset lira
safegrip benchmark --dataset lira --preset trust --proposal pfr --protocol controlled
```

## Primary real benchmark 2: Guo et al. MSSP 2023

The public project describes real video and vehicle-dynamics data collected for tire-road peak-friction/adhesion estimation. The GitHub repository points to a separate public Baidu acquisition link rather than embedding the payload. SafeGrip therefore downloads source metadata/instructions only and **never substitutes generated data**.

```bash
safegrip download --datasets mssp2023_friction
# Follow data/raw/mssp2023_friction/MANUAL_REAL_DATA_REQUIRED.txt
# Put the authors' extracted CSV/TXT/XLS/XLSX dynamics tables there.
safegrip prepare --dataset mssp2023_friction
safegrip benchmark --dataset mssp2023_friction --preset trust --proposal pfr --protocol controlled
```

The adapter requires real columns corresponding to:

- friction/adhesion coefficient;
- vehicle speed;
- longitudinal acceleration;
- lateral acceleration.

If those measured channels are absent, preparation fails with a schema audit. No values are synthesized.

## Real auxiliary validation datasets

### KU Leuven LMSD Concept Car

Used for real-vehicle wheel-force validation. It is not presented as an independent `mu_max` benchmark because it does not expose the same continuous peak-friction target as LiRA/MSSP.

```bash
safegrip download --datasets kuleuven
safegrip prepare --dataset kuleuven
safegrip force-validate --dataset kuleuven
```

### KIT tire data

Used for measured tire-force/utilization validation rather than the main road-reference prediction table.

```bash
safegrip download --datasets kit
safegrip prepare --dataset kit
safegrip force-validate --dataset kit
```

### Mendeley tire-pavement friction data

Used as an external real friction/speed/surface reference. Because it does not provide the synchronized vehicle-dynamics sequence required by SafeGrip-PFR-ECR, the repository performs a reference-distribution validation instead of fabricating missing sensors.

```bash
safegrip download --datasets mendeley_friction
safegrip prepare --dataset mendeley_friction
safegrip friction-reference-validate --dataset mendeley_friction
```

## Other real sources

`deep_dynamics`, `comma2k19`, `extreme_road`, and `bicycle_tire` remain available for domain, multimodal, or mechanics studies. They are not silently mixed into the primary friction-regression table because their targets differ.

```bash
safegrip datasets
safegrip download --datasets lira kit mendeley_friction
safegrip download --datasets all
```

Every automatic download writes source metadata. Sources that require explicit user-side acquisition/terms are reported transparently rather than bypassed.
