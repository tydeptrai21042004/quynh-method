# Unified SafeGrip proposal and D2--D4 comparator implementations

## One proposal method across all four datasets

The paper proposal is **one dataset-agnostic model**, not four dataset-specific networks:

```text
raw dataset
  -> dataset adapter (column names, timestamps, canonical physical units only)
  -> physical sensor tokens
  -> shared physically typed token encoder
  -> shared sensor-set latent-attention backbone
  -> shared compositional physical-query decoder
  -> point estimate + learned scale
  -> quantity-triggered mechanics / uncertainty losses when the needed quantities exist
```

The default/full proposal contains **no dataset-ID embedding and no dataset-specific
backbone or output head**.  One `UniversalSafeGrip` instance can therefore process
D1--D4 without being told which dataset produced a record.

What is allowed to differ is the *measurement interface*, not the method:

- adapters map each archive's columns and units into the same physical ontology;
- each benchmark requests the physical targets it actually measures (friction,
  tire forces/slip, vehicle state, or localization error);
- target normalization/statistics are fitted on that dataset's training split.

`dataset_id_conditioning` remains only as an explicit ablation.  If it performs
better, that result must be reported as evidence about the value of dataset
identity; it is not the default proposal.

## D2--D4 paper baselines

All six previously registry-only comparators now have trainable local
**paper-structured reproductions**.  This wording is deliberate: the repository
does not claim unpublished fuzzy FIS files, original learned checkpoints, or
bit-for-bit upstream source equivalence.

| Dataset | Local comparator | Local mathematical/model mechanism | Outputs used |
|---|---|---|---|
| D2 UC3M | `mendoza2019_fuzzy` | hierarchical trainable TSK fuzzy blocks; paper rule counts 21/217/460/145 | Fx, Fy, Fz, slip angle |
| D2 UC3M | `yunta2018_fuzzy_lfc` | fuzzy slip/load/lateral-friction estimator | Fy, Fz, slip angle |
| D3 IAC | `chrosniak2024_ddm` | recurrent parameter estimator + bounded Physics Guard + differentiable single-track/Pacejka step | vx, vy, yaw rate |
| D3 IAC | `fang_yu2025_fthd` | DDM physics core + learned residual + hybrid supervised/physics-anchor loss | vx, vy, yaw rate |
| D4 IO-VNBD | `onyekpe2021_qgru` | Hamilton-product quaternion GRU recurrent maps | displacement, orientation |
| D4 IO-VNBD | `wang2023_transformer` | positional Transformer encoder for wheel-odometry error prediction | displacement, orientation |

Yunta is intentionally evaluated only on quantities its local reproduction
supports.  It does not manufacture a longitudinal force target merely to make
all D2 output vectors the same size.

## Verification

Smoke-test a dataset's registered local comparators:

```bash
safegrip baseline-check --dataset uc3m_tire --train-step
safegrip baseline-check --dataset deep_dynamics_iac --train-step
safegrip baseline-check --dataset io_vnbd --train-step
```

Verify that the same proposal architecture runs a dataset's physical query set:

```bash
safegrip universal-check --dataset lira_cd
safegrip universal-check --dataset uc3m_tire
safegrip universal-check --dataset deep_dynamics_iac
safegrip universal-check --dataset io_vnbd
```

The old `safegrip benchmark` engine remains D1/LiRA-specific because it assumes
the historical LiRA tabular layout.  D2--D4 should be evaluated through labeled
universal prepared records; the code intentionally refuses to synthesize labels
that an archive does not contain.
