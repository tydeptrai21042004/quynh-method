# NPI-v3 ablation protocol

## Active proposal

There is exactly one active proposal: **Universal SafeGrip with Normalized Physical Innovation v3 (NPI-v3)**.

For an observable semantic physical reference,

\[
\widehat y_q=P_q(X)+c_{q,\mathrm{ref}}+s_{q,\mathrm{ref}}R_\theta(X,q).
\]

If the reference is not observable, including after whole-channel dropout,

\[
\widehat y_q=c_{q,\mathrm{direct}}+s_{q,\mathrm{direct}}R_\theta(X,q).
\]

The robust centers and scales are fitted on the training split only and stored as non-trainable semantic buffers. No alternative proposal revision is used in the paper experiment path.

## Primary ablation table

The primary table is intentionally compact and tests the central mathematical claim before secondary architecture details.

| Variant | Controlled change | Question answered |
|---|---|---|
| `direct_normalized` | Remove the physical reference; use robust direct-target normalization | Does \(P_q(X)\) help beyond the shared neural model? |
| `reference_only` | Use \(P_q(X)\) only; no learned correction | How strong is the deterministic physical reference by itself? |
| `old_physical_innovation` | \(\widehat y=P_q+R_\theta\), no robust residual center/scale | Does NPI-v3 improve the previous innovation coordinate? |
| `npi_no_center` | Set residual/direct center to zero | Is robust centering necessary? |
| `npi_no_scale` | Set residual/direct scale to one | Is robust scaling necessary? |
| `mean_pool` | Replace latent set attention with mean pooling | Does the sensor-set fusion mechanism matter? |
| `channel_id_only` | Remove physical/time/unit metadata and use channel IDs | Does physical semantic typing matter beyond arbitrary channel identity? |
| `npi_v3` | Full proposal | Reference result |

All trainable variants use the same real-data split, seed, epoch budget, optimizer, validation checkpoint rule, target definitions and metrics. `reference_only` has no learned correction and therefore requires no optimization.

## Supplementary controls

- `no_physical_metadata`: removes quantity/axis/location embeddings while retaining other metadata.
- `no_spectrum`: removes spectral descriptors.
- `no_sensor_dropout`: removes whole-channel dropout.
- `dataset_id_conditioning`: explicitly injects dataset identity; this is a negative control for the dataset-independent design claim.
- `path_length_reference`: D4-only control replacing curvature-aware endpoint displacement with the earlier travelled-path-length reference \(\sum v\,\Delta t\).
- `no_learned_scale`: removes the learned predictive uncertainty scale while leaving point prediction unchanged.

The former mechanics-query/force-consistency/friction-inequality switches are deliberately excluded from the NPI-v3 paper ablation set because the current D1--D4 query sets do not jointly activate the required force/utilization outputs. Reporting them would not provide evidence about the active NPI-v3 experiments.

## Running

Primary three-seed ablations:

```bash
python scripts/KAGGLE_NPI_V3_ABLATIONS.py
```

Add supplementary controls:

```bash
SAFEGRIP_INCLUDE_SUPPLEMENTARY=1 python scripts/KAGGLE_NPI_V3_ABLATIONS.py
```

Run selected controls only:

```bash
SAFEGRIP_ABLATIONS=npi_v3,old_physical_innovation,npi_no_center,npi_no_scale \
SAFEGRIP_ABLATION_SEEDS=3101,3102,3103 \
python scripts/KAGGLE_NPI_V3_ABLATIONS.py
```

The wrapper disables literature baselines during ablation runs and reuses the same one-proposal real-data runner. It writes per-seed and aggregate CSV files and a resumable `safegrip_npi_v3_ablation_master.zip`.
