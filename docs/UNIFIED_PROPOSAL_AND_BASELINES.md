# Universal SafeGrip-NPI and paper baselines

## One active proposal

The paper proposal is **Universal SafeGrip-NPI (NPI-v3)**. The same model structure is used for all four real-data tasks:

```text
raw archive
  -> measurement adapter (columns / timestamps / physical units only)
  -> physically typed temporal tokens
  -> shared latent sensor-set backbone
  -> shared semantic query decoder
  -> NPI-v3 physical reconstruction
```

For query `q`, the model predicts a dimensionless innovation. With an observable semantic physical reference,

```text
y_hat = P_q(X) + c_q,ref + s_q,ref R_theta(X,q).
```

If the reference is absent, including after sensor-channel dropout,

```text
y_hat = c_q,direct + s_q,direct R_theta(X,q).
```

The robust center/scale statistics are fitted on the training partition only. They are indexed by physical-query semantics rather than dataset identity and introduce no trainable calibration head.

The active proposal contains no RA-NPI reference gain, no CSI affine reference calibration, no semantic relational-contrast preprocessing, no dataset-specific backbone, and no dataset-specific output head.

## Measurement interfaces

Adapters may differ only because source archives use different columns, units, timestamps, and available measured targets. These differences do not alter the proposal architecture or learning rule.

## Publication-backed comparators

| Dataset | Comparator | Local reproduction role |
|---|---|---|
| LiRA-CD | Du2023 Dynamics-InceptionTime | common-input paper-structured adaptation |
| LiRA-CD | Todorovic2022 CNN | common-input paper-supported adaptation |
| LiRA-CD | Lampe2023 GRU | recurrent paper-structured reproduction |
| LiRA-CD | Levenberg2023 STFT | low-rate method-structure adaptation |
| UC3M | Mendoza-Petit2019 fuzzy | local paper-structured fuzzy reproduction |
| UC3M | Yunta2018 fuzzy/LFC | local fuzzy reproduction on available target |
| IAC | Chrosniak2024 Deep Dynamics | local physics/data dynamics reproduction |
| IAC | Fang/Yu2025 FTHD | local hybrid dynamics reproduction |
| IO-VNBD | Onyekpe2021 QGRU | local quaternion-GRU wheel-odometry reproduction |
| IO-VNBD | Onyekpe2021 WhONet | local recurrent wheel-odometry reproduction |

Comparator results are controlled local reproductions, not copied values from the publications and not claims of bit-exact unpublished implementations.

## Real-data contract

The paper runner uses measured data only. Missing scientific targets are never synthesized to make datasets share a common output vector. D3 excludes simulator NPZ files; D4 displacement targets come from recorded GPS coordinates. A missing required real source causes an explicit unavailable status.

## Required final evidence

The final paper table should use the same NPI-v3 revision for seeds 3101, 3102, and 3103 and report mean ± standard deviation. Do not mix the one-seed NPI result with the older three-seed manuscript aggregate.
