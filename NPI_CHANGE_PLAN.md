# Universal SafeGrip — Normalized Physical Innovation v3

This revision deliberately keeps the tokenizer, physically typed metadata, latent set-attention backbone, Transformer depth, compositional query decoder, and dataset-agnostic interface unchanged. The change is concentrated in the prediction coordinate used by the existing decoder.

## Final equation

For semantic query `q`, the physical operator still returns `P_q(X)`. Training-only robust statistics are estimated from the physical residual when a reference exists and from the direct target when it does not. The decoder predicts a dimensionless innovation `z_hat` and the physical prediction is reconstructed as

`y_hat = P_q(X) + c_q + s_q z_hat`.

`c_q` is the training median and `s_q` is `1.4826*MAD`, with a standard-deviation fallback when MAD is degenerate. These values are deterministic buffers indexed by the semantic physical query, not trainable dataset-specific parameters.

## Reference-conditional normalization

Whole-channel dropout can remove a state channel used by `P_q`. The previous method then asked the same decoder output to represent a small residual when the reference existed and a large absolute target when it did not. The revised normalizer uses residual statistics while the reference is present and direct-target statistics while it is absent. This keeps both cases in a consistent dimensionless O(1) coordinate without introducing a gate or a second network.

## Displacement correction

The old localization reference `sum(v dt)` represents travelled path length and overestimates endpoint displacement on curved trajectories. The revised reference combines measured scalar vehicle speed and yaw rate using constant-curvature arc integration per token patch, with a straight-line fallback when yaw rate is unavailable. This directly matches the GPS endpoint-displacement target more closely.

## Optimization safeguards

The existing innovation output head is zero initialized, so epoch-zero prediction starts from the physical/reference coordinate instead of an arbitrary random residual. The Kaggle runner now evaluates the checkpoint with the lowest validation loss rather than the final epoch. Neither change adds model capacity.

## Expected target-specific effect

- `v_x`, yaw rate: primary targets for improvement because their current high R2 but poor MAE is consistent with innovation offset/scale conditioning.
- IO-VNBD displacement: primary target for improvement from the corrected curved-motion reference.
- LiRA friction: NPI and best-checkpoint selection target the observed seed instability.
- `v_y`: architecture and physical state reference are unchanged; NPI is intended to preserve its already strong result.
- UC3M slip angle: with no physical reference, NPI reduces to robust direct-target normalization; no artificial physics branch is introduced.

No numerical improvement is hard-coded or claimed before rerunning the real-data experiment. The repository includes regression tests verifying the new mathematical invariants and physical displacement behavior.
