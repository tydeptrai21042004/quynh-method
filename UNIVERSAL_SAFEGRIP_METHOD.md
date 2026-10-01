# Universal SafeGrip-PFR-ECR

This repository now contains a parallel research implementation for heterogeneous sensor-set learning. It does not replace the validated legacy SafeGrip-PFR-ECR benchmark.

## Architecture

```text
native measured data
      |
dataset-specific parsing / semantic mapping
      |
SensorRecord (canonical units + physical metadata)
      |
physical-time patch tokenizer
      |-- local waveform
      |-- statistics
      |-- relative spectral energy
      `-- absolute-Hz spectral energy
      |
physically typed sensor tokens
      |
latent cross-attention sensor-set backbone
      |
shared latent state L
      |
compositional physical queries
      |-- friction
      |-- Fx/Fy/Fz
      |-- vehicle states
      `-- localization
      |
query-matched physical innovation anchor (when an exact observed state exists)
      |
partial multi-task supervision + mechanics consistency
      |
PFR / one-sided ECR safety integration
```


## Minimal physical innovation reparameterization

The shared architecture is unchanged.  The only proposal-level refinement is a
parameter-free physical anchor for queries whose quantity, axis, and location
exactly match an observed sensor channel.  Let \(P_q(X)\) be the latest
canonical-unit value of that matched channel, with \(P_q(X)=0\) when no exact
match exists.  The existing decoder output is interpreted as an innovation:

\[
\widehat y_q = P_q(X) + R_\theta(X,q).
\]

This is especially natural for history-to-next-state estimation: \(P_q\) is the
latest measured state and the neural decoder estimates only its evolution.  It
is not a target copy.  Matching is ontology-based rather than dataset-based,
requires exact quantity/axis/location agreement, ignores static context, and is
permutation invariant when several equivalent sensors are present.

For unobserved targets such as LiRA road friction, \(P_q(X)=0\), so the method
reduces exactly to the original UniversalSafeGrip predictor.  Whole-channel
dropout is also respected: when a matching state channel is dropped during
training, the anchor is unavailable and the shared model must infer the query
from the remaining sensor set.  No trainable parameters, dataset-specific heads,
new losses, or dataset identifiers are introduced.

### Non-interference property

If no input channel has the same physical quantity, axis, and location as query
\(q\), then \(P_q(X)=0\) and therefore

\[
\widehat y_q = R_\theta(X,q),
\]

which is exactly the pre-existing proposal.

## Physics rule

The universal implementation deliberately distinguishes utilization from available friction:

\[
 u=\frac{\sqrt{F_x^2+F_y^2}}{|F_z|+\epsilon}.
\]

When the stated mechanics assumptions justify it, \(u\le\mu\) can be used as physical lower evidence or as a consistency inequality. The implementation does **not** redefine \(u\) as a friction estimate.

## Current runnable architecture audit

```bash
safegrip universal-smoke
```

The smoke command builds mixed-rate synthetic sensor channels solely to test architecture mechanics (not scientific performance). It audits variable sensor counts, positive scale outputs, and token-order invariance.

## Implemented modules

- `safegrip.universal.schema`: sensor/context/target records.
- `safegrip.universal.units`: explicit canonical unit conversion.
- `safegrip.universal.ontology`: physical semantics.
- `safegrip.universal.tokenizer`: physical-time waveform/statistical/spectral descriptors.
- `safegrip.universal.batching`: variable-token padding/masking.
- `safegrip.model`: physical token encoder, latent sensor-set backbone, compositional query decoder.
- `safegrip.training`: partial-supervision objectives, dataset-balanced sampler, whole-channel dropout, research loss/training primitive.
- `safegrip.universal.ecr`: target/domain-conditional wrapper around the existing one-sided ECR core.
- `safegrip.sensor_io`: generic DataFrame mapping and bridge from the existing prepared friction tables.

## Not yet claimed complete

Full real multi-dataset training is intentionally not marked complete because this source archive does not contain all external measured datasets. KU Leuven/KIT sequence semantics must be verified from the actual downloaded payload before those tables are used as temporal universal-training records. Main-paper results, leave-one-domain-out evidence and final ECR coverage must therefore be generated only after real data are present.
