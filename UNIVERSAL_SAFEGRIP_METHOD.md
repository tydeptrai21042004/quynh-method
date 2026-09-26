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
      |-- utilization
      `-- grip margin
      |
partial multi-task supervision + mechanics consistency
      |
PFR / one-sided ECR safety integration
```

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
