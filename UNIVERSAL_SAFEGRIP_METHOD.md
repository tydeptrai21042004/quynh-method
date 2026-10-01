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


## Semantic physical-reference innovation reparameterization

The shared tokenizer, latent backbone and query decoder are unchanged.  The
proposal uses one parameter-free physical reference operator before the final
output:

\[
\widehat y_q = P_q(X) + R_\theta(X,q).
\]

The learned term `R_theta` is always the innovation.  The reference `P_q` is
selected from a registry keyed by **physical target type**, never by dataset
identity:

- `state_component`: latest canonical-unit sensor observation having the same
  quantity, axis and location as the query;
- `road_friction`: horizontal specific-force demand
  \(\sqrt{a_x^2+a_y^2}/g\), computed from measured body accelerations;
- `localization` displacement: time integral of measured vehicle-body speed;
- any unsupported or unobservable query: the neutral reference \(P_q=0\).

The friction reference is an excitation/reference coordinate, **not** asserted
to equal the road-friction coefficient and **not** treated as a deterministic
lower bound.  The decoder learns the remaining road/surface innovation.  This
adds no trainable parameters and keeps one estimator equation for D1--D4.

### Non-interference property

For a query with no registered physical construction from the available sensor
set, \(P_q(X)=0\), hence

\[
\widehat y_q = R_\theta(X,q),
\]

which is exactly the pre-existing UniversalSafeGrip prediction path.  Whole-
channel dropout is respected because every reference is computed only from
tokens that remain visible in the batch mask.

### Unit consistency

All reference operators consume values after canonical unit conversion.  Thus
state references have the query's state unit, the acceleration reference is
dimensionless through division by standard gravity, and integrated speed is in
metres.  Dataset names and dataset-specific numerical calibration constants do
not enter the reference operator.

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
