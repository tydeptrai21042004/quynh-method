# Universal SafeGrip — Experiment Protocol

## 1. Preserve the legacy reference

Do not replace the current SafeGrip-PFR-ECR benchmark. Report it as a fixed-schema reference/ablation while the universal method is developed in parallel.

## 2. Data integrity

Only measured/source-supported targets may be used. Never synthesize missing friction, force, timing, or modality information merely to make datasets fit a shared tensor. Unit conversion must occur before tokenization and all conversion rules must be auditable.

## 3. Leakage control

Train/calibration/validation/test partitions must be made at the run/trip/trajectory level wherever available. Windows from the same trajectory must not cross partitions. Conformal calibration data must remain disjoint from model fitting and final test data.

## 4. Progressive training study

Use an incremental design so the effect of each source can be identified:

1. LiRA only.
2. LiRA + one mechanics source whose timing/sequence semantics are verified.
3. Add the second mechanics source.
4. Add other compatible labeled/auxiliary domains one at a time.
5. Full compatible source set.

Do not pool a dataset merely because a loader exists.

## 5. Main ablations

| Variant | Question |
|---|---|
| Legacy fixed-schema PFR | Reference architecture |
| Universal without physical metadata | Does semantic typing matter? |
| Universal without spectrum | Do high-rate descriptors matter? |
| Universal without absolute-Hz spectrum | Does physical frequency information matter? |
| Universal without sensor dropout | Does channel-drop training matter? |
| Mean/set pooling instead of latent cross-attention | Does latent fusion matter? |
| No force/utilization auxiliary supervision | Does cross-dataset mechanics help? |
| No utilization consistency | Does mechanics coupling help? |
| Direct friction query only | Is PFR structure useful? |
| PFR without ECR | What comes from calibration/safety fusion? |
| Full Universal SafeGrip-PFR-ECR | Complete proposal |

## 6. Robustness tests

- Variable channel count.
- Sensor order permutation.
- Leave-one-sensor-out.
- Structured dropout of an entire sensor family.
- Unseen sensor combinations composed of known physical quantities.
- Multiple sampling rates (at least 20/100/1000 Hz in architecture tests).

Do not call this 'arbitrary sensor generalization'; the supported claim is robustness to variable subsets/combinations represented by the ontology.

## 7. Transfer tests

Where source semantics permit:

- within-domain friction evaluation;
- cross-domain friction transfer;
- leave-one-domain-out transfer;
- mechanics-to-friction auxiliary transfer;
- source-by-source incremental training.

## 8. Statistical evaluation

For regression report MAE, RMSE, R2 where meaningful, multiple seeds, and uncertainty intervals/bootstraps. For the controller-facing lower output report one-sided coverage, lower-bound tightness, unsafe overestimation behavior, and per-domain calibration rather than only pooled averages.

## 9. Conformal calibration

The predictive model is dataset-independent. Calibration may be target/deployment-domain conditional because exchangeability pertains to the deployment distribution. Always report calibration sample count and achieved test coverage by domain.

## 10. Evidence threshold for paper claims

Every principal claim must map to a predeclared experiment or theorem/audit. If a source lacks the required measured fields, report it as unsupported for that experiment rather than imputing a scientific target.
