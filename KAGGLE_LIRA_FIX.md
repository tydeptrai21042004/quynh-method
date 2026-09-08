# Kaggle LiRA preparation fix

## Failure reproduced

The earlier Kaggle run downloaded the LiRA platoon-test files successfully but failed during
`safegrip prepare --dataset lira` with:

```text
RuntimeError: LiRA alignment requires GPS or distance columns
```

The issue was not a renamed GPS column alone. The public platoon-test release stores the
vehicle channels as separate asynchronous files such as:

- `task_7505_gps_raw.txt`
- `task_7505_speed.txt`
- `task_7505_acc_lon.txt`
- `task_7505_acc_trans.txt`
- `task_7505_acc_yaw.txt`
- other CAN streams

Only the GPS file is expected to carry location. Treating every TXT file as a complete vehicle
trip therefore made alignment impossible for speed/acceleration/etc.

## Correction

The LiRA pipeline now:

1. groups `task_<id>_<sensor>.txt` files by task ID;
2. preserves a common timestamp origin across files;
3. parses the GPS stream separately;
4. resolves required speed/longitudinal/lateral-acceleration streams by filename plus column aliases;
5. synchronizes the asynchronous sensor streams at `lira.resample_hz`;
6. interpolates GPS only within `lira.gps_interp_max_gap_s`;
7. applies a maximum nearest-sensor time gap (`lira.sensor_merge_max_gap_s`);
8. aligns the assembled vehicle task independently against each candidate VIAFRIK trace/direction;
9. keeps the nearest valid match per vehicle timestamp after distance, heading and monotonic checks;
10. performs the spatial split before any later split-local imputation.

GPS remains alignment metadata and is not used as a learned model feature.

## Regression coverage

`tests/test_data.py::test_official_lira_task_streams_are_synchronised_before_alignment`
constructs separate GPS/speed/acceleration task files and verifies that complete LiRA preparation
succeeds. The full packaged test suite currently reports 23 passing tests.

## Diagnostic outputs

After preparation inspect:

```text
data/processed/lira/lira_stream_assembly_report.json
data/processed/lira/lira_alignment_report.csv
data/processed/lira/lira_preprocessing_report.json
```

The first report shows which raw sensor files were resolved and whether elapsed-time fallback
was required. The second shows matching retention and reference traces. The third records the
final preprocessing protocol and split counts.
