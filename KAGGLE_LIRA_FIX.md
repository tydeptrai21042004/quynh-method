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
succeeds. The full packaged test suite currently reports 25 passing tests.

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


## VIAFRIK reference-schema fix

A second Kaggle failure occurred after stream assembly succeeded: the downloaded
`m3_custom_fric_hh.csv` / `m3_custom_fric_vh.csv` reference files did not resolve to
`mu_ref` under the previous strict alias set. The parser now:

1. supports the official LiRA/VIAFRIK headers `μ_V [-]` and `μ_H [-]`;
2. tolerates Unicode/mojibake and decimal-comma numeric exports;
3. recognises common ASCII friction/coefficient header variants;
4. falls back conservatively to up to two dimensionless O(1) numeric channels only after
   excluding time, distance, GPS, speed, slip, force and bearing fields;
5. writes `lira_friction_schema_report.csv` so the exact downloaded headers and selected
   source columns are auditable.

Inspect these four files after preparation:

```text
data/processed/lira/lira_friction_schema_report.csv
data/processed/lira/lira_stream_assembly_report.json
data/processed/lira/lira_alignment_report.csv
data/processed/lira/lira_preprocessing_report.json
```


## September 2026 zero-byte Figshare mirror regression

A later Kaggle run exposed a separate download-layer failure: Figshare file URLs could return an
HTTP-success response with a **zero-byte body**. The previous downloader treated those responses
as successful, so every `task_7505_*.txt` and `m3_custom_fric_*.csv` file was created at size 0.
`prepare_lira` then failed inside pandas with `csv.Error: Could not determine delimiter`.

The downloader now treats payload integrity as part of the dataset contract:

1. streamed downloads are rejected when the payload is empty;
2. when Figshare metadata provides a byte count, the downloaded size must match it exactly;
3. a bad partial file is never promoted after an HTTP 416 resume response;
4. LiRA per-file downloads try the metadata URL plus DTU/Figshare `ndownloader/files/<id>` mirrors;
5. failed mirror attempts are recorded in `SOURCE.json`; and
6. if the per-file route remains unusable, LiRA falls back to the public article-level bulk ZIP.

Regression tests reproduce both a zero-byte HTTP-success payload and a truncated-size payload.
This prevents the preprocessing stage from ever receiving the silent zero-byte files seen in the
failed Kaggle run.
