# Literature comparator policy

The repository keeps five tire/road-friction literature comparators. Every direct numerical score in this repository is produced on the common prepared benchmark; source-paper scores from different datasets are not mixed into the table.

| ID | Role | Fidelity wording |
|---|---|---|
| `todorovic2022_cnn` | temporal CNN comparator | architecture-faithful adaptation; common LiRA inputs and scalar target |
| `lampe2023_lstm` | recurrent comparator | architecture- and reported-training-setting adaptation; LiRA sensor subset/protocol differ |
| `lampe2023_gru` | recurrent comparator | architecture- and reported-training-setting adaptation; LiRA sensor subset/protocol differ |
| `schaefke2023_transformer` | Transformer comparator | methodology-level adaptation; exact source architecture not claimed |
| `chen2025_svdkl` | uncertainty/deep-kernel comparator | methodology-level adaptation; source category-selection stage not reproduced |

## Two benchmark protocols

### Controlled

Used for the primary method comparison. Trial counts, tuning seeds, and training budget are controlled. Model architecture/preprocessing families remain intact.

### Source-faithful

Uses the repository's recoverable source-setting training defaults (for example the longer Lampe schedule). This table answers a different question and is never described as compute-matched.

## Direct control

`direct_gru_control` is not a literature baseline. It is an internal scientific control with the same GRU encoder width/depth as SafeGrip-FRC and a closely matched regression head. Its purpose is to isolate the value of response inversion/certification from recurrent capacity.
