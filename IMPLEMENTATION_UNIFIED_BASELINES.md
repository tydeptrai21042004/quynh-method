# Implemented in this revision

- Added trainable D2 baselines: Mendoza2019 hierarchical fuzzy estimator; Yunta2018 fuzzy LFC estimator.
- Added trainable D3 baselines: Chrosniak2024 Deep Dynamics structural PCNN; Fang/Yu2025 FTHD hybrid fine-tuning model.
- Added trainable D4 baselines: Onyekpe2021 quaternion GRU; Wang2023 Transformer wheel-odometry estimator.
- Added common D2--D4 baseline training helper and `safegrip baseline-check --dataset ... --train-step`.
- Marked D2--D4 registry entries runnable while explicitly labeling them paper-structured local reproductions rather than exact unpublished/source checkpoint reproductions.
- Enforced the full/default UniversalSafeGrip proposal as dataset-agnostic: no dataset embedding or dataset-specific parameters are allocated.
- Kept dataset-ID conditioning only as an explicit ablation.
- Added tests proving one model instance runs all D1--D4 query sets and one shared mixed D1--D4 training batch backpropagates successfully.
- Added `docs/UNIFIED_PROPOSAL_AND_BASELINES.md` describing the unified method boundary and baseline fidelity.

Regression result: 145 pre-existing/new tests passed in grouped full-suite execution before the final mixed-batch test was added; that added test also passed (4/4 unified tests). Warnings were PyTorch/pandas performance/format warnings only.
