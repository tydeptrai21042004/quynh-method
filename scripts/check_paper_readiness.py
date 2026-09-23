#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'results/lira_paper'
PROC=ROOT/'data/processed/lira'
TUNE=ROOT/'results/lira_pfr_tuning'
BTUNE=ROOT/'results/lira_baseline_tuning'

required_main=[
    'metrics.csv','metrics_by_seed.csv','predictions_by_seed.csv',
    'pfr_component_ablation.csv','pfr_component_ablation_by_seed.csv',
    'pfr_safety_metrics.csv','pfr_safety_metrics_by_seed.csv','pfr_safety_fusion_audit.csv','pfr_theorem_audit.json','pfr_method_audit.json',
    'fairness_audit.json','reproducibility_manifest.json','statistics/paired_bootstrap_rmse.csv',
]
required_proc=[
    'lira_aligned.csv','lira_signal_audit.csv','lira_physics_audit.json',
    'lira_friction_schema_report.csv','lira_alignment_report.csv','lira_preprocessing_report.json',
]
paper_models={
    'todorovic2022_cnn','lampe2023_lstm','lampe2023_gru',
    'schaefke2023_transformer','chen2025_svdkl','direct_gru_control','safegrip_pfr',
}
checks={}
for f in required_main: checks[f'main:{f}']=(MAIN/f).exists()
for f in required_proc: checks[f'processed:{f}']=(PROC/f).exists()

if (MAIN/'pfr_theorem_audit.json').exists():
    ta=json.loads((MAIN/'pfr_theorem_audit.json').read_text())
    checks['theorem_audit_pass']=ta.get('status')=='PASS' and int(ta.get('deterministic_fusion_logic_violations',1))==0
else: checks['theorem_audit_pass']=False

if (MAIN/'fairness_audit.json').exists():
    fa=json.loads((MAIN/'fairness_audit.json').read_text())
    checks['fairness_audit_pass']=(
        fa.get('status')=='PASS'
        and bool(fa.get('same_locked_endpoints'))
        and bool(fa.get('point_and_scale_network_uses_training_labels_only'))
        and bool(fa.get('physics_conformal_quantile_uses_calibration_labels_only'))
        and bool(fa.get('statistical_conformal_quantile_uses_calibration_labels_only'))
        and not bool(fa.get('test_labels_used_for_training_calibration_or_selection'))
    )
else: checks['fairness_audit_pass']=False

if (MAIN/'metrics_by_seed.csv').exists():
    m=pd.read_csv(MAIN/'metrics_by_seed.csv'); obs=set(m.model.astype(str)); counts=m.groupby('model').seed.nunique().to_dict()
    checks['all_paper_models_present']=paper_models.issubset(obs)
    checks['five_seeds_each_main_model']=all(int(counts.get(x,0))>=5 for x in paper_models)
else:
    checks['all_paper_models_present']=False; checks['five_seeds_each_main_model']=False

if (MAIN/'pfr_component_ablation_by_seed.csv').exists():
    a=pd.read_csv(MAIN/'pfr_component_ablation_by_seed.csv')
    expected={'direct_gru_control','pfr_residual_raw','pfr_physics_features_no_attention','safegrip_pfr','safegrip_pfr_safe'}
    checks['decisive_ablation_present']=expected.issubset(set(a.model.astype(str)))
else: checks['decisive_ablation_present']=False

checks['pfr_tuning_present']=all((TUNE/f).exists() for f in ['best_hparams.yaml','tuning_summary.json','tuning_endpoint_manifest.json'])
checks['baseline_tuning_present']=all((BTUNE/f).exists() for f in ['best_hparams.yaml','tuning_summary.json'])
checks['tuning_endpoint_parity']=False
pm=TUNE/'tuning_endpoint_manifest.json'; bms=sorted(BTUNE.glob('*/tuning_endpoint_manifest.json'))
if pm.exists() and bms:
    p=json.loads(pm.read_text())
    checks['tuning_endpoint_parity']=all(
        int(json.loads(x.read_text()).get('eval_start',-1))==int(p.get('eval_start',-2))
        and int(json.loads(x.read_text()).get('n',-1))==int(p.get('n',-2))
        and str(json.loads(x.read_text()).get('sha256'))==str(p.get('sha256'))
        for x in bms
    )

if (PROC/'lira_preprocessing_report.json').exists():
    prep=json.loads((PROC/'lira_preprocessing_report.json').read_text()).get('preprocessing',{})
    for key in ('split_before_imputation','partition_local_imputation','trajectory_aware_splitting','segment_safe_windows'):
        checks[key]=bool(prep.get(key))
else:
    for key in ('split_before_imputation','partition_local_imputation','trajectory_aware_splitting','segment_safe_windows'): checks[key]=False

status='PAPER_READY' if all(checks.values()) else 'REVIEW'
report={
    'status':status,'checks':checks,
    'proposal':'SafeGrip-PFR-ECR Excitation-Aware Conformal Risk-Controlled Residual Estimation',
    'note':'PAPER_READY means implementation/fairness/theorem-consistency gates passed; it does not establish novelty or empirical superiority.',
}
MAIN.mkdir(parents=True,exist_ok=True); (MAIN/'paper_readiness.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if status!='PAPER_READY': sys.exit(2)
