#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'results/lira_paper'
PROC=ROOT/'data/processed/lira'
TUNE=ROOT/'results/lira_frc_tuning'
BTUNE=ROOT/'results/lira_baseline_tuning'

required_main=[
    'metrics.csv','metrics_by_seed.csv','predictions_by_seed.csv',
    'fixed_horizon_ablation.csv','frc_resolution_certificates.csv',
    'frc_separation_curves.csv','frc_certificate_validity.csv',
    'theorem_audit.json','fairness_audit.json','projection_control.csv',
    'reproducibility_manifest.json','statistics/paired_bootstrap_rmse.csv',
]
required_proc=[
    'lira_aligned.csv','lira_signal_audit.csv','lira_physics_audit.json',
    'lira_friction_schema_report.csv','lira_alignment_report.csv','lira_preprocessing_report.json',
]
paper_models={
    'todorovic2022_cnn','lampe2023_lstm','lampe2023_gru',
    'schaefke2023_transformer','chen2025_svdkl','direct_gru_control','safegrip_frc',
}
checks={}
for f in required_main: checks[f'main:{f}']=(MAIN/f).exists()
for f in required_proc: checks[f'processed:{f}']=(PROC/f).exists()

if (MAIN/'theorem_audit.json').exists():
    ta=json.loads((MAIN/'theorem_audit.json').read_text())
    checks['theorem_audit_pass']=ta.get('status')=='PASS' and int(ta.get('grid_theorem_violations',1))==0 and int(ta.get('continuous_bound_violations',1))==0
else: checks['theorem_audit_pass']=False
if (MAIN/'fairness_audit.json').exists():
    fa=json.loads((MAIN/'fairness_audit.json').read_text())
    checks['fairness_audit_pass']=fa.get('status')=='PASS' and bool(fa.get('same_locked_endpoints')) and not bool(fa.get('test_labels_used_for_tuning'))
else: checks['fairness_audit_pass']=False

if (MAIN/'metrics_by_seed.csv').exists():
    m=pd.read_csv(MAIN/'metrics_by_seed.csv'); obs=set(m.model.astype(str)); counts=m.groupby('model').seed.nunique().to_dict()
    checks['all_paper_models_present']=paper_models.issubset(obs)
    checks['five_seeds_each_main_model']=all(int(counts.get(x,0))>=5 for x in paper_models)
else:
    checks['all_paper_models_present']=False; checks['five_seeds_each_main_model']=False

if (MAIN/'fixed_horizon_ablation_by_seed.csv').exists():
    a=pd.read_csv(MAIN/'fixed_horizon_ablation_by_seed.csv')
    expected={f'frc_fixed_h{h}' for h in (4,8,16,32)}
    checks['fixed_horizon_ablations_present']=expected.issubset(set(a.model.astype(str)))
else: checks['fixed_horizon_ablations_present']=False

checks['frc_tuning_present']=all((TUNE/f).exists() for f in ['best_hparams.yaml','tuning_summary.json','tuning_endpoint_manifest.json'])
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
    'proposal':'SafeGrip-FRC finite-window friction resolution certification',
    'note':'PAPER_READY means implementation/fairness/theorem-consistency gates passed; it does not establish novelty or empirical superiority.',
}
MAIN.mkdir(parents=True,exist_ok=True); (MAIN/'paper_readiness.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if status!='PAPER_READY': sys.exit(2)
