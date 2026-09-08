# SafeGrip v0.5.0 - trustworthy LiRA Kaggle run
# Paste this entire file into one Kaggle cell AFTER pushing v0.5.0 to GitHub.
# Internet ON; T4 GPU recommended.
import os, sys, json, shutil, subprocess
from pathlib import Path

WORK=Path('/kaggle/working'); REPO=WORK/'quynh-method'; ZIP=WORK/'safegrip_lira_trust_results.zip'
def run(cmd,cwd=REPO):
    print('\n'+'='*100+'\nRUN:',cmd,'\n'+'='*100)
    subprocess.run(cmd,shell=True,check=True,cwd=str(cwd),executable='/bin/bash')

shutil.rmtree(REPO,ignore_errors=True)
if ZIP.exists(): ZIP.unlink()
run('git clone --depth 1 https://github.com/tydeptrai21042004/quynh-method.git',WORK)
SRC=REPO/'src'; sys.path.insert(0,str(SRC)); os.environ['PYTHONPATH']=str(SRC)+os.pathsep+os.environ.get('PYTHONPATH','')
run(f'{sys.executable} -m pip install -q -e \".[dev]\"')
import safegrip, pandas as pd
assert safegrip.__version__ == '0.5.0', f'Expected corrected SafeGrip 0.5.0, got {safegrip.__version__}. Push the supplied ZIP to GitHub first.'
run(f'{sys.executable} -m pytest -q')
SG=f'{sys.executable} -m safegrip.cli --config configs/kaggle_trust.yaml'
run(f'{SG} download --datasets lira')
run(f'{SG} prepare --dataset lira')

proc=REPO/'data/processed/lira'
for name in ['lira_signal_audit.csv','lira_physics_audit.json','lira_preprocessing_report.json']:
    p=proc/name
    print('\n###',name)
    print(p.read_text()[:12000] if p.suffix=='.json' else pd.read_csv(p).to_string(index=False))

run(f'{SG} benchmark --dataset lira --preset trust --models todorovic2022_cnn,lampe2023_gru')
run(f'{SG} ablation --dataset lira --preset trust --variants safegrip_data_only,safegrip_no_projection,safegrip_no_uq,safegrip_no_physics_loss,safegrip_no_calibration,safegrip_no_temporal,safegrip')

health=json.loads((REPO/'results/lira_trust/result_health.json').read_text())
print('\nSCIENTIFIC HEALTH:\n',json.dumps(health,indent=2))
print('\nMAIN METRICS:\n',pd.read_csv(REPO/'results/lira_trust/metrics.csv').to_string(index=False))
print('\nSANITY BASELINES:\n',pd.read_csv(REPO/'results/lira_trust/sanity_baselines.csv').to_string(index=False))
print('\nABLATION:\n',pd.read_csv(REPO/'results/lira_ablation_trust/ablation_metrics.csv').to_string(index=False))

stage=WORK/'_safegrip_trust_stage'; shutil.rmtree(stage,ignore_errors=True); stage.mkdir()
shutil.copytree(REPO/'results/lira_trust',stage/'lira_trust')
shutil.copytree(REPO/'results/lira_ablation_trust',stage/'lira_ablation_trust')
diag=stage/'diagnostics'; diag.mkdir()
for name in ['lira_signal_audit.csv','lira_physics_audit.json','lira_friction_schema_report.csv','lira_stream_assembly_report.json','lira_alignment_report.csv','lira_preprocessing_report.json']:
    p=proc/name
    if p.exists(): shutil.copy2(p,diag/name)
shutil.copy2(REPO/'configs/kaggle_trust.yaml',stage/'kaggle_trust.yaml')
shutil.make_archive(str(ZIP.with_suffix('')),'zip',root_dir=stage); shutil.rmtree(stage)
print('\nRESULT ZIP:',ZIP)
try:
    from IPython.display import FileLink,display
    display(FileLink(str(ZIP)))
except Exception: pass


if health.get("status") != "PASS":
    raise RuntimeError("TRUST RUN DID NOT PASS scientific health gates. Use the exported diagnostics to fix/interpret the result; do not report it as a paper result.")
