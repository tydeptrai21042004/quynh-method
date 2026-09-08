# SafeGrip / quynh-method — single Kaggle development cell
# Internet: ON | Accelerator: T4 GPU recommended
# Runs LiRA -> SafeGrip + Todorovic CNN + Lampe GRU -> 7 ablations -> ZIP.

import os, sys, json, shutil, subprocess, textwrap
from pathlib import Path

WORK = Path('/kaggle/working')
REPO = WORK / 'quynh-method'
OUTZIP = WORK / 'safegrip_lira_small_results.zip'

def run(cmd, cwd=None):
    print('\n' + '='*100)
    print('RUN:', cmd)
    print('='*100)
    subprocess.run(cmd, shell=True, check=True, cwd=str(cwd) if cwd else None, executable='/bin/bash')

# 1) Fresh clone + install
shutil.rmtree(REPO, ignore_errors=True)
if OUTZIP.exists(): OUTZIP.unlink()
run('git clone --depth 1 https://github.com/tydeptrai21042004/quynh-method.git', WORK)
os.chdir(REPO)
commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
print('Git commit:', commit)
run(f'{sys.executable} -m pip install -q --upgrade pip setuptools wheel', REPO)
run(f'{sys.executable} -m pip install -q -e .', REPO)

import yaml, pandas as pd, torch
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

# 2) Hotfix VIAFRIK schema parsing on older GitHub commits.
# prepare_lira() resolves canonical_friction at runtime, so overriding the global
# function at the end of data.py is sufficient and does not change model logic.
data_py = REPO / 'src/safegrip/data.py'
source = data_py.read_text(encoding='utf-8')
if 'KAGGLE_VIAFRIK_SCHEMA_HOTFIX_V032' not in source:
    hotfix = r'''

# KAGGLE_VIAFRIK_SCHEMA_HOTFIX_V032
# Supports official LiRA μ_V/μ_H fields plus custom/encoded header variants.
def _via_num_v032(s):
    base = pd.to_numeric(s, errors='coerce')
    if base.notna().sum() >= max(3, int(0.8 * len(base))):
        return base
    txt = pd.Series(s, index=getattr(s, 'index', None), dtype='string').str.strip()
    txt = txt.str.replace('−','-',regex=False).str.replace(' ','',regex=False)
    if txt.str.contains(',', regex=False, na=False).any():
        alt = pd.to_numeric(txt.str.replace(',','.',regex=False), errors='coerce')
        if alt.notna().sum() > base.notna().sum():
            return alt
    return base

def _via_plausible_v032(s):
    x = _via_num_v032(s)
    x = x[np.isfinite(x)]
    if len(x) < max(3, len(s)//10):
        return False
    q01, q50, q99 = np.nanquantile(x, [0.01,0.50,0.99])
    return bool(q01 >= -0.05 and 0.02 <= q50 <= 2.0 and q99 <= 2.5)

def canonical_friction(df):
    out = pd.DataFrame(index=df.index)
    aliases = {
        'time': [('tid',),('timestamp',),('time',)],
        'distance': [('totaldist',),('distance',),('dist',)],
        'lat': [('latitude',),('lat',)],
        'lon': [('longitude',),('lon',)],
        'mu_l': [('muv',),('mu','left'),('frictioncoefficient','left'),('friction','left')],
        'mu_r': [('muh',),('mu','right'),('frictioncoefficient','right'),('friction','right')],
        'fz_l': [('fvertikalv',),('vertical','left')],
        'fz_r': [('fvertikalh',),('vertical','right')],
        'fx_l': [('ffriksjonv',),('frictional','left')],
        'fx_r': [('ffriksjonh',),('frictional','right')],
        'slip_l': [('slipv',),('sliprate','left')],
        'slip_r': [('sliph',),('sliprate','right')],
    }
    used = set()
    for k,pats in aliases.items():
        c = find_col(df,pats)
        if c is not None:
            used.add(c)
            out[k] = _time_seconds(df[c]) if k == 'time' else _via_num_v032(df[c])

    missing = [k for k in ('mu_l','mu_r') if k not in out]
    if missing:
        nm = _normmap(df)
        semantic_tokens = ('muv','muh','muleft','muright','frictioncoefficient','frictioncoeff','frictionnumber','frictionvalue','friction')
        candidates = []
        for c,n in nm.items():
            if c in used: continue
            if any(tok in n for tok in semantic_tokens) and _via_plausible_v032(df[c]):
                candidates.append(c)
        exclude = ('time','tid','timestamp','dist','lat','lon','bearing','speed','kmh','kmt','velocity','slip','percent','pct','vertical','vertikal','force','friksjon','newton','wheel','mw','tw','index','id')
        if not candidates:
            scored=[]
            for c,n in nm.items():
                if c in used or any(tok in n for tok in exclude): continue
                if _via_plausible_v032(df[c]):
                    x=_via_num_v032(df[c]); finite=x[np.isfinite(x)]
                    scored.append((-len(finite)/max(len(x),1), float(np.nanstd(finite)), list(df.columns).index(c), c))
            scored.sort(); candidates=[x[-1] for x in scored[:2]]
        for target,c in zip(missing,candidates):
            out[target] = _via_num_v032(df[c]); used.add(c)

    mus=[c for c in ('mu_l','mu_r') if c in out]
    if mus: out['mu_ref']=out[mus].mean(axis=1,skipna=True)
    return out
'''
    data_py.write_text(source + hotfix, encoding='utf-8')
    print('Applied VIAFRIK schema hotfix v0.3.2')
else:
    print('VIAFRIK schema hotfix already present')

# 3) Small development configuration
cfg = yaml.safe_load((REPO/'configs/default.yaml').read_text(encoding='utf-8'))
cfg['sequence_length'] = 32
cfg['stride'] = 16
cfg.setdefault('benchmark',{})['common_warmup_samples'] = 32
cfg.setdefault('evaluation',{})['projection_parity_controls'] = False
cfg['evaluation']['seeds'] = [0]
cfg.setdefault('lira',{}).update({
    'match_max_m':10.0, 'heading_tolerance_deg':45.0, 'k_candidates':8,
    'enforce_monotonic':True, 'resample_hz':10.0,
    'sensor_merge_max_gap_s':0.5, 'gps_interp_max_gap_s':2.5,
    'resample_max_gap_s':0.25, 'interpolation_limit':5, 'split_guard_samples':0,
})
cfg['training']['epochs_quick'] = 3
cfg['training']['batch_size'] = 512
cfg['training']['patience'] = 3
cfg.setdefault('proposal',{}).update({'hidden':32,'tcn_blocks':2,'kernel_size':3,'batch_size':512})
small_cfg = REPO/'configs/kaggle_small.yaml'
small_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

# 4) Tests + download
run(f'{sys.executable} -m pytest -q', REPO)
run('safegrip --config configs/kaggle_small.yaml download --datasets lira', REPO)

# 5) Print exact downloaded VIAFRIK schema before preparation.
from safegrip.data import read_table, canonical_friction
raw = REPO/'data/raw/lira'
print('\n' + '='*100 + '\nVIAFRIK RAW SCHEMA CHECK\n' + '='*100)
fric_files = sorted([p for p in raw.rglob('*.csv') if 'fric' in p.name.lower()])
if not fric_files:
    raise RuntimeError('No LiRA friction CSV was downloaded.')
for p in fric_files:
    rdf = read_table(p)
    can = canonical_friction(rdf)
    print('\n', p.name)
    print('raw columns:', list(rdf.columns))
    print('canonical columns:', list(can.columns))
    print('finite mu_ref:', int(can['mu_ref'].notna().sum()) if 'mu_ref' in can else 0)
    if 'mu_ref' not in can or can['mu_ref'].notna().sum() == 0:
        raise RuntimeError(f'VIAFRIK mu_ref still unresolved in {p.name}; raw columns printed above.')

# 6) Prepare real LiRA
run('safegrip --config configs/kaggle_small.yaml prepare --dataset lira', REPO)
aligned = REPO/'data/processed/lira/lira_aligned.csv'
if not aligned.exists(): raise FileNotFoundError(aligned)
df = pd.read_csv(aligned)
print('\nPrepared rows:', len(df))
print('Split counts:\n', df['split'].value_counts().to_string())
required={'train','calibration','validation','test'}
missing=required-set(df['split'].astype(str))
if missing: raise RuntimeError(f'Missing split(s): {sorted(missing)}')

# 7) Proposal + exactly two baselines
run('safegrip --config configs/kaggle_small.yaml benchmark --dataset lira --preset quick --models todorovic2022_cnn,lampe2023_gru', REPO)

# 8) Full 7-component SafeGrip ablation
run('safegrip --config configs/kaggle_small.yaml ablation --dataset lira --preset quick --variants safegrip_data_only,safegrip_no_projection,safegrip_no_uq,safegrip_no_physics_loss,safegrip_no_calibration,safegrip_no_temporal,safegrip', REPO)

# 9) Show tables
main_path=REPO/'results/lira_quick/metrics.csv'
abl_path=REPO/'results/lira_ablation_quick/ablation_metrics.csv'
main=pd.read_csv(main_path); abl=pd.read_csv(abl_path)
print('\n'+'='*100+'\nMAIN: SAFEGRIP + CNN + GRU\n'+'='*100)
print(main.to_string(index=False))
print('\n'+'='*100+'\nABLATION\n'+'='*100)
print(abl.to_string(index=False))

# 10) ZIP results + diagnostics + exact config/commit (no raw 200 MB data)
stage=WORK/'_safegrip_result_stage'; shutil.rmtree(stage,ignore_errors=True); stage.mkdir()
shutil.copytree(REPO/'results', stage/'results')
summary=stage/'summary'; summary.mkdir()
main.to_csv(summary/'main_metrics.csv',index=False)
abl.to_csv(summary/'ablation_metrics.csv',index=False)
shutil.copy2(small_cfg, summary/'kaggle_small.yaml')
(summary/'git_commit.txt').write_text(commit+'\n',encoding='utf-8')
(summary/'environment.json').write_text(json.dumps({
    'python':sys.version,'torch':torch.__version__,'cuda':bool(torch.cuda.is_available()),
    'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
},indent=2),encoding='utf-8')
proc=REPO/'data/processed/lira'
for name in ('lira_friction_schema_report.csv','lira_stream_assembly_report.json','lira_alignment_report.csv','lira_preprocessing_report.json'):
    p=proc/name
    if p.exists(): shutil.copy2(p,summary/name)
shutil.make_archive(str(OUTZIP.with_suffix('')), 'zip', root_dir=stage)
shutil.rmtree(stage)
print('\nDONE:', OUTZIP)
try:
    from IPython.display import FileLink, display
    display(FileLink(str(OUTZIP)))
except Exception:
    pass
