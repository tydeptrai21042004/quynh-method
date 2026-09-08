#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ready=ROOT/'results/lira_paper/paper_readiness.json'
if not ready.exists():
    raise SystemExit('paper_readiness.json is missing. Run scripts/check_paper_readiness.py first.')
report=json.loads(ready.read_text())
if report.get('status')!='PAPER_READY':
    raise SystemExit('Result status is not PAPER_READY; refusing to package it as a paper release.')
stage=ROOT/'results/_paper_release_stage'
shutil.rmtree(stage,ignore_errors=True); stage.mkdir(parents=True)
for rel in ['results/lira_paper','results/lira_ablation_paper','results/lira_excitation','results/lira_robustness','results/lira_tuning','results/lira_baseline_tuning','data/processed/lira']:
    src=ROOT/rel
    if src.exists():
        dst=stage/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(src,dst)
for rel in ['configs/default.yaml','RESEARCH_PROTOCOL.md','LITERATURE_BASELINES.md','FINAL_RESEARCH_RELEASE.md']:
    src=ROOT/rel
    if src.exists():
        dst=stage/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
out=ROOT/'results/safegrip_paper_release'
zip_path=shutil.make_archive(str(out),'zip',root_dir=stage)
shutil.rmtree(stage)
print(zip_path)
