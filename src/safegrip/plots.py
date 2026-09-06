from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _bar(df,col,path,ylabel):
    if col not in df: return
    fig,ax=plt.subplots(figsize=(10,5)); z=df.sort_values(col)
    ax.bar(z.model,z[col]); ax.set_ylabel(ylabel); ax.tick_params(axis="x",rotation=45); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def make_plots(results_dir):
    r=Path(results_dir); p=pd.read_csv(r/"predictions.csv"); m=pd.read_csv(r/"metrics.csv")
    _bar(m,"rmse",r/"rmse.png","RMSE")
    _bar(m,"unsafe_overestimate_rate_005",r/"unsafe_overestimate_rate.png","Unsafe overestimate rate (>0.05)")
    if "physics_lower" in p:
        width=1.3-p.physics_lower.to_numpy(); fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(p.physics_lower,width,s=5,alpha=.25)
        ax.set_xlabel("Calibrated physical lower bound"); ax.set_ylabel("Identified-set width"); fig.tight_layout(); fig.savefig(r/"identified_set_width.png",dpi=180); plt.close(fig)
    if "safegrip" in p:
        y=p.y_true.to_numpy(); q=p.safegrip.to_numpy(); fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(y,q,s=5,alpha=.25)
        lo=min(y.min(),q.min()); hi=max(y.max(),q.max()); ax.plot([lo,hi],[lo,hi]); ax.set_xlabel("Reference friction"); ax.set_ylabel("SafeGrip estimate"); fig.tight_layout(); fig.savefig(r/"safegrip_scatter.png",dpi=180); plt.close(fig)


def make_ablation_plots(results_dir):
    r=Path(results_dir); m=pd.read_csv(r/"ablation_metrics.csv")
    _bar(m,"rmse",r/"ablation_rmse.png","RMSE")
    _bar(m,"lower_violation_rate",r/"ablation_lower_violation.png","Lower-bound violation rate")
