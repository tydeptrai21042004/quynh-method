from __future__ import annotations
import csv, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .utils import normalize_name, ensure_dir
from .physics import vehicle_level_lower_bound


def read_table(path: Path) -> pd.DataFrame:
    last=None
    for enc in ("utf-8-sig","utf-8","latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except Exception as e: last=e
    raise last


def _normmap(df): return {c: normalize_name(c) for c in df.columns}

def find_col(df: pd.DataFrame, groups, exclude=()):
    nm=_normmap(df)
    excludes=[normalize_name(x) for x in exclude]
    # groups = iterable of alternatives; each alternative can be tuple tokens all required.
    for alt in groups:
        toks=alt if isinstance(alt,(tuple,list)) else (alt,)
        toks=[normalize_name(t) for t in toks]
        for c,n in nm.items():
            if all(t in n for t in toks) and not any(x in n for x in excludes):
                return c
    return None

CANONICAL = {
 "time": [("timestamp",),("time",),("tid",)],
 "distance": [("totaldist",),("distance",),("dist",)],
 "lat": [("latitude",),("gps","lat"),("lat",)],
 "lon": [("longitude",),("gps","lon"),("lon",)],
 "speed": [("vehicle","speed"),("carspeed",),("vehspd",),("speed",)],
 "ax": [("longitudinal","acc"),("acceleration","x"),("accel","x"),("ax",)],
 "ay": [("lateral","acc"),("acceleration","y"),("accel","y"),("ay",)],
 "yaw_rate": [("yaw","rate"),("yawrate",)],
 "steer": [("steer","angle"),("steering",),("steer",)],
 "wheel_fl": [("wheel","speed","front","left"),("wheelspeedfl",),("fl","wheel","speed")],
 "wheel_fr": [("wheel","speed","front","right"),("wheelspeedfr",),("fr","wheel","speed")],
 "wheel_rl": [("wheel","speed","rear","left"),("wheelspeedrl",),("rl","wheel","speed")],
 "wheel_rr": [("wheel","speed","rear","right"),("wheelspeedrr",),("rr","wheel","speed")],
 "torque": [("estimated","torque"),("requested","torque"),("drive","torque"),("motortorque",),("torque",)],
 "brake_torque": [("brake","wheel","torque"),("brake","torque")],
 "pressure_fl": [("pressure","front","left"),("tirepressurefl",)],
 "pressure_fr": [("pressure","front","right"),("tirepressurefr",)],
 "pressure_rl": [("pressure","rear","left"),("tirepressurerl",)],
 "pressure_rr": [("pressure","rear","right"),("tirepressurerr",)],
}


def _to_numeric(s): return pd.to_numeric(s, errors="coerce")

def canonical_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    out=pd.DataFrame(index=df.index)
    for key, pats in CANONICAL.items():
        c=find_col(df,pats)
        if c is not None: out[key]=_to_numeric(df[c])
    return out


def canonical_friction(df: pd.DataFrame) -> pd.DataFrame:
    out=pd.DataFrame(index=df.index)
    aliases={
      "time": [("tid",),("time",)], "distance": [("totaldist",),("distance",)],
      "lat": [("lat",)], "lon": [("lon",)],
      "mu_l": [("muv",),("frictioncoefficient","left")],
      "mu_r": [("muh",),("frictioncoefficient","right")],
      "fz_l": [("fvertikalv",),("vertical","left")],
      "fz_r": [("fvertikalh",),("vertical","right")],
      "fx_l": [("ffriksjonv",),("frictional","left")],
      "fx_r": [("ffriksjonh",),("frictional","right")],
      "slip_l": [("slipv",),("sliprate","left")],
      "slip_r": [("sliph",),("sliprate","right")],
    }
    for k,p in aliases.items():
        c=find_col(df,p)
        if c is not None: out[k]=_to_numeric(df[c])
    mus=[c for c in ["mu_l","mu_r"] if c in out]
    if mus: out["mu_ref"]=out[mus].mean(axis=1)
    return out


def haversine_project(lat, lon, lat0=None):
    lat=np.asarray(lat,float); lon=np.asarray(lon,float)
    lat0=np.nanmedian(lat) if lat0 is None else lat0
    R=6371000.0
    x=np.deg2rad(lon)*R*np.cos(np.deg2rad(lat0)); y=np.deg2rad(lat)*R
    return np.c_[x,y]


def spatial_align(car: pd.DataFrame, ref: pd.DataFrame, max_m=35.0) -> pd.DataFrame:
    if all(c in car for c in ("lat","lon")) and all(c in ref for c in ("lat","lon")):
        cvalid=car[["lat","lon"]].notna().all(axis=1); rvalid=ref[["lat","lon"]].notna().all(axis=1)
        c=car.loc[cvalid].copy(); r=ref.loc[rvalid].copy()
        lat0=np.nanmedian(np.r_[c.lat.values,r.lat.values])
        tree=cKDTree(haversine_project(r.lat,r.lon,lat0))
        dist,idx=tree.query(haversine_project(c.lat,c.lon,lat0),k=1)
        rr=r.iloc[idx].reset_index(drop=True)
        cc=c.reset_index(drop=True)
        for col in rr.columns:
            if col not in cc or col in ("mu_ref","mu_l","mu_r","fz_l","fz_r","fx_l","fx_r","slip_l","slip_r"):
                cc[f"ref_{col}" if col in cc else col]=rr[col].values
        cc["match_distance_m"]=dist
        return cc.loc[cc.match_distance_m<=max_m].reset_index(drop=True)
    if "distance" in car and "distance" in ref:
        c=car.dropna(subset=["distance"]).sort_values("distance")
        r=ref.dropna(subset=["distance"]).sort_values("distance")
        return pd.merge_asof(c,r,on="distance",direction="nearest",suffixes=("","_ref")).reset_index(drop=True)
    raise RuntimeError("LiRA alignment requires GPS or distance columns; inspect raw schema and extend aliases if upstream names changed.")


def prepare_lira(raw: str|Path, out: str|Path, cfg: dict) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out)
    fric=list(raw.rglob("*fric_custom*.csv")) or [p for p in raw.rglob("*.csv") if "fric" in p.name.lower()]
    car=list(raw.rglob("task_7505*.txt")) or list(raw.rglob("*.txt"))
    if not fric or not car:
        raise FileNotFoundError(f"Expected LiRA *fric_custom*.csv and task_7505*.txt under {raw}; found {len(fric)} friction, {len(car)} car files")
    ref=pd.concat([canonical_friction(read_table(p)) for p in fric],ignore_index=True)
    cars=[]
    for p in car:
        x=canonical_vehicle(read_table(p)); x["source_file"]=p.name; cars.append(x)
    veh=pd.concat(cars,ignore_index=True)
    df=spatial_align(veh,ref,max_m=35.0)
    if "mu_ref" not in df:
        raise RuntimeError("Could not identify LiRA VIAFRIK mu columns. See schema_report.json.")
    # Unit heuristics: km/h -> m/s; g -> m/s2 when clearly needed.
    if "speed" in df and np.nanmedian(np.abs(df.speed))>25: df["speed"]=df.speed/3.6
    for c in ("ax","ay"):
        if c in df and np.nanpercentile(np.abs(df[c].dropna()),95)<3.0 and np.nanpercentile(np.abs(df[c].dropna()),95)>0.2:
            # Keep SI by default; this heuristic only catches obviously g-normalized logs near +/-1.
            pass
    for needed in ("speed","ax","ay"):
        if needed not in df: df[needed]=0.0
    v=cfg["vehicle"]
    df["physics_lower_raw"]=vehicle_level_lower_bound(df.ax,df.ay,df.speed,
      mass=v["mass_kg"],g=v["gravity"],crr=v["crr"],rho_air=v["rho_air"],cdA=v["cdA_m2"],
      accel_error=v["accel_error_ms2"],external_force_margin=v["external_force_margin_n"],
      vertical_force_margin=v["vertical_force_margin_n"])
    # Strict production-sensor whitelist; GPS is used only for alignment/splitting, never as a model feature.
    feats=[c for c in CANONICAL if c in df and c not in ("time","distance","lat","lon")]
    if len(feats)<3:
        raise RuntimeError(f"Only {feats} production features resolved. Extend CANONICAL aliases for this LiRA revision.")
    keep=feats+[c for c in ["distance","lat","lon","match_distance_m","mu_ref","physics_lower_raw","source_file"] if c in df]
    df=df[keep].replace([np.inf,-np.inf],np.nan)
    df=df.dropna(subset=["mu_ref"]).reset_index(drop=True)
    # Light interpolation for features only, never labels.
    df[feats]=df[feats].interpolate(limit_direction="both").ffill().bfill()
    df=df.dropna(subset=feats).reset_index(drop=True)
    # Spatial/order coordinate for leakage-safe split.
    if "distance" in df and df.distance.notna().sum()>0:
        order=df.distance.rank(method="first",pct=True)
    else:
        order=pd.Series(np.linspace(0,1,len(df),endpoint=False),index=df.index)
    s=cfg["split"]; a=s["train"]; b=a+s["calibration"]; c=b+s["validation"]
    df["split"]=np.where(order<a,"train",np.where(order<b,"calibration",np.where(order<c,"validation","test")))
    path=out/"lira_aligned.csv"; df.to_csv(path,index=False)
    (out/"lira_features.json").write_text(json.dumps(feats,indent=2),encoding="utf-8")
    return path


def prepare_kuleuven(raw: str|Path, out: str|Path) -> Path:
    """Canonicalize KU Leuven files for optional wheel-force virtual sensing.

    Because the source exposes many maneuvers, we concatenate CSVs and resolve WFT
    columns by names. The generated file keeps only runs where at least one WFT force is found.
    """
    raw=Path(raw); out=Path(out); ensure_dir(out)
    rows=[]
    for p in raw.rglob("*.csv"):
        try: df=read_table(p)
        except Exception: continue
        n=_normmap(df)
        def find(tokens):
            for col,nm in n.items():
                if all(t in nm for t in tokens): return col
        # Generic Kistler/WFT aliases.
        targets={}
        for side in ("fl","fr"):
            for comp in ("fx","fy","fz"):
                pats=[(side,comp),("wft",side,comp),("roadyn",side,comp)]
                col=None
                for pat in pats:
                    col=find(pat)
                    if col: break
                if col: targets[f"{comp}_{side}"]=pd.to_numeric(df[col],errors="coerce")
        if not targets: continue
        x=canonical_vehicle(df)
        for k,v in targets.items(): x[k]=v
        x["run"]=p.stem
        rows.append(x)
    if not rows:
        raise RuntimeError("No KU Leuven wheel-force columns resolved. Raw files are present; update alias resolver using the included source readme/column names.")
    allx=pd.concat(rows,ignore_index=True).dropna(how="all")
    path=out/"kuleuven_wft.csv"; allx.to_csv(path,index=False); return path


def prepare_kit(raw: str|Path, out: str|Path) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out)
    rows=[]
    for p in list(raw.rglob("*.csv"))+list(raw.rglob("*.txt")):
        try: df=read_table(p)
        except Exception: continue
        nm=_normmap(df)
        def choose(*alts):
            for alt in alts:
                toks=[normalize_name(t) for t in alt]
                for col,name in nm.items():
                    if all(t in name for t in toks): return col
        fx=choose(("fx",),("longitudinal","force")); fy=choose(("fy",),("lateral","force")); fz=choose(("fz",),("vertical","force"),("normal","force"))
        if not fz or (not fx and not fy): continue
        x=pd.DataFrame()
        x["fx"]=pd.to_numeric(df[fx],errors="coerce") if fx else 0.0
        x["fy"]=pd.to_numeric(df[fy],errors="coerce") if fy else 0.0
        x["fz"]=pd.to_numeric(df[fz],errors="coerce")
        for key, alts in {
            "slip_ratio":(("slip","ratio"),("sr",)), "slip_angle":(("slip","angle"),("sa",)),
            "speed":(("speed",),("velocity",)), "pressure":(("pressure",),), "camber":(("camber",),("inclination",))}.items():
            cc=choose(*alts)
            if cc: x[key]=pd.to_numeric(df[cc],errors="coerce")
        x["source_file"]=p.name
        rows.append(x)
    if not rows: raise RuntimeError("No KIT force tables resolved; inspect extracted archive and extend aliases.")
    z=pd.concat(rows,ignore_index=True).dropna(subset=["fz"])
    z=z[np.abs(z.fz)>10].copy(); z["mu_util"]=np.hypot(z.fx,z.fy)/np.abs(z.fz)
    path=out/"kit_force.csv"; z.to_csv(path,index=False); return path


def make_synthetic(out: str|Path, n=12000, seed=20260905) -> Path:
    rng=np.random.default_rng(seed); out=Path(out); ensure_dir(out)
    t=np.arange(n)/20.0
    block=max(50,n//12); mu=np.repeat(rng.uniform(.25,1.1,size=max(1,n//block+1)),block)[:n]
    speed=np.clip(18+5*np.sin(t/37)+rng.normal(0,.5,n),2,35)
    excitation=np.clip(rng.beta(1.2,4,size=n),0,1)
    theta=rng.uniform(-np.pi,np.pi,n)
    rho=mu*excitation
    ax=9.81*rho*np.cos(theta)+rng.normal(0,.08,n)
    ay=9.81*rho*np.sin(theta)+rng.normal(0,.08,n)
    steer=np.clip(ay/np.maximum(speed,2)**2*2.7,-.5,.5)+rng.normal(0,.01,n)
    yaw=ay/np.maximum(speed,2)+rng.normal(0,.01,n)
    base=speed/.31*60/(2*np.pi)
    wheels=np.column_stack([base+rng.normal(0,2,n) for _ in range(4)])
    torque=np.maximum(0,150*ax)+rng.normal(0,20,n)
    df=pd.DataFrame({"speed":speed,"ax":ax,"ay":ay,"yaw_rate":yaw,"steer":steer,
      "wheel_fl":wheels[:,0],"wheel_fr":wheels[:,1],"wheel_rl":wheels[:,2],"wheel_rr":wheels[:,3],
      "torque":torque,"mu_ref":mu,"distance":np.cumsum(speed/20)})
    df["physics_lower_raw"]=np.clip(rho-rng.uniform(0,.03,n),0,1.3)
    q=np.linspace(0,1,n,endpoint=False); df["split"]=np.where(q<.6,"train",np.where(q<.7,"calibration",np.where(q<.8,"validation","test")))
    path=out/"synthetic.csv"; df.to_csv(path,index=False); return path


def _write_manifest(paths, out_path: Path, root: Path):
    rows=[]
    for p in paths:
        try: size=p.stat().st_size
        except OSError: size=None
        rows.append({"path":str(p.relative_to(root)),"suffix":p.suffix.lower(),"size_bytes":size})
    pd.DataFrame(rows).to_csv(out_path,index=False)
    return out_path


def prepare_deep_dynamics(raw: str|Path, out: str|Path) -> Path:
    """Collect open Deep Dynamics/IAC/BayesRace tables without inventing friction labels."""
    raw=Path(raw); out=Path(out); ensure_dir(out)
    csvs=list(raw.rglob("*.csv"))
    if not csvs: return _write_manifest(list(raw.rglob("*")),out/"deep_dynamics_manifest.csv",raw)
    rows=[]
    for p in csvs:
        try: df=read_table(p)
        except Exception: continue
        veh=canonical_vehicle(df)
        if len(veh.columns)<2: continue
        veh["source_file"]=str(p.relative_to(raw)); rows.append(veh)
    if not rows: return _write_manifest(csvs,out/"deep_dynamics_manifest.csv",raw)
    z=pd.concat(rows,ignore_index=True); path=out/"deep_dynamics_vehicle.csv"; z.to_csv(path,index=False); return path


def _load_numpy_pair(folder: Path):
    """Load comma2k19's t/value arrays; handles .npy and extension-less numpy files."""
    if not folder.exists(): return None
    def load_candidate(names):
        for n in names:
            p=folder/n
            if p.exists():
                try: return np.load(p,allow_pickle=False)
                except Exception:
                    try: return np.fromfile(p,dtype=np.float64)
                    except Exception: pass
        return None
    t=load_candidate(["t","t.npy","time","time.npy"])
    v=load_candidate(["value","value.npy","values","values.npy"])
    if t is None or v is None: return None
    return np.asarray(t),np.asarray(v)


def prepare_comma2k19(raw: str|Path, out: str|Path) -> Path:
    """Prepare the bundled 1-minute comma2k19 example when present.

    This auxiliary dataset has no friction ground truth and is therefore never
    included in the direct friction benchmark. It is suitable for SSL/domain tests.
    """
    raw=Path(raw); out=Path(out); ensure_dir(out)
    plogs=[p for p in raw.rglob("processed_log") if p.is_dir()]
    records=[]
    for pl in plogs:
        specs={
            "speed":["CAN/car_speed","car_speed"],
            "steer":["CAN/steering_angle","steering_angle"],
            "wheels":["CAN/wheel_speeds","wheel_speeds"],
            "acc":["IMU/acceleration","acceleration"],
            "gyro":["IMU/gyro","gyro"],
        }
        loaded={}
        for key,rels in specs.items():
            for rel in rels:
                pair=_load_numpy_pair(pl/rel)
                if pair is not None: loaded[key]=pair; break
        if "speed" not in loaded: continue
        t0,v0=loaded["speed"]; t0=np.asarray(t0).reshape(-1); v0=np.asarray(v0).reshape(-1)
        n=min(len(t0),len(v0)); t0=t0[:n]; v0=v0[:n]
        frame=pd.DataFrame({"time":t0,"speed":v0})
        for key,(tt,vv) in loaded.items():
            if key=="speed": continue
            tt=np.asarray(tt).reshape(-1); vv=np.asarray(vv)
            if len(tt)==0 or len(vv)==0: continue
            if vv.ndim==1:
                frame[key]=np.interp(t0,tt[:len(vv)],vv[:len(tt)])
            else:
                m=min(len(tt),len(vv)); tt=tt[:m]; vv=vv[:m]
                for j in range(vv.shape[1]): frame[f"{key}_{j}"]=np.interp(t0,tt,vv[:,j])
        frame["source_segment"]=str(pl.parent.relative_to(raw)); records.append(frame)
    if not records: return _write_manifest(list(raw.rglob("*")),out/"comma2k19_manifest.csv",raw)
    z=pd.concat(records,ignore_index=True); path=out/"comma2k19_example.csv"; z.to_csv(path,index=False); return path


def prepare_extreme_road(raw: str|Path, out: str|Path) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out)
    exts={".jpg",".jpeg",".png",".bmp",".webp"}; imgs=[p for p in raw.rglob("*") if p.suffix.lower() in exts]
    rows=[]
    for p in imgs:
        rel=p.relative_to(raw); parts=[x.lower() for x in rel.parts]
        label=rel.parent.name
        rows.append({"path":str(rel),"label":label,"filename":p.name})
    path=out/"extreme_road_images.csv"; pd.DataFrame(rows).to_csv(path,index=False); return path


def prepare_bicycle_tire(raw: str|Path, out: str|Path) -> Path:
    """Create a transparent file manifest; YAML/MAT data remain in source form.

    We intentionally do not coerce bicycle test-rig measurements into passenger-car
    mu labels. This dataset is auxiliary mechanics evidence only.
    """
    raw=Path(raw); out=Path(out); ensure_dir(out)
    return _write_manifest([p for p in raw.rglob("*") if p.is_file()],out/"bicycle_tire_manifest.csv",raw)


def prepare_mendeley_friction(raw: str|Path, out: str|Path) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out); tables=[]
    files=list(raw.rglob("*.xlsx"))+list(raw.rglob("*.xls"))+list(raw.rglob("*.csv"))
    for p in files:
        try:
            if p.suffix.lower() in (".xlsx",".xls"):
                book=pd.read_excel(p,sheet_name=None)
                for sheet,df in book.items():
                    df=df.copy(); df["source_sheet"]=sheet; df["source_file"]=p.name; tables.append(df)
            else:
                df=read_table(p); df["source_file"]=p.name; tables.append(df)
        except Exception: continue
    if not tables: raise RuntimeError("No readable Mendeley friction workbook/CSV found")
    z=pd.concat(tables,ignore_index=True,sort=False)
    # Preserve source columns but add canonical friction/speed/surface when discoverable.
    mu=find_col(z,[("friction","coefficient"),("coefficient","friction"),("mu",)])
    speed=find_col(z,[("vehicle","speed"),("speed",),("velocity",)])
    surface=find_col(z,[("road","surface"),("surface",),("pavement",)])
    if mu: z["mu_ref"]=pd.to_numeric(z[mu],errors="coerce")
    if speed: z["speed_canonical"]=pd.to_numeric(z[speed],errors="coerce")
    if surface: z["surface_canonical"]=z[surface].astype(str)
    path=out/"mendeley_friction.csv"; z.to_csv(path,index=False); return path


def prepare_dataset(name: str, raw: str|Path, out: str|Path, cfg: dict | None = None) -> Path:
    name=name.lower(); cfg=cfg or {}
    if name=="lira": return prepare_lira(raw,out,cfg)
    if name=="kit": return prepare_kit(raw,out)
    if name=="kuleuven": return prepare_kuleuven(raw,out)
    if name=="deep_dynamics": return prepare_deep_dynamics(raw,out)
    if name=="comma2k19": return prepare_comma2k19(raw,out)
    if name=="extreme_road": return prepare_extreme_road(raw,out)
    if name=="bicycle_tire": return prepare_bicycle_tire(raw,out)
    if name=="mendeley_friction": return prepare_mendeley_friction(raw,out)
    if name=="synthetic": return make_synthetic(out,seed=(cfg or {}).get("seed",20260905))
    raise ValueError(name)
