from __future__ import annotations
import hashlib, json, os, re, shutil, tarfile, zipfile
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from .utils import ensure_dir
from .datasets import DATASET_REGISTRY

UA = {"User-Agent": "SafeGripOpen/0.2 academic-research"}

SOURCE = {
    "lira": {"kind":"figshare", "article_id":23096600},
    "kuleuven": {"kind":"dataverse", "server":"https://rdr.kuleuven.be", "pid":"doi:10.48804/PHMF9D"},
    "kit": {"kind":"radar", "landing":"https://radar.kit.edu/radar/en/dataset/p0rr2jc5wmf0drf8"},
    "deep_dynamics": {"kind":"github", "repo":"linklab-uva/deep-dynamics"},
    "comma2k19": {"kind":"github", "repo":"commaai/comma2k19"},
    "extreme_road": {"kind":"github", "repo":"sean-shiyuez/Extreme-Road-Image-Dataset"},
    "bicycle_tire": {"kind":"zenodo", "record":7866646},
    "mendeley_friction": {"kind":"mendeley", "slug":"trrcrgzg75", "version":1,
                           "landing":"https://data.mendeley.com/datasets/trrcrgzg75/1"},
}


def _download(url: str, dest: Path, *, session=None, headers=None, timeout=120) -> Path:
    session = session or requests.Session(); h = dict(UA)
    if headers: h.update(headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume = part.stat().st_size if part.exists() else 0
    if resume: h["Range"] = f"bytes={resume}-"
    with session.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=h) as r:
        if r.status_code == 416 and part.exists():
            part.replace(dest); return dest
        r.raise_for_status()
        mode = "ab" if resume and r.status_code == 206 else "wb"
        if mode == "wb": resume = 0
        total = int(r.headers.get("content-length", 0)) + resume
        with open(part, mode) as f, tqdm(total=total or None, initial=resume, unit="B", unit_scale=True, desc=dest.name) as p:
            for chunk in r.iter_content(1024 * 1024):
                if chunk: f.write(chunk); p.update(len(chunk))
    part.replace(dest); return dest


def _safe_extract_zip(path: Path, out: Path):
    root = out.resolve()
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            target = (out / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        z.extractall(out)


def _extract(path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True); low = path.name.lower()
    try:
        if low.endswith(".zip"): _safe_extract_zip(path, out)
        elif low.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
            with tarfile.open(path) as t:
                root=out.resolve()
                for m in t.getmembers():
                    target=(out/m.name).resolve()
                    if root not in target.parents and target != root:
                        raise RuntimeError(f"Unsafe archive path: {m.name}")
                t.extractall(out)
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"[download] warning: {path.name} is not an archive: {e}")


def _write_source(name: str, out: Path, extra=None):
    meta = dict(DATASET_REGISTRY[name]); meta.update(SOURCE.get(name, {}))
    if extra: meta.update(extra)
    (out / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def download_lira(out: Path) -> None:
    api = f"https://api.figshare.com/v2/articles/{SOURCE['lira']['article_id']}"
    r = requests.get(api, headers=UA, timeout=60); r.raise_for_status()
    files = r.json().get("files", [])
    if not files: raise RuntimeError("Figshare returned no files for LiRA")
    for item in files:
        dest = out / item["name"]
        if not (dest.exists() and dest.stat().st_size == item.get("size", -1)):
            _download(item["download_url"], dest)
        _extract(dest, out)
    _write_source("lira", out)


def _dataverse_metadata(server: str, pid: str) -> dict:
    r = requests.get(f"{server}/api/datasets/:persistentId/", params={"persistentId":pid}, headers=UA, timeout=60)
    r.raise_for_status(); payload=r.json(); return payload.get("data", payload)


def download_kuleuven(out: Path, csv_only: bool = False) -> None:
    cfg=SOURCE["kuleuven"]; meta=_dataverse_metadata(cfg["server"], cfg["pid"])
    files=meta.get("latestVersion",{}).get("files",[])
    if not files: raise RuntimeError("KU Leuven Dataverse metadata returned no files")
    for entry in files:
        df=entry.get("dataFile",{}); name=df.get("filename",""); label=entry.get("directoryLabel","")
        if csv_only and not name.lower().endswith((".csv",".txt",".md")): continue
        file_id=df.get("id"); size=df.get("filesize");
        if not file_id: continue
        dest=(out/label/name) if label else (out/name)
        if dest.exists() and (not size or dest.stat().st_size==size): continue
        headers={}; token=os.getenv("KULEUVEN_API_TOKEN")
        if token: headers["X-Dataverse-key"]=token
        try: _download(f"{cfg['server']}/api/access/datafile/{file_id}",dest,headers=headers)
        except requests.HTTPError as e:
            if getattr(e.response,"status_code",None) in (401,403):
                raise RuntimeError("KU Leuven requires its guestbook/terms to be accepted. Accept them on the DOI page; if needed set KULEUVEN_API_TOKEN, then rerun. SafeGrip never bypasses repository terms.") from e
            raise
        _extract(dest,out)
    _write_source("kuleuven",out)


def _find_radar_download_url(html: str, base: str) -> str | None:
    soup=BeautifulSoup(html,"html.parser"); candidates=[]
    for a in soup.find_all("a",href=True):
        href=urljoin(base,a["href"]); text=a.get_text(" ",strip=True).lower(); low=href.lower()
        score=int("download" in low)+int("download" in text)+int(any(x in low for x in [".zip","archive"]))
        if score: candidates.append((score,href))
    return sorted(candidates,reverse=True)[0][1] if candidates else None


def download_kit(out: Path) -> None:
    cfg=SOURCE["kit"]; s=requests.Session(); r=s.get(cfg["landing"],headers=UA,timeout=60); r.raise_for_status()
    url=_find_radar_download_url(r.text,r.url)
    if not url: raise RuntimeError("Could not discover the KIT RADAR download URL; upstream markup may have changed.")
    dest=_download(url,out/"kit_dataset.download",session=s)
    magic=dest.read_bytes()[:4]
    if magic.startswith(b"PK"):
        proper=dest.with_suffix(".zip"); dest.replace(proper); dest=proper
    _extract(dest,out); _write_source("kit",out)


def _github_default_branch(repo: str) -> str:
    r=requests.get(f"https://api.github.com/repos/{repo}",headers=UA,timeout=60); r.raise_for_status()
    return r.json().get("default_branch","main")


def download_github_repo(name: str, out: Path) -> None:
    repo=SOURCE[name]["repo"]; branch=_github_default_branch(repo)
    url=f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    dest=_download(url,out/f"{repo.split('/')[-1]}-{branch}.zip")
    _extract(dest,out); _write_source(name,out,{"default_branch":branch})


def download_zenodo(name: str, out: Path, *, yaml_only: bool = False) -> None:
    record=SOURCE[name]["record"]; r=requests.get(f"https://zenodo.org/api/records/{record}",headers=UA,timeout=60); r.raise_for_status()
    files=r.json().get("files",[])
    if yaml_only:
        files=[f for f in files if ".yaml" in f.get("key","").lower() or "readme" in f.get("key","").lower()]
    for item in files:
        key=item.get("key") or item.get("filename"); links=item.get("links",{}); url=links.get("self") or links.get("content")
        if not key or not url: continue
        dest=out/key
        if not dest.exists(): _download(url,dest)
        _extract(dest,out)
    _write_source(name,out,{"record":record,"download_subset":"yaml+readme" if yaml_only else "all"})


def _mendeley_file_candidates(slug: str, version: int):
    """Try documented/current public endpoints, then landing-page embedded links."""
    endpoints=[
        f"https://data.mendeley.com/public-api/datasets/{slug}/versions/{version}/files",
        f"https://data.mendeley.com/public-api/datasets/{slug}/{version}/files",
    ]
    s=requests.Session()
    for ep in endpoints:
        r=s.get(ep,headers=UA,timeout=60)
        if not r.ok: continue
        try: payload=r.json()
        except ValueError: continue
        items=payload if isinstance(payload,list) else payload.get("files",payload.get("data",[]))
        found=[]
        for it in items or []:
            if not isinstance(it,dict): continue
            name=it.get("filename") or it.get("name")
            url=it.get("download_url") or it.get("downloadUrl") or it.get("url")
            if isinstance(it.get("links"),dict): url=url or it["links"].get("download") or it["links"].get("self")
            if name and url: found.append((name,url))
        if found: return found
    landing=SOURCE["mendeley_friction"]["landing"]
    r=s.get(landing,headers=UA,timeout=60); r.raise_for_status(); soup=BeautifulSoup(r.text,"html.parser")
    found=[]
    for a in soup.find_all("a",href=True):
        href=urljoin(r.url,a["href"]); label=(a.get("download","") or a.get_text(" ",strip=True))
        if any(x in href.lower() for x in ["download","public-files",".xlsx",".csv"]):
            name=Path(href.split("?")[0]).name or label or "mendeley_file"
            found.append((name,href))
    # Embedded JSON often contains an expiring public-file URL.
    for m in re.finditer(r'https?://[^"\\]+(?:\.xlsx|\.csv|public-files)[^"\\]*',r.text,re.I):
        url=m.group(0).replace("\\u0026","&").replace("\\/","/")
        found.append((Path(url.split("?")[0]).name or "mendeley_file",url))
    dedup={u:n for n,u in found}; return [(n,u) for u,n in dedup.items()]


def download_mendeley(out: Path) -> None:
    cfg=SOURCE["mendeley_friction"]; files=_mendeley_file_candidates(cfg["slug"],cfg["version"])
    # We intentionally download only the friction workbook/csv if discoverable.
    files=[(n,u) for n,u in files if "friction" in n.lower() or n.lower().endswith((".xlsx",".csv"))] or files
    if not files:
        raise RuntimeError("Mendeley public file URL could not be discovered automatically. The dataset is public, but its frontend/API may have changed; see DOI 10.17632/trrcrgzg75.1. No credentials are bypassed.")
    for n,u in files:
        n=re.sub(r'[^A-Za-z0-9._-]+','_',n)[:180] or "mendeley_file"
        if not (out/n).exists(): _download(u,out/n)
    _write_source("mendeley_friction",out)


def download_dataset(name: str, root: str | Path="data/raw", force: bool=False, full: bool=False) -> Path:
    name=name.lower();
    if name not in DATASET_REGISTRY: raise ValueError(f"unknown dataset: {name}. Choices: {', '.join(DATASET_REGISTRY)}")
    out=ensure_dir(Path(root)/name); marker=out/".complete"
    if marker.exists() and not force:
        print(f"[download] {name}: already complete"); return out
    if force and out.exists():
        for p in list(out.iterdir()):
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
    if name=="lira": download_lira(out)
    elif name=="kuleuven": download_kuleuven(out,csv_only=not full)
    elif name=="kit": download_kit(out)
    elif name in ("deep_dynamics","comma2k19","extreme_road"): download_github_repo(name,out)
    elif name=="bicycle_tire": download_zenodo(name,out,yaml_only=not full)
    elif name=="mendeley_friction": download_mendeley(out)
    marker.write_text("ok\n"); return out
