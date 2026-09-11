from __future__ import annotations
import hashlib, json, os, re, shutil, tarfile, zipfile
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from .utils import ensure_dir
from .datasets import DATASET_REGISTRY

UA = {"User-Agent": "SafeGripOpen/1.0 academic-research"}


class DownloadIntegrityError(RuntimeError):
    """Raised when a remote endpoint returns an unusable or truncated payload."""


SOURCE = {
    "lira": {
        "kind":"figshare",
        "article_id":23096600,
        "version":1,
        "landing":"https://data.dtu.dk/articles/dataset/Data_subset_for_road_condition_modelling_-_platoon_friction_test/23096600/1",
    },
    "kuleuven": {"kind":"dataverse", "server":"https://rdr.kuleuven.be", "pid":"doi:10.48804/PHMF9D"},
    "kit": {"kind":"radar", "landing":"https://radar.kit.edu/radar/en/dataset/p0rr2jc5wmf0drf8"},
    "deep_dynamics": {"kind":"github", "repo":"linklab-uva/deep-dynamics"},
    "comma2k19": {"kind":"github", "repo":"commaai/comma2k19"},
    "extreme_road": {"kind":"github", "repo":"sean-shiyuez/Extreme-Road-Image-Dataset"},
    "bicycle_tire": {"kind":"zenodo", "record":7866646},
    "mendeley_friction": {"kind":"mendeley", "slug":"trrcrgzg75", "version":1,
                           "landing":"https://data.mendeley.com/datasets/trrcrgzg75/1"},
}


def _download(
    url: str,
    dest: Path,
    *,
    session=None,
    headers=None,
    timeout=120,
    expected_size: int | None = None,
    allow_empty: bool = False,
) -> Path:
    """Stream ``url`` to ``dest`` and reject silent empty/truncated responses.

    Figshare/DTU can occasionally return HTTP 200 with a zero-byte body from a
    download mirror.  ``requests.raise_for_status`` cannot detect that failure,
    so payload integrity is checked before the temporary file is promoted.
    When repository metadata provides an exact byte count, callers should pass
    it through ``expected_size``.
    """
    session = session or requests.Session()
    h = dict(UA)
    if headers:
        h.update(headers)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume = part.stat().st_size if part.exists() else 0
    if resume:
        h["Range"] = f"bytes={resume}-"

    with session.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=h) as r:
        if r.status_code == 416 and part.exists():
            size = part.stat().st_size
            valid = (allow_empty or size > 0) and (expected_size is None or size == int(expected_size))
            if valid:
                part.replace(dest)
                return dest
            part.unlink(missing_ok=True)
            raise DownloadIntegrityError(
                f"Server rejected resume for {url}, and the partial payload is invalid "
                f"({size} bytes; expected {expected_size!r})."
            )

        r.raise_for_status()
        mode = "ab" if resume and r.status_code == 206 else "wb"
        if mode == "wb":
            resume = 0
        try:
            content_length = int(r.headers.get("content-length", 0) or 0)
        except (TypeError, ValueError):
            content_length = 0
        total = content_length + resume
        with open(part, mode) as f, tqdm(
            total=total or None,
            initial=resume,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as progress:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

    size = part.stat().st_size if part.exists() else 0
    if not allow_empty and size == 0:
        part.unlink(missing_ok=True)
        raise DownloadIntegrityError(f"Downloaded an empty payload from {url}.")
    if expected_size is not None and size != int(expected_size):
        part.unlink(missing_ok=True)
        raise DownloadIntegrityError(
            f"Downloaded payload size mismatch from {url}: got {size} bytes, "
            f"expected {int(expected_size)} bytes."
        )

    part.replace(dest)
    return dest


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


def _figshare_browser_headers(*, accept_json: bool = False) -> dict:
    """Headers that work with Figshare/DTU anti-bot frontends on hosted notebooks."""
    h = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*" if accept_json else "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": SOURCE["lira"]["landing"],
    }
    return h


def _lira_metadata_files() -> tuple[list[dict], list[str]]:
    """Return LiRA file metadata, tolerating Figshare API/WAF failures."""
    cfg = SOURCE["lira"]
    article_id = int(cfg["article_id"])
    version = int(cfg.get("version", 1))
    urls = [
        f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}",
        f"https://api.figshare.com/v2/articles/{article_id}",
    ]
    errors: list[str] = []
    for api in urls:
        try:
            r = requests.get(api, headers=_figshare_browser_headers(accept_json=True), timeout=60)
            if not r.ok:
                errors.append(f"{api} -> HTTP {r.status_code}")
                continue
            payload = r.json()
            files = payload.get("files", []) if isinstance(payload, dict) else []
            files = [x for x in files if isinstance(x, dict) and x.get("name") and x.get("download_url")]
            if files:
                return files, errors
            errors.append(f"{api} -> no downloadable files in metadata")
        except (requests.RequestException, ValueError) as e:
            errors.append(f"{api} -> {type(e).__name__}: {e}")
    return [], errors


def _lira_file_download_urls(item: dict) -> list[str]:
    """Return de-duplicated per-file mirrors for one Figshare metadata item."""
    urls: list[str] = []
    reported = str(item.get("download_url") or "").strip()
    if reported:
        urls.append(reported)
    file_id = item.get("id")
    if file_id not in (None, ""):
        fid = str(file_id)
        urls.extend(
            [
                f"https://data.dtu.dk/ndownloader/files/{fid}",
                f"https://ndownloader.figshare.com/files/{fid}",
                f"https://figshare.com/ndownloader/files/{fid}",
            ]
        )
    # Preserve order while avoiding duplicate mirrors.
    return list(dict.fromkeys(urls))


def _download_lira_file(item: dict, dest: Path, errors: list[str]) -> str:
    """Download one LiRA file, validating metadata size and trying safe mirrors."""
    expected = item.get("size")
    expected_size = int(expected) if expected not in (None, "") else None
    urls = _lira_file_download_urls(item)
    if not urls:
        raise DownloadIntegrityError(f"No download URL available for LiRA file {dest.name!r}.")

    attempts: list[str] = []
    for url in urls:
        # A .part file belongs to one concrete mirror.  Never append bytes from
        # a different host to it when falling back.
        part = dest.with_suffix(dest.suffix + ".part")
        dest.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        try:
            _download(
                url,
                dest,
                headers=_figshare_browser_headers(),
                timeout=300,
                expected_size=expected_size,
            )
            size = dest.stat().st_size if dest.exists() else 0
            if size <= 0:
                raise DownloadIntegrityError(f"LiRA mirror returned an empty file: {url}")
            if expected_size is not None and size != expected_size:
                raise DownloadIntegrityError(
                    f"LiRA mirror returned {size} bytes for {dest.name}, expected {expected_size}."
                )
            return url
        except (requests.RequestException, DownloadIntegrityError, OSError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}" if status is not None else f"{type(exc).__name__}: {exc}"
            attempt = f"{url} -> {detail}"
            attempts.append(attempt)
            errors.append(f"{dest.name}: {attempt}")

    raise DownloadIntegrityError(
        f"All per-file mirrors failed for LiRA file {dest.name!r}. "
        + " | ".join(attempts)
    )


def _download_lira_bulk(out: Path, errors: list[str]) -> tuple[Path, str]:
    """Fallback to the public bulk-downloader used by the DTU Figshare UI."""
    cfg = SOURCE["lira"]
    article_id = int(cfg["article_id"])
    version = int(cfg.get("version", 1))
    urls = [
        f"https://data.dtu.dk/ndownloader/articles/{article_id}/versions/{version}",
        f"https://ndownloader.figshare.com/articles/{article_id}/versions/{version}",
        f"https://figshare.com/ndownloader/articles/{article_id}/versions/{version}",
    ]
    dest = out / f"lira_platoon_friction_test_{article_id}_v{version}.zip"
    for url in urls:
        try:
            # Do not resume a payload obtained from another mirror; a stale partial
            # file can otherwise corrupt the archive when the fallback host changes.
            part = dest.with_suffix(dest.suffix + ".part")
            if dest.exists():
                dest.unlink()
            if part.exists():
                part.unlink()
            _download(url, dest, headers=_figshare_browser_headers(), timeout=300)
            if not zipfile.is_zipfile(dest):
                size = dest.stat().st_size if dest.exists() else 0
                errors.append(f"{url} -> response is not a ZIP archive ({size} bytes)")
                if dest.exists():
                    dest.unlink()
                continue
            return dest, url
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            suffix = f"HTTP {status}" if status is not None else f"{type(e).__name__}: {e}"
            errors.append(f"{url} -> {suffix}")
        except (DownloadIntegrityError, OSError, zipfile.BadZipFile) as e:
            errors.append(f"{url} -> {type(e).__name__}: {e}")
    raise RuntimeError(
        "LiRA automatic download failed through both Figshare metadata and public bulk-download routes. "
        "The dataset is published at DOI 10.11583/DTU.23096600.v1. "
        "Upstream attempts:\n  - " + "\n  - ".join(errors)
    )


def download_lira(out: Path) -> None:
    """Download the LiRA platoon-friction subset with a DTU bulk-download fallback."""
    out.mkdir(parents=True, exist_ok=True)
    files, errors = _lira_metadata_files()
    if files:
        try:
            used_urls: dict[str, str] = {}
            for item in files:
                dest = out / str(item["name"])
                expected = item.get("size")
                expected_size = int(expected) if expected not in (None, "") else None
                size_ok = (
                    dest.exists()
                    and dest.stat().st_size > 0
                    and (expected_size is None or dest.stat().st_size == expected_size)
                )
                if not size_ok:
                    used_urls[dest.name] = _download_lira_file(item, dest, errors)
                _extract(dest, out)
            _write_source(
                "lira",
                out,
                {
                    "download_route": "figshare_api_file_metadata",
                    "download_urls": used_urls,
                    "metadata_warnings": errors,
                },
            )
            return
        except (requests.RequestException, DownloadIntegrityError, OSError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            text = str(e)
            if status is not None:
                suffix = f"HTTP {status}"
            elif "HTTP 403" in text:
                suffix = "HTTP 403"
            elif "HTTP 401" in text:
                suffix = "HTTP 401"
            else:
                suffix = f"{type(e).__name__}: {e}"
            errors.append(f"Figshare per-file download -> {suffix}")

    print("[download] LiRA: Figshare API route unavailable; trying DTU public bulk downloader.")
    archive, url = _download_lira_bulk(out, errors)
    _extract(archive, out)
    _write_source(
        "lira",
        out,
        {
            "download_route": "public_bulk_ndownloader",
            "bulk_download_url": url,
            "metadata_warnings": errors,
        },
    )


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
