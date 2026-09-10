from __future__ import annotations

import json
import zipfile
from pathlib import Path

import safegrip.download as dl


class _Response:
    def __init__(self, status_code: int, payload=None):
        self.status_code = int(status_code)
        self.ok = 200 <= self.status_code < 300
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_lira_metadata_route_tolerates_primary_api_403(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("headers", {})))
        if "/versions/1" in url:
            return _Response(403)
        return _Response(
            200,
            {
                "files": [
                    {
                        "name": "task_7505_speed.txt",
                        "size": 7,
                        "download_url": "https://ndownloader.figshare.com/files/123",
                    }
                ]
            },
        )

    monkeypatch.setattr(dl.requests, "get", fake_get)
    files, errors = dl._lira_metadata_files()

    assert files[0]["name"] == "task_7505_speed.txt"
    assert any("HTTP 403" in x for x in errors)
    assert len(calls) == 2
    # Hosted notebook requests must look like a normal browser, because Figshare
    # can reject generic scripting identities at the WAF layer.
    assert calls[0][1]["User-Agent"].startswith("Mozilla/5.0")
    assert calls[0][1]["Referer"].startswith("https://data.dtu.dk/")


def test_lira_403_falls_back_to_dtu_bulk_ndownloader(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(dl.requests, "get", lambda *args, **kwargs: _Response(403))
    downloaded = []

    def fake_download(url, dest, **kwargs):
        downloaded.append(url)
        assert kwargs["headers"]["User-Agent"].startswith("Mozilla/5.0")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("task_7505_speed.txt", "timestamp,value\n0,36\n")
            z.writestr("m3_custom_fric_hh.csv", "Lat,Lon,mu\n55,12,0.6\n")
        return dest

    monkeypatch.setattr(dl, "_download", fake_download)
    dl.download_lira(tmp_path)

    assert downloaded == ["https://data.dtu.dk/ndownloader/articles/23096600/versions/1"]
    assert (tmp_path / "task_7505_speed.txt").exists()
    assert (tmp_path / "m3_custom_fric_hh.csv").exists()
    source = json.loads((tmp_path / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["download_route"] == "public_bulk_ndownloader"
    assert source["bulk_download_url"] == downloaded[0]
    assert any("HTTP 403" in x for x in source["metadata_warnings"])


def test_lira_bulk_rejects_html_and_tries_next_mirror(monkeypatch, tmp_path: Path):
    calls = []

    def fake_download(url, dest, **kwargs):
        calls.append(url)
        dest = Path(dest)
        if len(calls) == 1:
            dest.write_text("<html>blocked</html>", encoding="utf-8")
        else:
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("ok.txt", "ok")
        return dest

    monkeypatch.setattr(dl, "_download", fake_download)
    errors = []
    archive, used = dl._download_lira_bulk(tmp_path, errors)

    assert zipfile.is_zipfile(archive)
    assert used == "https://ndownloader.figshare.com/articles/23096600/versions/1"
    assert calls[:2] == [
        "https://data.dtu.dk/ndownloader/articles/23096600/versions/1",
        "https://ndownloader.figshare.com/articles/23096600/versions/1",
    ]
    assert any("not a ZIP archive" in x for x in errors)


def test_lira_per_file_403_also_falls_back_to_bulk(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        dl,
        "_lira_metadata_files",
        lambda: ([{"name": "x.txt", "size": 1, "download_url": "https://blocked.example/x"}], []),
    )
    calls = []

    def fake_download(url, dest, **kwargs):
        calls.append(url)
        if "blocked.example" in url:
            response = type("R", (), {"status_code": 403})()
            raise dl.requests.HTTPError("403", response=response)
        dest = Path(dest)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("task_7505_speed.txt", "timestamp,value\n0,36\n")
        return dest

    monkeypatch.setattr(dl, "_download", fake_download)
    dl.download_lira(tmp_path)

    assert calls[0] == "https://blocked.example/x"
    assert calls[1] == "https://data.dtu.dk/ndownloader/articles/23096600/versions/1"
    source = json.loads((tmp_path / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["download_route"] == "public_bulk_ndownloader"
    assert any("per-file download -> HTTP 403" in x for x in source["metadata_warnings"])
