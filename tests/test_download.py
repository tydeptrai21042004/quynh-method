from __future__ import annotations

import json
import zipfile

import pytest
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


def test_lira_zero_byte_per_file_payload_falls_back_to_bulk(monkeypatch, tmp_path: Path):
    """Regression: Kaggle received HTTP-successful zero-byte Figshare files."""
    monkeypatch.setattr(
        dl,
        "_lira_metadata_files",
        lambda: (
            [
                {
                    "id": 123,
                    "name": "task_7505_speed.txt",
                    "size": 21,
                    "download_url": "https://ndownloader.figshare.com/files/123",
                }
            ],
            [],
        ),
    )
    calls = []

    def fake_download(url, dest, **kwargs):
        calls.append(url)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "/files/123" in url:
            # Reproduce the Kaggle failure: HTTP path appears successful, but no
            # payload bytes are written.
            dest.write_bytes(b"")
        else:
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("task_7505_speed.txt", "timestamp,value\n0,36\n")
                z.writestr("m3_custom_fric_hh.csv", "Lat,Lon,mu\n55,12,0.6\n")
        return dest

    monkeypatch.setattr(dl, "_download", fake_download)
    dl.download_lira(tmp_path)

    source = json.loads((tmp_path / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["download_route"] == "public_bulk_ndownloader"
    assert (tmp_path / "task_7505_speed.txt").stat().st_size > 0
    assert any("empty file" in x.lower() for x in source["metadata_warnings"])
    assert any("/articles/23096600/versions/1" in x for x in calls)


def test_lira_per_file_size_mismatch_tries_next_mirror(monkeypatch, tmp_path: Path):
    item = {
        "id": 456,
        "name": "task_7505_speed.txt",
        "size": 7,
        "download_url": "https://primary.example/files/456",
    }
    calls = []

    def fake_download(url, dest, **kwargs):
        calls.append(url)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "primary.example" in url:
            dest.write_bytes(b"bad")
        else:
            dest.write_bytes(b"1234567")
        return dest

    monkeypatch.setattr(dl, "_download", fake_download)
    errors = []
    used = dl._download_lira_file(item, tmp_path / item["name"], errors)

    assert used == "https://data.dtu.dk/ndownloader/files/456"
    assert (tmp_path / item["name"]).read_bytes() == b"1234567"
    assert any("expected 7" in x for x in errors)  # failed primary mirror remains auditable
    assert calls[:2] == [
        "https://primary.example/files/456",
        "https://data.dtu.dk/ndownloader/files/456",
    ]


class _StreamingResponse:
    def __init__(self, chunks, status_code=200, headers=None):
        self._chunks = list(chunks)
        self.status_code = int(status_code)
        self.headers = dict(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            response = type("R", (), {"status_code": self.status_code})()
            raise dl.requests.HTTPError(str(self.status_code), response=response)

    def iter_content(self, _chunk_size):
        yield from self._chunks


class _StreamingSession:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def test_download_rejects_http_success_with_empty_body(tmp_path: Path):
    dest = tmp_path / "empty.txt"
    session = _StreamingSession(_StreamingResponse([], headers={"content-length": "0"}))

    with pytest.raises(dl.DownloadIntegrityError, match="empty payload"):
        dl._download("https://example.test/empty", dest, session=session)

    assert not dest.exists()
    assert not dest.with_suffix(".txt.part").exists()


def test_download_rejects_metadata_size_mismatch(tmp_path: Path):
    dest = tmp_path / "short.txt"
    session = _StreamingSession(_StreamingResponse([b"abc"], headers={"content-length": "3"}))

    with pytest.raises(dl.DownloadIntegrityError, match="size mismatch"):
        dl._download(
            "https://example.test/short",
            dest,
            session=session,
            expected_size=7,
        )

    assert not dest.exists()
    assert not dest.with_suffix(".txt.part").exists()
