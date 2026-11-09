import gzip
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.error

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assets  # noqa: E402

URL = "https://example.invalid/releases/download/runner-assets-x/runner-assets-x.tar.gz"
FILES = {"calc.tfpc": b"circuit", "graph-v7-powmod/bank/bank.json": b"{}"}


def archive_bytes(files, extra=None):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra:
            tar.addfile(extra)
    return raw.getvalue()


def pinned(tmp_path, files, archive=None, extra=None):
    """A manifest for `files`, a release file pinning the archive of `archive` (default: `files`) and that archive."""
    manifest = tmp_path / "assets.sha256"
    manifest.write_text("".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in sorted(files.items())))
    data = archive_bytes(files if archive is None else archive, extra)
    rel = tmp_path / "assets.release"
    rel.write_text(f"# test\nurl {URL}\nsha256 {hashlib.sha256(data).hexdigest()}\n")
    return manifest, rel, data


class Served:
    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        return False


def test_exact_files_pass_and_every_difference_is_named(tmp_path):
    manifest, _, _ = pinned(tmp_path, FILES)
    root = tmp_path / "runner"
    for name, data in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    assert assets.check(root, manifest) == []
    (root / "calc.tfpc").write_bytes(b"other")
    (root / "extra").write_bytes(b"")
    (root / "graph-v7-powmod/bank/bank.json").unlink()
    assert assets.check(root, manifest) == [
        "missing graph-v7-powmod/bank/bank.json", "unlisted extra", "sha256 differs calc.tfpc"]


def test_symlinks_and_escaping_names_are_refused(tmp_path):
    manifest, _, _ = pinned(tmp_path, {"calc.tfpc": b"circuit"})
    root = tmp_path / "runner"
    root.mkdir()
    (root / "calc.tfpc").write_bytes(b"circuit")
    (root / "link").symlink_to(root / "calc.tfpc")
    with pytest.raises(ValueError, match="not a regular file: link"):
        assets.check(root, manifest)
    manifest.write_text(f"{'0' * 64}  ../calc.tfpc\n")
    with pytest.raises(ValueError, match="bad manifest line"):
        assets.listed(manifest)


def test_download_checks_the_archive_and_every_file(tmp_path, monkeypatch):
    manifest, rel, data = pinned(tmp_path, FILES)
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout: Served(data))
    assets.fetch(tmp_path / "runner", rel, manifest)
    assert (tmp_path / "runner/calc.tfpc").read_bytes() == b"circuit"
    assert not list(tmp_path.glob(".runner-*"))  # the downloaded archive does not stay


def test_unpacked_modes_come_from_the_archive_not_the_umask(tmp_path):
    manifest, rel, _ = pinned(tmp_path, {"graph-v7-powmod/bank/bank.json": b"{}"})
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name in ("graph-v7-powmod", "graph-v7-powmod/bank"):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        info = tarfile.TarInfo("graph-v7-powmod/bank/bank.json")
        info.size, info.mode = 2, 0o644
        tar.addfile(info, io.BytesIO(b"{}"))
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(raw.getvalue())
    rel.write_text(f"url {URL}\nsha256 {hashlib.sha256(archive.read_bytes()).hexdigest()}\n")
    old = os.umask(0o002)
    try:
        assets.fetch(tmp_path / "runner", rel, manifest, archive=archive)
    finally:
        os.umask(old)
    modes = {p.relative_to(tmp_path / "runner").as_posix(): p.stat().st_mode & 0o777
             for p in (tmp_path / "runner").rglob("*")}
    assert modes == {"graph-v7-powmod": 0o755, "graph-v7-powmod/bank": 0o755, "graph-v7-powmod/bank/bank.json": 0o644}


def test_a_failed_download_names_the_file_and_the_url(tmp_path, monkeypatch):
    manifest, rel, _ = pinned(tmp_path, FILES)

    def refuse(url, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(assets.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(assets.time, "sleep", lambda s: None)
    with pytest.raises(assets.AssetError, match="cannot download runner-assets-x.tar.gz from " + URL):
        assets.fetch(tmp_path / "runner", rel, manifest)


def test_a_wrong_archive_hash_stops_before_unpacking(tmp_path, monkeypatch):
    manifest, rel, data = pinned(tmp_path, FILES)
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout: Served(data + b"x"))
    with pytest.raises(assets.AssetError, match=f"sha256 mismatch for runner-assets-x.tar.gz from {URL}"):
        assets.fetch(tmp_path / "runner", rel, manifest)
    assert not any((tmp_path / "runner").iterdir())


def test_a_missing_or_changed_file_is_named(tmp_path):
    manifest, rel, data = pinned(tmp_path, FILES, archive={"calc.tfpc": b"other"})
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(data)
    with pytest.raises(assets.AssetError) as error:
        assets.fetch(tmp_path / "runner", rel, manifest, archive=archive)
    message = str(error.value)
    assert "missing graph-v7-powmod/bank/bank.json" in message and "sha256 differs calc.tfpc" in message
    assert URL in message


def test_unsafe_archive_entries_are_refused(tmp_path):
    link = tarfile.TarInfo("calc.tfpc.link")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    manifest, rel, data = pinned(tmp_path, FILES, extra=link)
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(data)
    with pytest.raises(assets.AssetError, match="unsafe entry: calc.tfpc.link"):
        assets.fetch(tmp_path / "runner", rel, manifest, archive=archive)


def test_the_cli_reports_on_stderr_and_fails(tmp_path):
    result = subprocess.run([sys.executable, str(HERE / "assets.py"), "fetch", "--archive", str(tmp_path / "none.tgz"),
                             str(tmp_path / "runner")], capture_output=True, text=True)
    assert result.returncode == 1 and result.stderr.startswith("runner assets: ")


def test_the_pins_are_deepseek_v41_s_runner_asset_release():
    url, digest = assets.release()
    assert url == ("https://github.com/jayleaton/deepseek-v41-tensorfold-spark/releases/download/"
                   "runner-assets-20261107/runner-assets-20261107.tar.gz")
    assert digest == "b5c40d32bcab0fa896795d08aaa651d55d0574e500f3624e61db23058cdd800f"
    pins = assets.listed()
    assert pins["calc.tfpc"] == "3a421a2b8ba7c138c502ac205a8139ea05eeb2f7086715fe939b492ebbc18cf3"
    assert pins["graph-v7-powmod/bank/bank.json"] == "26862f0a758af4e01a1d8c69d501ad49ee1657162b36d057b8a3c312fcf12512"
    assert "graph-v7-powmod/bank/Qwenputer-MIT.txt" in pins
