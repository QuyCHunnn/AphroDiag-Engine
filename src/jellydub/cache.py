"""Filesystem cache helpers. OWNED BY THE MANAGER — shared by all modules."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def file_digest(path: Path) -> str:
    """Hash content, including changes that preserve file size and mtime."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_file_matches(path: Path, signature: str) -> bool:
    if not path.is_file() or not path.stat().st_size:
        return False
    try:
        saved = json.loads(path.with_name(path.name + ".cache.json").read_text())
        return saved == {"signature": signature, "sha256": file_digest(path)}
    except (OSError, ValueError):
        return False


def record_cached_file(path: Path, signature: str) -> None:
    atomic_write_text(path.with_name(path.name + ".cache.json"),
                      json.dumps({"signature": signature, "sha256": file_digest(path)}))


@contextmanager
def cache_lock(path: Path):
    """Serialize writers of a shared cue across processes and threads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            import time
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def cache_key(*parts: object) -> str:
    """Stable sha256 hex digest (first 16 chars) of the JSON-encoded parts."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def tmp_path_for(path: Path) -> Path:
    """Temp sibling path to write into, then os.replace() onto `path` (keeps the suffix for ffmpeg)."""
    return path.with_name(f"{path.stem}.part{path.suffix}")
