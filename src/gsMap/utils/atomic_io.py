"""Publish output files atomically: readers see the old file or the complete new one.

Each write goes to a temporary file in the destination directory, is flushed to disk
(fsync) and then renamed over the target. A crash or kill mid-write leaves the previous
file (or none), never a truncated one that a later run would trust.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    with contextlib.suppress(OSError):  # not supported on every filesystem
        _fsync_path(directory)


@contextlib.contextmanager
def atomic_output(path: str | os.PathLike) -> Iterator[Path]:
    """Yield a temporary path to write; on success it replaces ``path`` atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial")
    try:
        yield temporary
        if temporary.is_file():
            _fsync_path(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.is_dir():
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
        else:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | os.PathLike, content: str, encoding: str = "utf-8") -> None:
    with atomic_output(path) as temporary:
        temporary.write_text(content, encoding=encoding)


def atomic_write_bytes(path: str | os.PathLike, content: bytes) -> None:
    with atomic_output(path) as temporary:
        temporary.write_bytes(content)
