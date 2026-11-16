"""Maintainer utility: download and verify catalog bundles (not a tutorial runner)."""

import argparse
import hashlib
import http.client
import json
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

DEFAULT_CATALOG = "https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/catalog.json"
# The resource server rejects the default ``Python-urllib`` agent with HTTP 403.
USER_AGENT = "gsMap-resource-fetch"


# Long transfers from the resource server to CI runners can stall; resume them.
DOWNLOAD_ATTEMPTS = 6
TRANSIENT_ERRORS = (OSError, http.client.HTTPException)  # timeouts, resets, URLError


def open_url(url, timeout, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def download(url, output, attempts=DOWNLOAD_ATTEMPTS, timeout=120, pause=10):
    """Download ``url`` to ``output``, resuming with HTTP Range after a stall."""
    for attempt in range(1, attempts + 1):
        offset = output.stat().st_size if output.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        try:
            with open_url(url, timeout, headers) as source:
                resumed = offset and getattr(source, "status", 200) == 206
                with output.open("ab" if resumed else "wb") as handle:
                    shutil.copyfileobj(source, handle, length=8 << 20)
            return
        except urllib.error.HTTPError:
            raise
        except TRANSIENT_ERRORS as error:
            if attempt == attempts:
                raise
            size = output.stat().st_size if output.exists() else 0
            print(
                f"Download interrupted at {size} bytes ({error}); retry {attempt}/{attempts - 1}"
            )
            time.sleep(pause * attempt)


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def verify_extraction(output):
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"Extraction must be a regular directory: {output}")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Extraction contains a symlink: {path}")
    expected_files = set()
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        path = output / name
        if (
            name in expected_files
            or not re.fullmatch(r"[a-f0-9]{64}", expected)
            or not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or not path.resolve().is_relative_to(output.resolve())
            or path.is_symlink()
            or digest(path) != expected
        ):
            raise ValueError(f"Extracted checksum failed: {path}")
        expected_files.add(name)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual - {"SHA256SUMS", ".archive_sha256"} != expected_files:
        raise ValueError(f"Extracted file inventory differs from SHA256SUMS: {output}")


def fetch(catalog_location, destination, identifiers):
    local = Path(catalog_location)
    if local.is_file():
        catalog = json.loads(local.read_text())
    else:
        with open_url(catalog_location, timeout=60) as response:
            catalog = json.load(response)
    entries = catalog["resources"]
    catalog = {entry["id"]: entry for entry in entries}
    if len(catalog) != len(entries):
        raise ValueError("Duplicate resource IDs in catalog")
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / ".downloads"
    if cache.is_symlink():
        raise ValueError(f"Download cache must not be a symlink: {cache}")
    cache.mkdir(exist_ok=True)
    for identifier in identifiers:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
            raise ValueError(f"Invalid resource ID: {identifier}")
        if identifier not in catalog:
            raise ValueError(f"Unknown resource: {identifier}")
        entry = catalog[identifier]
        if "sha256" not in entry:
            raise ValueError(f"{identifier} is external-only; follow its preparation tutorial")
        if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError(f"Invalid archive SHA-256: {identifier}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", entry["revision"]):
            raise ValueError(f"Invalid resource revision: {entry['revision']}")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Invalid catalog path: {entry['path']}")
        archive_path = cache / f"{identifier}-{entry['revision']}.tar.gz"
        if archive_path.is_symlink():
            raise ValueError(f"Download archive must not be a symlink: {archive_path}")
        if not archive_path.exists() or digest(archive_path) != entry["sha256"]:
            with tempfile.TemporaryDirectory(dir=cache, prefix=f".{identifier}-") as temporary:
                partial = Path(temporary) / "bundle.tar.gz"
                if local.is_file():
                    shutil.copyfile(local.parent / entry["path"], partial)
                else:
                    download(entry["url"], partial)
                if digest(partial) != entry["sha256"]:
                    raise ValueError(f"Archive checksum failed: {identifier}")
                partial.replace(archive_path)
        output = destination / identifier
        if output.is_symlink():
            raise ValueError(f"Extraction must not be a symlink: {output}")
        if output.exists():
            marker = output / ".archive_sha256"
            if marker.is_file() and marker.read_text().strip() == entry["sha256"]:
                verify_extraction(output)
                print(f"Verified existing extraction: {identifier}")
                continue
            raise FileExistsError(f"Refusing to overwrite an existing extraction: {output}")
        with tarfile.open(archive_path, "r:gz") as archive:
            names = set()
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or len(name.parts) < 2
                    or name.parts[0] != identifier
                    or name.as_posix() != member.name
                    or member.name in names
                ):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                names.add(member.name)
            with tempfile.TemporaryDirectory(
                dir=destination, prefix=f".{identifier}-"
            ) as temporary:
                staging = Path(temporary)
                archive.extractall(staging, filter="data")
                extracted = staging / identifier
                verify_extraction(extracted)
                (extracted / ".archive_sha256").write_text(entry["sha256"] + "\n")
                if output.exists() or output.is_symlink():
                    raise FileExistsError(
                        f"Extraction appeared concurrently; retry verification: {output}"
                    )
                extracted.rename(output)
        print(f"Verified and extracted: {identifier}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", default=DEFAULT_CATALOG, help="Public URL or an offline catalog.json"
    )
    parser.add_argument("--destination", type=Path, default=Path("resources"))
    parser.add_argument("identifiers", nargs="+")
    args = parser.parse_args()
    fetch(args.catalog, args.destination, args.identifiers)
