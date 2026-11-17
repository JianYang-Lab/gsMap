"""Project-local logging for gsMap command and pipeline runs."""

from __future__ import annotations

import atexit
import copy
import json
import logging
import logging.handlers
import multiprocessing
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import psutil
from rich.console import Console
from rich.logging import RichHandler

from gsMap.resource_usage import ResourceMonitor, log_resource_usage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


LOGGER_NAME = "gsMap"
FILE_LOG_MAX_BYTES = 256 * 1024**2
FILE_LOG_BACKUP_COUNT = 4
RESOURCE_HEARTBEAT_SECONDS = 60.0

_STEP_FILENAMES = {
    "find_latent": "01_find_latent.log",
    "latent_to_gene": "02_latent_to_gene.log",
    "spatial_ldsc": "03_spatial_ldsc.log",
    "cauchy_combination": "04_cauchy_combination.log",
    "report": "05_report.log",
    "quick_mode": "quick_mode.log",
    "ldscore_weight_matrix": "ldscore_weight_matrix.log",
    "format_sumstats": "format_sumstats.log",
}
_STAGE_ALIASES = {
    "find_latent_representations": "find_latent",
    "latent2gene": "latent_to_gene",
    "cauchy": "cauchy_combination",
    "run_report": "report",
}
_REDACTED_KEYS = {
    "api_key",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}

P = ParamSpec("P")
R = TypeVar("R")


class _BelowWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


class _ConciseRichHandler(RichHandler):
    """Render exception summaries on the terminal and retain tracebacks for files."""

    def emit(self, record: logging.LogRecord) -> None:
        console_record = copy.copy(record)
        console_record.exc_info = None
        console_record.exc_text = None
        console_record.stack_info = None
        super().emit(console_record)


class _ConciseStreamHandler(logging.StreamHandler):
    """Write one unwrapped batch-log record while keeping tracebacks in files."""

    def emit(self, record: logging.LogRecord) -> None:
        console_record = copy.copy(record)
        console_record.exc_info = None
        console_record.exc_text = None
        console_record.stack_info = None
        super().emit(console_record)


class _ExternalWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(LOGGER_NAME)


class _IsoFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        del datefmt
        return (
            datetime.fromtimestamp(record.created, UTC)
            .astimezone()
            .isoformat(timespec="milliseconds")
        )


def _normalize_stage(stage: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage.strip()).strip("_.-").lower()
    return _STAGE_ALIASES.get(normalized, normalized or "gsmap")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.-") or "unknown"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_symlink(target: Path, link: Path) -> None:
    """Atomically point a human-facing index entry at an immutable run artifact."""

    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    relative_target = os.path.relpath(target, start=link.parent)
    try:
        temporary.symlink_to(relative_target)
        temporary.replace(link)
    finally:
        temporary.unlink(missing_ok=True)


def _latest_log_path(logs_dir: Path, stage: str, trait: str | None) -> Path:
    normalized = _normalize_stage(stage)
    filename = _STEP_FILENAMES.get(normalized, f"{_safe_name(normalized)}.log")
    if normalized == "spatial_ldsc" and trait is not None:
        return logs_dir / "latest" / filename.removesuffix(".log") / f"{_safe_name(trait)}.log"
    return logs_dir / "latest" / filename


def _infer_run_trait(config_or_project: Any, explicit_trait: str | None) -> str | None:
    if explicit_trait is not None:
        return str(explicit_trait)
    trait_name = getattr(config_or_project, "trait_name", None)
    if trait_name:
        return str(trait_name)
    trait_mapping = getattr(config_or_project, "sumstats_config_dict", None)
    if isinstance(trait_mapping, dict) and len(trait_mapping) == 1:
        return str(next(iter(trait_mapping)))
    return None


def rebuild_log_index(project_dir: str | Path) -> dict[str, str]:
    """Rebuild the human-facing ``logs/latest`` links from run manifests."""

    logs_dir = Path(project_dir).expanduser().resolve() / "logs"
    candidates: dict[tuple[str, str | None], tuple[str, Path, Path]] = {}
    latest_run: tuple[str, Path] | None = None
    if not logs_dir.is_dir():
        return {}

    for manifest_path in logs_dir.rglob("run_manifest.json"):
        if manifest_path.is_symlink():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        run_timestamp = str(
            manifest.get("finished_at_utc") or manifest.get("started_at_utc") or ""
        )
        if latest_run is None or run_timestamp > latest_run[0]:
            latest_run = (run_timestamp, manifest_path)

        for entry in manifest.get("stages", []):
            log_path = Path(entry.get("log_file", ""))
            if not log_path.is_absolute():
                log_path = manifest_path.parent / log_path
            if not log_path.is_file():
                continue
            stage = _normalize_stage(str(entry.get("stage", "gsmap")))
            trait_value = entry.get("trait")
            trait = _safe_name(str(trait_value)) if trait_value is not None else None
            timestamp = str(
                entry.get("finished_at_utc") or entry.get("started_at_utc") or run_timestamp
            )
            key = (stage, trait)
            current = candidates.get(key)
            if current is None or timestamp > current[0]:
                candidates[key] = (timestamp, log_path, manifest_path)

    rebuilt: dict[str, str] = {}
    for (stage, trait), (_, log_path, manifest_path) in candidates.items():
        latest_log = _latest_log_path(logs_dir, stage, trait)
        _atomic_symlink(log_path, latest_log)
        _atomic_symlink(manifest_path, latest_log.with_suffix(".manifest.json"))
        label = f"{stage}/{trait}" if trait is not None else stage
        rebuilt[label] = str(latest_log)

    if latest_run is not None:
        _atomic_symlink(latest_run[1], logs_dir / "latest" / "run_manifest.json")
    return rebuilt


def _redact_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in _REDACTED_KEYS or any(
        normalized.endswith(f"_{suffix}") for suffix in _REDACTED_KEYS
    )


def _jsonable(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _redact_key(key):
        return "<redacted>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def _redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if argument.startswith("--"):
            option, separator, _ = argument.partition("=")
            key = option.lstrip("-").replace("-", "_")
            if _redact_key(key):
                if separator:
                    redacted.append(f"{option}=<redacted>")
                else:
                    redacted.append(option)
                    redact_next = True
                continue
        redacted.append(argument)
    return redacted


def _package_version() -> str | None:
    from gsMap import __version__

    return None if __version__ == "unknown" else __version__


def resolve_project_dir(config_or_path: Any) -> Path:
    """Resolve the directory that owns the automatic ``logs`` folder."""

    if isinstance(config_or_path, str | Path):
        return Path(config_or_path).expanduser().resolve()
    project_dir = getattr(config_or_path, "project_dir", None)
    if project_dir is not None:
        return Path(project_dir).expanduser().resolve()
    workdir = getattr(config_or_path, "workdir", None)
    project_name = getattr(config_or_path, "project_name", None)
    if workdir is not None and project_name:
        return (Path(workdir) / str(project_name)).expanduser().resolve()
    output_dir = getattr(config_or_path, "output_dir", None)
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    output_file = getattr(config_or_path, "out", None)
    if output_file is not None:
        return Path(output_file).expanduser().resolve().parent
    return Path.cwd().resolve()


def resolve_project_dir_from_kwargs(kwargs: dict[str, Any]) -> Path:
    """Resolve a log owner before a CLI dataclass has been constructed."""

    if kwargs.get("workdir") is not None and kwargs.get("project_name"):
        return (Path(kwargs["workdir"]) / str(kwargs["project_name"])).expanduser().resolve()
    if kwargs.get("output_dir") is not None:
        return Path(kwargs["output_dir"]).expanduser().resolve()
    if kwargs.get("out") is not None:
        return Path(kwargs["out"]).expanduser().resolve().parent
    return Path.cwd().resolve()


def is_interactive_output(stream: Any | None = None) -> bool:
    """Return whether Rich live rendering is safe for the selected output stream."""
    target = sys.stdout if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


def _build_console_handler(
    stream: Any,
    *,
    level: int,
) -> logging.Handler:
    """Use Rich on a TTY and stable one-record-per-line output in batch jobs."""
    if is_interactive_output(stream):
        handler: logging.Handler = _ConciseRichHandler(
            console=Console(file=stream),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        handler.setFormatter(logging.Formatter("{name} - {message}", style="{"))
        handler._gsmap_console_mode = "rich"  # type: ignore[attr-defined]
    else:
        handler = _ConciseStreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler._gsmap_console_mode = "plain"  # type: ignore[attr-defined]
    handler.setLevel(level)
    handler._gsmap_managed_console = True  # type: ignore[attr-defined]
    return handler


def _install_console_handlers() -> tuple[list[logging.Handler], bool]:
    gsmap_logger = logging.getLogger(LOGGER_NAME)
    existing = [
        handler
        for handler in gsmap_logger.handlers
        if getattr(handler, "_gsmap_managed_console", False)
    ]
    if existing:
        return existing, False

    stdout_handler = _build_console_handler(sys.stdout, level=logging.INFO)
    stdout_handler.addFilter(_BelowWarningFilter())

    stderr_handler = _build_console_handler(sys.stderr, level=logging.WARNING)

    gsmap_logger.setLevel(logging.DEBUG)
    gsmap_logger.propagate = False
    gsmap_logger.addHandler(stdout_handler)
    gsmap_logger.addHandler(stderr_handler)
    return [stdout_handler, stderr_handler], True


def configure_console_logging() -> None:
    """Install concise console logging for commands that have no project directory."""

    _install_console_handlers()


def _remove_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def _git_context() -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    try:
        root = subprocess.check_output(
            ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", root, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"root": root, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "commit": None, "dirty": None}


def _slurm_context() -> dict[str, str]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_JOB_NAME",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURMD_NODENAME",
        "CUDA_VISIBLE_DEVICES",
    )
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _cgroup_memory() -> dict[str, int | None]:
    try:
        cgroup_line = next(
            line
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return {"current": None, "peak": None, "limit": None}

    relative = cgroup_line.split("::", maxsplit=1)[1].lstrip("/")
    current_path = Path("/sys/fs/cgroup") / relative
    cgroup_root = Path("/sys/fs/cgroup")
    while current_path == cgroup_root or cgroup_root in current_path.parents:
        limit = _read_int(current_path / "memory.max")
        if limit is not None:
            return {
                "current": _read_int(current_path / "memory.current"),
                "peak": _read_int(current_path / "memory.peak"),
                "limit": limit,
            }
        if current_path == cgroup_root:
            break
        current_path = current_path.parent
    return {"current": None, "peak": None, "limit": None}


def _resource_snapshot(process: psutil.Process) -> dict[str, Any]:
    memory = process.memory_info()
    try:
        io = process.io_counters()
        read_bytes = io.read_bytes
        write_bytes = io.write_bytes
    except (AttributeError, OSError, psutil.Error):
        read_bytes = None
        write_bytes = None
    cgroup = _cgroup_memory()
    return {
        "rss_bytes": memory.rss,
        "vms_bytes": memory.vms,
        "cpu_percent": process.cpu_percent(interval=None),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "cgroup_current_bytes": cgroup["current"],
        "cgroup_peak_bytes": cgroup["peak"],
        "cgroup_limit_bytes": cgroup["limit"],
    }


def _format_gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1024**3:.2f}"


def _log_resource_snapshot(label: str, process: psutil.Process) -> None:
    try:
        snapshot = _resource_snapshot(process)
        logging.getLogger(LOGGER_NAME).debug(
            f"{label} | "
            f"rss_gib={_format_gib(snapshot['rss_bytes'])} "
            f"vms_gib={_format_gib(snapshot['vms_bytes'])} "
            f"cpu_percent={snapshot['cpu_percent']:.1f} "
            f"read_gib={_format_gib(snapshot['read_bytes'])} "
            f"write_gib={_format_gib(snapshot['write_bytes'])} "
            f"cgroup_current_gib={_format_gib(snapshot['cgroup_current_bytes'])} "
            f"cgroup_peak_gib={_format_gib(snapshot['cgroup_peak_bytes'])} "
            f"cgroup_limit_gib={_format_gib(snapshot['cgroup_limit_bytes'])}"
        )
    except (OSError, psutil.Error, RuntimeError) as error:
        logging.getLogger(LOGGER_NAME).debug(f"{label} unavailable: {error}")


@dataclass
class _FileSink:
    path: Path
    log_queue: multiprocessing.Queue = field(init=False)
    queue_handler: logging.handlers.QueueHandler = field(init=False)
    external_queue_handler: logging.handlers.QueueHandler = field(init=False)
    file_handler: logging.handlers.RotatingFileHandler = field(init=False)
    listener: logging.handlers.QueueListener = field(init=False)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_queue = multiprocessing.Queue(-1)

        self.file_handler = logging.handlers.RotatingFileHandler(
            self.path,
            mode="a",
            maxBytes=FILE_LOG_MAX_BYTES,
            backupCount=FILE_LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(
            _IsoFormatter(
                "{asctime} {levelname:<8} [{process}/{threadName}] "
                "{name}:{funcName}:{lineno} | {message}",
                style="{",
            )
        )

        self.listener = logging.handlers.QueueListener(
            self.log_queue,
            self.file_handler,
            respect_handler_level=True,
        )
        self.listener.start()

        self.queue_handler = logging.handlers.QueueHandler(self.log_queue)
        self.queue_handler.setLevel(logging.DEBUG)
        self.queue_handler._gsmap_file_sink = True  # type: ignore[attr-defined]
        logging.getLogger(LOGGER_NAME).addHandler(self.queue_handler)

        self.external_queue_handler = logging.handlers.QueueHandler(self.log_queue)
        self.external_queue_handler.setLevel(logging.WARNING)
        self.external_queue_handler.addFilter(_ExternalWarningFilter())
        self.external_queue_handler._gsmap_external_file_sink = True  # type: ignore[attr-defined]
        logging.getLogger().addHandler(self.external_queue_handler)

    def close(self) -> None:
        logging.getLogger(LOGGER_NAME).removeHandler(self.queue_handler)
        logging.getLogger().removeHandler(self.external_queue_handler)
        self.listener.stop()
        self.queue_handler.close()
        self.external_queue_handler.close()
        self.file_handler.close()
        self.log_queue.close()
        self.log_queue.join_thread()


class RunLoggingSession:
    """Own one run directory and any nested stage-specific log files."""

    def __init__(self, project_dir: Path, command: str, run_trait: str | None = None) -> None:
        self.project_dir = project_dir
        self.command = _normalize_stage(command)
        self.run_trait = _safe_name(run_trait) if run_trait is not None else None
        self.owner_pid = os.getpid()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        slurm_id = os.environ.get("SLURM_JOB_ID")
        identity = [timestamp, self.command]
        if self.run_trait is not None:
            identity.append(self.run_trait)
        if slurm_id:
            identity.append(f"slurm{_safe_name(slurm_id)}")
        identity.extend(
            [
                _safe_name(socket.gethostname()),
                f"pid{self.owner_pid}",
                uuid.uuid4().hex[:8],
            ]
        )
        self.run_id = "__".join(identity)
        self.logs_dir = self.project_dir / "logs"
        self.run_dir = self.logs_dir / self.run_id
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.latest_path = self.logs_dir / "latest.json"
        self._lock = threading.RLock()
        self._active: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._failed_stage_seen = False
        self._logged_exceptions: list[BaseException] = []
        self._closed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._resource_monitor = ResourceMonitor()
        self._console_handlers: list[logging.Handler] = []
        self._owns_console_handlers = False
        self._owns_warning_capture = False
        self.manifest: dict[str, Any] = {}

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        gsmap_logger = logging.getLogger(LOGGER_NAME)
        self._previous_logger_state = (gsmap_logger.level, gsmap_logger.propagate)
        self._console_handlers, self._owns_console_handlers = _install_console_handlers()
        self._owns_warning_capture = getattr(logging, "_warnings_showwarning", None) is None
        logging.captureWarnings(True)
        self.manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": "running",
            "command": self.command,
            "run_trait": self.run_trait,
            "started_at_utc": _utc_now(),
            "project_dir": str(self.project_dir),
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "pid": self.owner_pid,
            "python": sys.version,
            "gsmap_version": _package_version(),
            "argv": _redact_argv(list(sys.argv)),
            "slurm": _slurm_context(),
            "git": _git_context(),
            "stages": [],
        }
        self._write_manifest()
        self._write_latest()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            name="gsmap-resource-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        self._resource_monitor.start()

    def _write_manifest(self) -> None:
        with self._lock:
            _atomic_json(self.manifest_path, self.manifest)

    def _write_latest(self) -> None:
        _atomic_json(
            self.latest_path,
            {
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "manifest_path": str(self.manifest_path),
                "status": self.manifest.get("status"),
                "updated_at_utc": _utc_now(),
            },
        )
        try:
            _atomic_symlink(self.manifest_path, self.logs_dir / "latest" / "run_manifest.json")
        except OSError as error:
            logging.getLogger(LOGGER_NAME).warning(
                f"Could not update latest run-manifest link: {error}"
            )

    def record_config(self, config: Any, *, stage: str | None = None) -> None:
        try:
            if hasattr(config, "to_dict_with_paths_as_strings"):
                resolved = config.to_dict_with_paths_as_strings()
            else:
                resolved = vars(config)
            serialized = _jsonable(resolved)
            normalized_stage = _normalize_stage(stage or self.command)
            self.manifest.setdefault("resolved_config", serialized)
            self.manifest.setdefault("stage_configs", {})[normalized_stage] = serialized
            self._write_manifest()
            logging.getLogger(LOGGER_NAME).debug(
                f"Resolved configuration for {normalized_stage}: "
                f"{json.dumps(serialized, sort_keys=True)}"
            )
        except Exception as error:  # noqa: BLE001 logging must not break a run
            logging.getLogger(LOGGER_NAME).warning(
                f"Could not record the resolved configuration: {error}"
            )

    def _stage_path(self, stage: str, trait: str | None) -> Path:
        if stage == "spatial_ldsc" and trait is not None:
            return self.run_dir / "03_spatial_ldsc" / f"{_safe_name(trait)}.log"
        filename = _STEP_FILENAMES.get(stage, f"{_safe_name(stage)}.log")
        return self.run_dir / filename

    @contextmanager
    def stage(self, stage: str, trait: str | None = None) -> Iterator[Path]:
        normalized = _normalize_stage(stage)
        safe_trait = _safe_name(trait) if trait is not None else None
        key = (normalized, safe_trait)
        with self._lock:
            existing = self._active.get(key)
            if existing is not None:
                existing["references"] += 1
                path = existing["path"]
                reused = True
            else:
                path = self._stage_path(normalized, safe_trait)
                sink = _FileSink(path)
                sink.start()
                entry = {
                    "id": len(self.manifest["stages"]) + 1,
                    "stage": normalized,
                    "trait": safe_trait,
                    "status": "running",
                    "started_at_utc": _utc_now(),
                    "log_file": str(path),
                }
                self.manifest["stages"].append(entry)
                self._active[key] = {
                    "references": 1,
                    "path": path,
                    "sink": sink,
                    "entry": entry,
                    "started": time.perf_counter(),
                    "resources": self._resource_monitor.begin(),
                    "error": None,
                }
                self._write_manifest()
                reused = False

        logger = logging.getLogger(LOGGER_NAME)
        if not reused:
            log_method = logger.debug if safe_trait is not None else logger.info
            stage_context = f"Stage {normalized}"
            if safe_trait is not None:
                stage_context = f"{stage_context} trait={safe_trait}"
            log_method(f"{stage_context} started | run_id={self.run_id}")

        try:
            yield path
        except BaseException as caught:
            with self._lock:
                # A nested use of the same stage can fail and be caught by its
                # caller while the outer stage continues. Preserve that failure
                # on the shared active entry so the final reference cannot
                # incorrectly overwrite it with "complete".
                if self._active[key]["error"] is None:
                    self._active[key]["error"] = caught
            if not any(caught is logged for logged in self._logged_exceptions):
                self._logged_exceptions.append(caught)
                logger.error(
                    f"Stage {normalized} failed: {caught}",
                    exc_info=(type(caught), caught, caught.__traceback__),
                )
            raise
        finally:
            with self._lock:
                active = self._active[key]
                active["references"] -= 1
                if active["references"] == 0:
                    stage_error = active["error"]
                    entry = active["entry"]
                    entry["finished_at_utc"] = _utc_now()
                    entry["duration_seconds"] = time.perf_counter() - active["started"]
                    entry["resource_usage"] = self._resource_monitor.finish(active["resources"])
                    if stage_error is None:
                        entry["status"] = "complete"
                    else:
                        entry["status"] = "failed"
                        entry["error_type"] = type(stage_error).__name__
                        entry["error_message"] = str(stage_error)
                        self._failed_stage_seen = True
                    self._write_manifest()
                    completion_target = f"Stage {normalized}"
                    resource_label = "Stage resource summary"
                    if safe_trait is not None:
                        completion_target = f"{completion_target} trait={safe_trait}"
                        resource_label = (
                            f"{resource_label} | stage={normalized} trait={safe_trait}"
                        )
                    _log_resource_snapshot(resource_label, psutil.Process(self.owner_pid))
                    log_resource_usage(entry["resource_usage"], label=completion_target)
                    if stage_error is None:
                        logger.info(
                            f"{completion_target} completed in {entry['duration_seconds']:.2f}s"
                        )
                    active["sink"].close()
                    latest_log = _latest_log_path(self.logs_dir, normalized, safe_trait)
                    entry["latest_log"] = str(latest_log)
                    try:
                        _atomic_symlink(path, latest_log)
                        _atomic_symlink(
                            self.manifest_path,
                            latest_log.with_suffix(".manifest.json"),
                        )
                    except OSError as index_error:
                        logger.warning(
                            f"Could not update latest log index for {normalized}: {index_error}"
                        )
                    self._write_manifest()
                    del self._active[key]

    def _heartbeat(self) -> None:
        process = psutil.Process(self.owner_pid)
        process.cpu_percent(interval=None)
        while not self._heartbeat_stop.wait(RESOURCE_HEARTBEAT_SECONDS):
            _log_resource_snapshot("Resource heartbeat", process)

    def close(self, *, failed: bool = False) -> None:
        if self._closed or os.getpid() != self.owner_pid:
            return
        self._closed = True
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        self._resource_monitor.close()

        with self._lock:
            for key, active in list(self._active.items()):
                active["entry"]["status"] = "interrupted"
                active["entry"]["finished_at_utc"] = _utc_now()
                active["entry"]["resource_usage"] = self._resource_monitor.finish(
                    active["resources"]
                )
                active["sink"].close()
                del self._active[key]

        statuses = [stage["status"] for stage in self.manifest["stages"]]
        if failed:
            status = "failed"
        elif self._failed_stage_seen or "failed" in statuses:
            status = "partial_failure" if "complete" in statuses else "failed"
        else:
            status = "complete"
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = _utc_now()
        started = datetime.fromisoformat(self.manifest["started_at_utc"])
        finished = datetime.fromisoformat(self.manifest["finished_at_utc"])
        self.manifest["duration_seconds"] = (finished - started).total_seconds()
        self._write_manifest()
        self._write_latest()

        if self._owns_warning_capture:
            logging.captureWarnings(False)
        if self._owns_console_handlers:
            gsmap_logger = logging.getLogger(LOGGER_NAME)
            _remove_handlers(gsmap_logger, self._console_handlers)
            # Leave the logger as found, so later messages (e.g. in a notebook) still
            # reach the caller's own logging configuration.
            level, propagate = getattr(self, "_previous_logger_state", (logging.NOTSET, True))
            gsmap_logger.setLevel(level)
            gsmap_logger.propagate = propagate


_STATE_LOCK = threading.RLock()
_CURRENT_SESSION: RunLoggingSession | None = None


@contextmanager
def stage_logging(
    config_or_project: Any,
    stage: str,
    *,
    trait: str | None = None,
) -> Iterator[RunLoggingSession]:
    """Open an idempotent project/run/stage logging context."""

    global _CURRENT_SESSION
    owns_session = False
    with _STATE_LOCK:
        if _CURRENT_SESSION is None or _CURRENT_SESSION.owner_pid != os.getpid():
            session = RunLoggingSession(
                resolve_project_dir(config_or_project),
                stage,
                run_trait=_infer_run_trait(config_or_project, trait),
            )
            session.start()
            _CURRENT_SESSION = session
            owns_session = True
        else:
            session = _CURRENT_SESSION

    failure: BaseException | None = None
    try:
        with session.stage(stage, trait=trait):
            yield session
    except BaseException as error:
        failure = error
        raise
    finally:
        if owns_session:
            session.close(failed=failure is not None)
            with _STATE_LOCK:
                if _CURRENT_SESSION is session:
                    _CURRENT_SESSION = None


def logged_stage(stage: str):
    """Decorate a public pipeline entry point with automatic stage logging."""

    def decorator(func: Any):
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            config = args[0] if args else kwargs.get("config")
            if config is None:
                return func(*args, **kwargs)
            with stage_logging(config, stage) as session:
                session.record_config(config, stage=stage)
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _shutdown_active_session() -> None:
    global _CURRENT_SESSION
    with _STATE_LOCK:
        session = _CURRENT_SESSION
        if session is not None and session.owner_pid == os.getpid():
            session.close(failed=True)
            _CURRENT_SESSION = None


def _reset_logging_for_tests() -> None:
    """Close owned handlers; intended only for isolated unit tests."""

    global _CURRENT_SESSION
    with _STATE_LOCK:
        if _CURRENT_SESSION is not None and _CURRENT_SESSION.owner_pid == os.getpid():
            _CURRENT_SESSION.close(failed=True)
        _CURRENT_SESSION = None
    gsmap_logger = logging.getLogger(LOGGER_NAME)
    handlers = [
        handler
        for handler in gsmap_logger.handlers
        if getattr(handler, "_gsmap_managed_console", False)
        or getattr(handler, "_gsmap_file_sink", False)
    ]
    _remove_handlers(gsmap_logger, handlers)
    root_logger = logging.getLogger()
    root_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_gsmap_external_file_sink", False)
    ]
    _remove_handlers(root_logger, root_handlers)


atexit.register(_shutdown_active_session)


__all__ = [
    "RunLoggingSession",
    "configure_console_logging",
    "is_interactive_output",
    "logged_stage",
    "rebuild_log_index",
    "resolve_project_dir",
    "resolve_project_dir_from_kwargs",
    "stage_logging",
]
