"""Single-node multi-GPU LDSC coordination, without parent GPU initialization.

Workers own disjoint cell ranges, not traits. Only finalized private gzip
members are joined; no result is published before every worker has succeeded.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse as sp

from gsMap.spatial_ldsc.execution.progress import LDSCProgress

logger = logging.getLogger("gsMap.spatial_ldsc.multi_gpu")


def partition_cells(cells: int, batch: int, devices: int):
    """Balanced nonempty ranges on chunk boundaries, preserving the final tail."""
    if min(cells, batch, devices) < 1:
        raise ValueError("Cells, batch size and GPU count must be positive")
    chunks = (cells + batch - 1) // batch
    devices = min(devices, chunks)
    return [
        (chunks * rank // devices * batch, min(chunks * (rank + 1) // devices * batch, cells))
        for rank in range(devices)
    ]


def visible_gpu_tokens(requested: int):
    """Use only CUDA-visible allocation tokens (ordinals, GPU UUIDs or MIG IDs)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        # Discovery occurs in a short-lived child, never the coordinator's JAX
        # runtime. CUDA device ordering is the same as for the worker processes.
        query = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        tokens = [str(i) for i in range(int(query.stdout.strip()))]
    else:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if (
        requested < 1
        or len(tokens) < requested
        or "-1" in tokens
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError(
            f"Requested {requested} LDSC GPUs, but CUDA_VISIBLE_DEVICES provides {tokens}"
        )
    return tokens[:requested]


def worker_cpu_sets(count: int):
    """Split the current allocation, excluding memoryless NUMA nodes if possible."""
    allowed = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )

    def parse_list(value):
        result = set()
        for field in value.strip().split(","):
            ends = list(map(int, field.split("-")))
            result.update(range(ends[0], ends[-1] + 1))
        return result

    try:
        status = Path("/proc/self/status").read_text()
        memory = next(
            line.split(":", 1)[1]
            for line in status.splitlines()
            if line.startswith("Mems_allowed_list:")
        )
        permitted, eligible = parse_list(memory), set()
        for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
            total = next(
                line for line in (node / "meminfo").read_text().splitlines() if "MemTotal:" in line
            )
            if int(node.name[4:]) in permitted and int(total.split()[-2]) > 0:
                eligible.update(parse_list((node / "cpulist").read_text()))
        local = sorted(set(allowed) & eligible)
        if len(local) >= count:
            allowed = local
    except (OSError, ValueError, StopIteration):
        pass  # portable fallback is still confined to the caller's affinity
    if len(allowed) < count:
        raise ValueError(
            f"Need at least one allocated CPU per GPU ({count}); found {len(allowed)}"
        )
    return [
        allowed[len(allowed) * rank // count : len(allowed) * (rank + 1) // count]
        for rank in range(count)
    ]


def worker_environment(token: str, cpu_count: int):
    """Bind one visible device without changing the caller's CUDA ordinal order."""
    env = os.environ.copy()
    # In particular, preserve CUDA_DEVICE_ORDER (including an absent setting).
    # Changing it would reinterpret the allocation's numeric visibility tokens.
    env.update(
        CUDA_VISIBLE_DEVICES=token,
        JAX_PLATFORMS="cuda",
        JAX_ENABLE_X64="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS=str(cpu_count),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS=str(cpu_count),
    )
    source_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    return env


def merge_gzip_members(parts, target):
    """Atomically publish one-header CSV from ordered, finalized worker members."""
    target = Path(target)
    if not parts:
        raise ValueError("Cannot publish an empty shard list")
    if target.exists():
        raise FileExistsError(target)
    for rank, part in enumerate(parts):
        with gzip.open(part, "rb") as handle:
            first = handle.readline()
        header = b"spot,beta,se,z,p,neg_log10_p\n"
        if not first or (rank == 0 and first != header) or (rank > 0 and first == header):
            raise ValueError("Invalid or repeated shard header")
    descriptor, name = tempfile.mkstemp(
        prefix=".ldsc-merge-", suffix=".partial", dir=target.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            for part in parts:
                with Path(part).open("rb") as source:
                    shutil.copyfileobj(source, destination, 4 * 1024**2)
        # A hard link is an atomic no-clobber publication on this filesystem.
        # The temporary file and target are deliberately in the same directory.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


DEFAULT_WORKER_STALL_SECONDS = 3600


def worker_stall_seconds() -> float:
    """Seconds without any worker activity before a worker counts as hung (0 disables)."""
    value = os.environ.get("GSMAP_LDSC_WORKER_STALL_SECONDS", "")
    try:
        return max(0.0, float(value)) if value.strip() else float(DEFAULT_WORKER_STALL_SECONDS)
    except ValueError:
        logger.warning("Ignoring invalid GSMAP_LDSC_WORKER_STALL_SECONDS=%r", value)
        return float(DEFAULT_WORKER_STALL_SECONDS)


def _last_activity(paths, since: float) -> float:
    """Latest wall-clock modification time of a worker's heartbeat files."""
    latest = since
    for path in paths:
        try:
            latest = max(latest, Path(path).stat().st_mtime)
        except OSError:
            pass  # Not written yet.
    return latest


def wait_workers(processes, *, on_progress=None, heartbeat_paths=None, stall_seconds=None):
    """Fail fast, terminate siblings and reap every child even on interruption.

    ``heartbeat_paths[rank]`` lists files a healthy worker keeps updating (its
    progress record and log). A running worker whose files stay unchanged for
    ``stall_seconds`` is treated as hung, so the run fails instead of waiting forever.
    """
    started_wall = time.time()
    try:
        last_progress = time.monotonic()
        last_stall_check = time.monotonic()
        while True:
            codes = [process.poll() for process in processes]
            for rank, code in enumerate(codes):
                if code not in (None, 0):
                    raise RuntimeError(
                        f"LDSC GPU worker {rank} exited with status {code}; inspect its worker log"
                    )
            done = all(code == 0 for code in codes)
            if on_progress is not None and (done or time.monotonic() - last_progress >= 30):
                on_progress()
                last_progress = time.monotonic()
            if done:
                return
            if (
                heartbeat_paths is not None
                and stall_seconds
                and time.monotonic() - last_stall_check >= min(30.0, stall_seconds)
            ):
                last_stall_check = time.monotonic()
                now = time.time()
                for rank, code in enumerate(codes):
                    idle = now - _last_activity(heartbeat_paths[rank], started_wall)
                    if code is None and idle >= stall_seconds:
                        raise RuntimeError(
                            f"LDSC GPU worker {rank} made no progress for {idle / 60:.0f} min "
                            f"and appears hung; inspect {heartbeat_paths[rank][-1]}. Set "
                            "GSMAP_LDSC_WORKER_STALL_SECONDS to change this limit (0 disables it)."
                        )
            time.sleep(0.2)
    finally:
        stop_workers(processes)


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_multi_gpu(engine, union, sparse, prepared):
    """Stage shared host inputs once, launch isolated GPUs, merge ordered shards."""
    started = time.monotonic()
    tokens = visible_gpu_tokens(engine.config.ldsc_num_gpus)
    ranges = partition_cells(engine.total_cells_to_process, engine.chunk_size, len(tokens))
    tokens = tokens[: len(ranges)]
    cpus = worker_cpu_sets(len(tokens))
    root = engine.config.memmap_tmp_dir or engine.output_dir
    processes, handles = [], []
    logs = engine.output_dir / "multi_gpu_logs"
    logs.mkdir(parents=True, exist_ok=True)
    logdir = Path(tempfile.mkdtemp(prefix="run-", dir=logs))
    logger.info("Multi-GPU LDSC: %d workers; logs: %s", len(tokens), logdir)
    progress = LDSCProgress(
        engine.total_cells_to_process,
        len(prepared),
        len(engine.chunk_starts),
        started=started,
        label=f"Multi-GPU LDSC ({len(tokens)} GPUs)",
    )
    worker_progress = [{} for _ in ranges]

    def report_progress():
        for rank in range(len(ranges)):
            try:
                worker_progress[rank] = json.loads(
                    (logdir / f"worker-{rank}.progress.json").read_text()
                )
            except (OSError, ValueError):
                pass  # Keep the last complete snapshot during startup/updates.
        progress.update(
            sum(record.get("completed_cells", 0) for record in worker_progress),
            sum(record.get("completed_chunks", 0) for record in worker_progress),
            force=True,
        )

    with tempfile.TemporaryDirectory(prefix="gsmap-ldsc-", dir=root) as temporary:
        staging = Path(temporary)
        np.save(staging / "union.npy", union, allow_pickle=False)
        left, right = engine.chunk_starts[0], engine._cell_end
        np.save(staging / "rows.npy", engine.spot_indices[left:right], allow_pickle=False)
        np.save(
            staging / "names.npy",
            engine.spot_names_filtered[left:right].astype(str),
            allow_pickle=False,
        )
        if sparse is not None:
            sp.save_npz(staging / "weight.npz", sparse, compressed=False)
            np.save(staging / "genes.npy", engine.marker_score_gene_indices, allow_pickle=False)
        for i, data in enumerate(prepared):
            np.savez(
                staging / f"trait-{i}.npz",
                **{key: value for key, value in data.items() if key not in ("name", "filename")},
            )
        marker = None
        if engine.mode == "direct_ldscore":
            source = dict(format="direct", path=str(engine.config.ldscore_h5ad_path))
        else:
            marker = engine.marker_score_adata.X
            if isinstance(marker, np.memmap):
                source = dict(
                    format="memmap",
                    path=str(marker.filename),
                    dtype=str(marker.dtype),
                    shape=list(marker.shape),
                    offset=int(marker.offset),
                    order="C" if marker.flags.c_contiguous else "F",
                )
            elif engine.config.marker_score_format == "h5ad":
                source = dict(format="h5ad", path=str(engine.config.marker_score_h5ad_path))
            elif engine.config.marker_score_format == "feather":
                source = dict(format="feather", path=str(engine.config.marker_score_feather_path))
            else:
                raise ValueError("Multi-GPU LDSC requires file-backed marker scores")
        metadata = dict(
            source=source,
            traits=[dict(name=d["name"], filename=d["filename"]) for d in prepared],
            n_blocks=engine.config.n_blocks,
            batch=engine.chunk_size,
            trait_batch=engine.config.ldsc_trait_batch_size,
            regression_backend=getattr(engine.config, "ldsc_regression_backend", "auto"),
            read_workers=getattr(engine.config, "ldsc_read_workers", 4),
            ranges=ranges,
        )
        (staging / "inputs.json").write_text(json.dumps(metadata))
        try:
            for rank, (token, cpu_set) in enumerate(zip(tokens, cpus, strict=True)):
                request = dict(
                    staging=str(staging),
                    rank=rank,
                    cpus=cpu_set,
                    progress_path=str(logdir / f"worker-{rank}.progress.json"),
                )
                request_path = staging / f"request-{rank}.json"
                request_path.write_text(json.dumps(request))
                # Keep editable installs and source checkouts usable in children.
                env = worker_environment(token, len(cpu_set))
                handle = (logdir / f"worker-{rank}.log").open("w")
                handles.append(handle)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "gsMap.spatial_ldsc.execution.multi_gpu_worker",
                            str(request_path),
                        ],
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                )
            wait_workers(
                processes,
                on_progress=report_progress,
                heartbeat_paths=[
                    [logdir / f"worker-{rank}.progress.json", logdir / f"worker-{rank}.log"]
                    for rank in range(len(processes))
                ],
                stall_seconds=worker_stall_seconds(),
            )
            receipts = [
                json.loads((staging / f"worker-{rank}.json").read_text())
                for rank in range(len(ranges))
            ]
            uuids = [receipt["gpu_uuid"] for receipt in receipts]
            if len(set(uuids)) != len(uuids) or any(
                not value or value == "unknown" for value in uuids
            ):
                raise RuntimeError("Multi-GPU workers did not use distinct identifiable GPUs")
            for rank, (receipt, bounds) in enumerate(zip(receipts, ranges, strict=True)):
                if (
                    receipt["range"] != list(bounds)
                    or receipt["rows"] != bounds[1] - bounds[0]
                    or len(receipt["paths"]) != len(prepared)
                ):
                    raise RuntimeError(f"Incomplete output from GPU worker {rank}")
            # Publication is atomic per trait. A rerun skips any already-published
            # traits if a later merge fails; private partials are never reusable.
            for i, data in enumerate(prepared):
                parts = [Path(receipt["paths"][i]) for receipt in receipts]
                merge_gzip_members(parts, engine.output_dir / data["filename"])
            (logdir / "completed.json").write_text(
                json.dumps(
                    dict(workers=receipts, traits=[d["filename"] for d in prepared]), indent=2
                )
            )
            logger.info("Published %d traits from %d GPU cell shards", len(prepared), len(ranges))
            seconds = max(time.monotonic() - started, 1e-9)
            logger.info(
                "Multi-GPU LDSC completed %d cells × %d traits in %.2fs "
                "(%.1f cell-traits/s, including staging, worker setup and writing/merge)",
                engine.total_cells_to_process,
                len(prepared),
                seconds,
                engine.total_cells_to_process * len(prepared) / seconds,
            )
        finally:
            stop_workers(processes)
            for handle in handles:
                handle.close()
