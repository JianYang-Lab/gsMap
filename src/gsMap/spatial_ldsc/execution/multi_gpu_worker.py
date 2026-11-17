"""Private subprocess entry point; CUDA visibility is set by the coordinator."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, request["cpus"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The sampler thread must inherit this worker's assigned CPU affinity.
    from gsMap.resource_usage import track_resource_usage

    return track_resource_usage(run_worker)(request)


def run_worker(request):
    from gsMap.utils.jax_setup import use_float32_jax

    use_float32_jax()
    logging.info(
        "Starting LDSC worker %s on CUDA token %s; CPU affinity %s",
        request["rank"],
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        request["cpus"],
    )
    # Do not move these imports above affinity/device selection.
    import anndata as ad
    import jax
    import numpy as np
    import torch
    from scipy import sparse as sp

    from gsMap.spatial_ldsc.data.chunks import SpatialLDChunkSource
    from gsMap.spatial_ldsc.data.loading import FeatherAnnData
    from gsMap.spatial_ldsc.execution.shared import run_shared_chunks

    if jax.default_backend() != "gpu" or torch.cuda.device_count() != 1:
        raise RuntimeError("Each LDSC worker requires exactly one allocated CUDA GPU")
    staging, rank = Path(request["staging"]), request["rank"]
    meta = json.loads((staging / "inputs.json").read_text())
    left, right = meta["ranges"][rank]
    prepared = []
    for i, record in enumerate(meta["traits"]):
        with np.load(staging / f"trait-{i}.npz", allow_pickle=False) as arrays:
            prepared.append({**record, **dict(arrays)})
    config = SimpleNamespace(
        project_name="shard",
        sample_filter=None,
        cell_indices_range=None,
        n_blocks=meta["n_blocks"],
        ldsc_trait_batch_size=meta["trait_batch"],
        ldsc_regression_backend=meta.get("regression_backend", "auto"),
        ldsc_read_workers=meta.get("read_workers", 4),
    )
    source_info = meta["source"]

    def record_progress(record):
        if "progress_path" not in request:
            return
        path = Path(request["progress_path"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        os.replace(temporary, path)

    marker, direct, matrix, sparse = None, None, None, None
    try:
        if source_info["format"] == "direct":
            direct = ad.read_h5ad(source_info["path"], backed="r")
        elif source_info["format"] == "memmap":
            matrix = np.memmap(
                source_info["path"],
                mode="r",
                dtype=source_info["dtype"],
                shape=tuple(source_info["shape"]),
                offset=source_info["offset"],
                order=source_info["order"],
            )
            marker = SimpleNamespace(X=matrix)
        elif source_info["format"] == "h5ad":
            marker = ad.read_h5ad(source_info["path"], backed="r")
        elif source_info["format"] == "feather":
            marker = FeatherAnnData(
                Path(source_info["path"]), index_col="HUMAN_GENE_SYM", transpose=True
            )
        else:
            raise ValueError("Unsupported shared input format")
        if marker is not None:
            sparse = sp.load_npz(staging / "weight.npz")
        source = SpatialLDChunkSource(
            mode="direct_ldscore" if direct is not None else "weight_matrix",
            chunk_starts=list(range(0, right - left, meta["batch"])),
            chunk_size=meta["batch"],
            cell_end=right - left,
            spot_indices=np.load(staging / "rows.npy", mmap_mode="r")[left:right],
            spot_names_filtered=np.load(staging / "names.npy", mmap_mode="r")[left:right],
            ldscore_adata=direct,
            ldscore_snp_positions=np.load(staging / "union.npy"),
            marker_score_adata=marker,
            marker_score_gene_indices=np.load(staging / "genes.npy")
            if marker is not None
            else None,
        )
        paths = run_shared_chunks(
            source,
            config,
            prepared,
            sparse,
            staging / f"rank-{rank}",
            right - left,
            emit_header=rank == 0,
            progress_callback=record_progress,
        )
        device = torch.cuda.get_device_properties(0)
        (staging / f"worker-{rank}.json").write_text(
            json.dumps(
                dict(
                    range=[left, right],
                    rows=right - left,
                    paths=[str(path) for path in paths],
                    gpu=device.name,
                    gpu_uuid=str(getattr(device, "uuid", "unknown")),
                    cpus=request["cpus"],
                )
            )
        )
    finally:
        for adata in (marker, direct):
            if getattr(adata, "isbacked", False):
                adata.file.close()
        if matrix is not None:
            matrix._mmap.close()


if __name__ == "__main__":
    main()
