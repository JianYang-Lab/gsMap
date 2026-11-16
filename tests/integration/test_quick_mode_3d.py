import anndata as ad
import numpy as np
import pytest

from gsMap.cli import app

SECTION_KEY = "spatial_3d"
# Distance between the two synthetic sections, in the units of the XY coordinates.
SECTION_SPACING = 10.0


def _write_xyz_sections(paths, output_dir):
    """Copy 2D sections and add registered XYZ coordinates (`spatial3D` requires N×3)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for index, path in enumerate(paths):
        adata = ad.read_h5ad(path)
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        z = np.full((xy.shape[0], 1), index * SECTION_SPACING)
        adata.obsm[SECTION_KEY] = np.hstack([xy, z])
        section = output_dir / f"section{index}_{path.name}"
        adata.write_h5ad(section)
        sections.append(section)
    return sections


@pytest.mark.real_data
def test_quick_mode_3d_pipeline(
    cli_runner,
    work_dir,
    h5ad_file_3d_slice1,
    h5ad_file_3d_slice2,
    homolog_file,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """Test the full 3D quick-mode pipeline end-to-end on two registered XYZ sections."""
    project_name = "test_3d"
    sections = _write_xyz_sections(
        [h5ad_file_3d_slice1, h5ad_file_3d_slice2], work_dir / f"{project_name}_inputs"
    )
    result = cli_runner.invoke(
        app,
        [
            "quick-mode",
            "--workdir",
            str(work_dir),
            "--project-name",
            project_name,
            "--dataset-type",
            "spatial3D",
            "--h5ad-path",
            str(sections[0]),
            "--h5ad-path",
            str(sections[1]),
            "--homolog-file",
            str(homolog_file),
            "--annotation",
            "annotation",
            "--data-layer",
            "count",
            "--spatial-key",
            SECTION_KEY,
            "--n-adjacent-slices",
            "1",
            "--cross-slice-marker-score-strategy",
            "hierarchical_pool",
            "--w-ld-dir",
            str(w_ld_dir),
            "--snp-gene-weight-adata-path",
            str(snp_gene_weight_path),
            "--trait-name",
            "IQ",
            "--sumstats-file",
            str(sumstats_file),
            "--no-gpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    project_dir = work_dir / project_name
    assert (project_dir / "find_latent_representations").is_dir()
    assert (project_dir / "latent_to_gene").is_dir()
    assert (project_dir / "spatial_ldsc").is_dir()
    assert (project_dir / "cauchy_combination").is_dir()
