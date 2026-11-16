import os
from pathlib import Path

# Typer forces rich terminal styling when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set,
# which inserts ANSI codes into --help output that tests compare as plain text. This must
# be set before typer's rich_utils is first imported.
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")

import pytest
from typer.testing import CliRunner

from tests.logging_guard import preserved_gsmap_logging


@pytest.fixture(autouse=True)
def isolate_gsmap_logging():
    """Do not let CLI/logging tests change subsequent caplog capture behavior.

    Session-scoped fixtures run before this function-scoped fixture, so a fixture that
    invokes the CLI must use ``preserved_gsmap_logging`` itself.
    """
    with preserved_gsmap_logging():
        yield


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-real-data",
        action="store_true",
        default=False,
        help="Run tests that require real data",
    )
    parser.addoption(
        "--test-data",
        action="store",
        default=None,
        help="Path to gsmap3d_ci_test_data directory",
    )
    parser.addoption(
        "--work-dir",
        action="store",
        default=None,
        help="Path to working directory for test outputs (defaults to a temporary directory)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "real_data: mark test that requires real data (disabled by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip real_data tests by default unless --run-real-data is specified."""
    if not config.getoption("--run-real-data"):
        skip_real_data = pytest.mark.skip(reason="need --run-real-data option to run")
        for item in items:
            if "real_data" in item.keywords:
                item.add_marker(skip_real_data)


# ---------------------------------------------------------------------------
# Session-scoped fixtures: real data paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_data_dir(request):
    """Root test data directory."""
    test_dir = request.config.getoption("--test-data")
    if not test_dir:
        pytest.fail("Pass --test-data when enabling real-data tests", pytrace=False)
    test_dir = Path(test_dir)
    if not test_dir.exists():
        pytest.fail(f"Test data directory does not exist: {test_dir}", pytrace=False)
    return test_dir


@pytest.fixture(scope="session")
def example_data_dir(test_data_dir):
    """gsMap_example_data/ directory."""
    d = test_data_dir / "gsMap_example_data"
    if not d.exists():
        pytest.fail(f"Example data directory not found: {d}", pytrace=False)
    return d


@pytest.fixture(scope="session")
def ldscore_data_dir(test_data_dir):
    """gsMap_ldscore_weight_matrix_example_data/ directory."""
    d = test_data_dir / "gsMap_ldscore_weight_matrix_example_data"
    if not d.exists():
        pytest.fail(f"LDScore data directory not found: {d}", pytrace=False)
    return d


@pytest.fixture(scope="session")
def quick_mode_resource_dir(test_data_dir):
    """gsMap_quick_mode_resource/ directory."""
    d = test_data_dir / "gsMap_quick_mode_resource"
    if not d.exists():
        pytest.fail(f"Quick mode resource directory not found: {d}", pytrace=False)
    return d


# --- Individual data file fixtures ---


@pytest.fixture(scope="session")
def h5ad_file_2d(example_data_dir):
    """2D spatial h5ad file (E16.5_E1S1)."""
    p = example_data_dir / "ST" / "E16.5_E1S1.MOSTA_subsampled.h5ad"
    if not p.exists():
        pytest.fail(f"h5ad file not found: {p}", pytrace=False)
    return p


@pytest.fixture(scope="session")
def h5ad_file_3d_slice1(h5ad_file_2d):
    """3D pipeline slice 1 (same as the 2D file)."""
    return h5ad_file_2d


@pytest.fixture(scope="session")
def h5ad_file_3d_slice2(example_data_dir):
    """3D pipeline slice 2 (E16.5_E2S11)."""
    p = example_data_dir / "ST" / "E16.5_E2S11.MOSTA_subsampled.h5ad"
    if not p.exists():
        pytest.fail(f"h5ad file not found: {p}", pytrace=False)
    return p


@pytest.fixture(scope="session")
def sumstats_file(example_data_dir):
    """GWAS summary statistics file."""
    p = example_data_dir / "GWAS" / "filtered_IQ_NG_2018.sumstats.gz"
    if not p.exists():
        pytest.fail(f"Summary statistics file not found: {p}", pytrace=False)
    return p


@pytest.fixture(scope="session")
def homolog_file(test_data_dir):
    """Mouse-to-human homolog file."""
    p = test_data_dir / "gsMap_homologs" / "mouse_human_homologs.txt"
    if not p.exists():
        pytest.fail(f"Homolog file not found: {p}", pytrace=False)
    return p


@pytest.fixture(scope="session")
def w_ld_dir(quick_mode_resource_dir):
    """Weights directory (HM3 no HLA)."""
    d = quick_mode_resource_dir / "weights_hm3_no_hla"
    if not d.exists():
        pytest.fail(f"Weights directory not found: {d}", pytrace=False)
    return d


@pytest.fixture(scope="session")
def snp_gene_weight_path(quick_mode_resource_dir):
    """SNP-gene weight matrix h5ad."""
    p = quick_mode_resource_dir / "1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad"
    if not p.exists():
        pytest.fail(f"SNP-gene weight matrix not found: {p}", pytrace=False)
    return p


# --- LDScore weight matrix fixtures ---


@pytest.fixture(scope="session")
def ld_ref_panel_prefix(ldscore_data_dir):
    """PLINK binary file prefix for LD reference panel."""
    prefix = (ldscore_data_dir / "LD_Reference_Panel_subset" / "1000G.EUR.QC.subset").as_posix()
    # Verify at least chr22 exists
    if not Path(f"{prefix}.22.bed").exists():
        pytest.fail(f"LD reference panel not found at {prefix}", pytrace=False)
    return prefix


@pytest.fixture(scope="session")
def hapmap3_snps_dir(ldscore_data_dir):
    """HapMap3 SNPs directory."""
    d = ldscore_data_dir / "hapmap3_snps"
    if not d.exists():
        pytest.fail(f"HapMap3 SNPs directory not found: {d}", pytrace=False)
    return d


@pytest.fixture(scope="session")
def gencode_bed(ldscore_data_dir):
    """Gencode protein-coding BED file."""
    p = ldscore_data_dir / "genome_annotation" / "gencode" / "gencode_v46lift37_protein_coding.bed"
    if not p.exists():
        pytest.fail(f"Gencode BED file not found: {p}", pytrace=False)
    return p


# ---------------------------------------------------------------------------
# Work directory & CLI runner
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def work_dir(request, tmp_path_factory):
    """Working directory for test outputs."""
    custom_dir = request.config.getoption("--work-dir")
    if custom_dir:
        d = Path(custom_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path_factory.mktemp("gsmap_test")


@pytest.fixture(scope="session")
def cli_runner():
    """Typer CliRunner instance."""
    return CliRunner()
