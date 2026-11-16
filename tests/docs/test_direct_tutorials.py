"""The displayed CLI and Python recipes must specify equivalent analysis settings."""

import json
import re
import shlex
import shutil
from types import SimpleNamespace
from urllib.parse import urljoin

import pytest
import yaml
from docs.tools.generate_downloads import render, render_catalog
from typer.testing import CliRunner

import gsMap.config
import gsMap.find_latent
import gsMap.pipeline
import gsMap.pipeline.quick_mode
from gsMap.cli import app
from gsMap.config.spatial_ldsc_config import GWASSumstatsConfig
from gsMap.config.utils import process_h5ad_inputs
from tests.docs.helpers import cli_settings
from tests.helpers.audit_commands import blocks
from tests.helpers.test_download_tutorial import download_blocks
from tests.helpers.validate_examples import TUTORIALS, tutorial_code
from tests.paths import REPO_ROOT as REPO


@pytest.mark.parametrize("example", TUTORIALS)
def test_direct_cli_python_parameters_match(example, monkeypatch):
    _, shell = tutorial_code(REPO, example, "cli")
    _, python = tutorial_code(REPO, example, "python")
    recorded = []

    def capture(command):
        def build(**kwargs):
            recorded.append((command, kwargs))
            return SimpleNamespace(**kwargs)

        return build

    monkeypatch.setattr(gsMap.config, "QuickModeConfig", capture("quick-mode"))
    monkeypatch.setattr(gsMap.config, "FindLatentRepresentationsConfig", capture("find-latent"))
    monkeypatch.setattr(gsMap.pipeline, "run_quick_mode", lambda config: None)
    monkeypatch.setattr(gsMap.find_latent, "run_find_latent_representation", lambda config: None)
    exec(compile(python, TUTORIALS[example], "exec"), {})
    assert recorded == cli_settings(shell)
    if example.startswith("e11-"):
        assert len(recorded) == 1
        command, settings = recorded[0]
        assert command == "quick-mode"
        assert settings["annotation"] == "spatial_domain"
        assert settings["n_adjacent_slices"] == 8
        for default in (
            "start_step",
            "cross_slice_selection_mode",
            "spatial_neighbors",
            "homogeneous_neighbors",
            "n_neighbors",
            "latent_representation_cell",
            "latent_representation_niche",
        ):
            assert default not in settings
    elif example == "human-gastrulation-cs8":
        for key in ("n_adjacent_slices", "graph_mode", "high_quality_cell_qc", "random_seed"):
            assert key not in recorded[0][1]


def test_gastrulation_packaged_inputs_survive_moving_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "original workspace"
    dataset = workspace / "example_data/human-gastrulation-cs8"
    data = dataset / "ST"
    data.mkdir(parents=True)
    for index in range(1, 63):
        (data / f"cs8_section_{index:03d}.h5ad").touch()
    templates = REPO / "tests/fixtures/tutorial_inputs/human-gastrulation-cs8"
    for name in ("sample_paths.txt", "gwas.yaml"):
        shutil.copyfile(templates / name, dataset / name)
    gwas = workspace / "example_data/example-traits"
    gwas.mkdir()
    for trait in ("IQ", "Height", "MCHC"):
        (gwas / f"{trait}.sumstats.gz").touch()
    relocated = tmp_path / "relocated workspace"
    workspace.rename(relocated)
    monkeypatch.chdir(tmp_path)
    dataset = relocated / "example_data/human-gastrulation-cs8"
    config = SimpleNamespace(sample_h5ad_dict=None, h5ad_list_file=dataset / "sample_paths.txt")
    samples = process_h5ad_inputs(config, {"h5ad_list_file": ("h5ad_list_file", "file")})
    assert [int(path.stem.rsplit("_", 1)[1]) for path in samples.values()] == list(range(1, 63))
    assert all(path.is_file() for path in samples.values())
    traits = GWASSumstatsConfig(sumstats_config_file=dataset / "gwas.yaml")
    assert traits.trait_name_list == ["IQ", "Height", "MCHC"]
    assert all(
        path.parent == relocated / "example_data/example-traits"
        for path in traits.sumstats_config_dict.values()
    )


def test_cs8_tutorial_names_and_geometry_summary_match_public_bundle():
    name = "human-gastrulation-cs8"
    paths = (REPO / "tests/fixtures/tutorial_inputs" / name / "sample_paths.txt").read_text()
    assert paths.splitlines() == [f"ST/cs8_section_{i:03d}.h5ad" for i in range(1, 63)]
    tutorial = (REPO / "docs/source/3d_tutorial/human_gastrulation.md").read_text()
    assert "human_gastrulation_slice_" not in tutorial
    catalog = json.loads((REPO / "docs/resource_catalog.json").read_text())
    entry = next(item for item in catalog["resources"] if item["id"] == name)
    for description in (tutorial, entry["description"], entry["display"]["description"]):
        assert "3D reconstruction" in description
        assert "aligned" in description
        for detail in ("MAPS", "inferred Z", "not author-verified", "20 µm"):
            assert detail not in description
    assert entry["revision"] == "r1"
    assert entry["path"] == f"datasets/st/3d/{name}/r1/bundle.tar.gz"
    assert all(item["id"] != "human-gastrulation" for item in catalog["resources"])


@pytest.mark.parametrize("example", ["human-gastrulation-cs8", "e11-ordinal"])
def test_3d_tutorials_use_dataset_local_inputs_without_setup_code(example):
    for interface in ("cli", "python"):
        setup, _ = tutorial_code(REPO, example, interface)
        assert setup == []
    text = (REPO / "docs/source" / TUTORIALS[example]).read_text()
    assert "Path order matters" in text
    assert re.search(r"along\s+the\s+z axis", text)
    assert "Prepare the section list" not in text
    assert "write_text(" not in text
    _, code = tutorial_code(REPO, example, "cli")
    dataset = "mouse-e11" if example.startswith("e11-") else "human-gastrulation-cs8"
    for _, settings in cli_settings(code):
        assert settings["h5ad_list_file"] == f"example_data/{dataset}/sample_paths.txt"
        if "sumstats_config_file" in settings:
            assert settings["sumstats_config_file"] == f"example_data/{dataset}/gwas.yaml"
    packaged = REPO / "tests/fixtures/tutorial_inputs" / dataset / "gwas.yaml"
    traits = ["IQ", "AF", "LDL"] if dataset == "mouse-e11" else ["IQ", "Height", "MCHC"]
    assert yaml.safe_load(packaged.read_text()) == {
        trait: f"../example-traits/{trait}.sumstats.gz" for trait in traits
    }


def test_public_documentation_has_no_wrapper_dependency():
    for path in (REPO / "docs/source").rglob("*"):
        if path.suffix in (".md", ".rst"):
            assert not re.search(
                r"examples/|\b(?:from|import)\s+examples\b|\brun_tutorial\b", path.read_text()
            ), path


@pytest.mark.parametrize("page", sorted(set(TUTORIALS.values())))
def test_each_tutorial_downloads_only_its_own_dataset(page):
    path = REPO / "docs/source" / page
    downloads = download_blocks(path)
    assert len(downloads) == 1
    assert "https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/" in downloads[0]
    assert "reference/ld_weights/" not in downloads[0]
    assert "reference/homologs/" not in downloads[0]
    assert "/gwas/" not in downloads[0]
    assert "-C example_data" in downloads[0]
    assert "&&" in downloads[0]
    assert "sha256sum" not in downloads[0]
    assert "download_dir=" not in downloads[0]
    assert len(downloads[0].splitlines()) <= 4
    assert "../downloads" in path.read_text()
    assert not re.search(
        r"resources/(?:mouse-visium|mouse-e16-mosta|human-gastrulation-cs8|e11-annotations|example-traits)\b",
        path.read_text(),
    )


def test_shared_downloads_are_separate_from_datasets():
    downloads = download_blocks(REPO / "docs/source/downloads.md")
    assert len(downloads) == 3
    assert "reference/ld_weights/" in downloads[0]
    assert "reference/homologs/" in downloads[1]
    assert "-C resources" in downloads[0]
    assert "-C resources" in downloads[1]
    assert "/gwas/" in downloads[2]
    assert "-C example_data" in downloads[2]
    assert not any("datasets/st/" in code for code in downloads)
    for code in downloads:
        assert len(code.splitlines()) <= 4
        assert "&&" in code
        assert "https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/" in code
        for boilerplate in ("for ", "GSMAP_DATA_URL", "download_dir=", "sha256sum"):
            assert boilerplate not in code


def test_downloads_is_single_generated_setup_page():
    catalog = json.loads((REPO / "docs/resource_catalog.json").read_text())
    generated = render(catalog)
    assert generated == (REPO / "docs/source/downloads.md").read_text()
    assert not (REPO / "docs/source/tutorial_setup.md").exists()
    assert "<!-- RESOURCE_CATALOG -->" not in generated
    assert generated.count("# Downloads\n") == 1
    assert "## Start a tutorial" in generated
    assert "## Browse all resources" in generated
    assert "reference/ld_weights/1000g-hc-grch38-v50-eur" in generated


def test_ci_and_downloader_use_the_documented_public_catalog():
    from tests.tools.fetch_resources import DEFAULT_CATALOG

    catalog = json.loads((REPO / "docs/resource_catalog.json").read_text())
    expected = f"{catalog['base_url'].rstrip('/')}/catalog.json"
    assert DEFAULT_CATALOG == expected
    workflow = yaml.safe_load((REPO / ".github/workflows/test_linux.yml").read_text())
    assert workflow["jobs"]["test"]["env"]["TEST_DATA_CATALOG"] == expected


def test_downloads_has_two_sections_and_expandable_verification():
    text = (REPO / "docs/source/downloads.md").read_text()
    assert re.findall(r"^## (.+)$", text, re.M) == ["Start a tutorial", "Browse all resources"]
    details = re.search(
        r"<details>\s*<summary>Verify your download</summary>(.*?)</details>", text, re.S
    )
    assert details
    assert "sha256sum" in details[1]
    assert "sha256sum" not in text[: details.start()]
    assert "How paths are used" not in text
    assert "Reuse resources for your own data or a later release" not in text


def test_catalog_keeps_every_archive_and_groups_ld_by_version():
    catalog = json.loads((REPO / "docs/resource_catalog.json").read_text())
    text = render_catalog(catalog)
    for entry in catalog["resources"]:
        assert text.count(f"[Archive]({entry['url']})") == 1
        assert text.count(f"[SHA-256]({entry['url']}.sha256)") == 1
    for heading in ("GENCODE v50 · GRCh38", "GENCODE v46 · GRCh37"):
        assert f"#### {heading}" in text
    for ancestry in ("AFR", "AMR", "EAS", "EUR", "SAS"):
        assert f"| {ancestry} |" in text
    assert "| EUR | 503 |" in text
    assert "| EUR | 489 |" in text


def test_downloads_identifies_phase3_low_coverage_resources():
    catalog = json.loads((REPO / "docs/resource_catalog.json").read_text())
    text = render_catalog(catalog)
    assert "V46 uses the GRCh37 1000 Genomes Phase 3 low-coverage EUR reference panel" in text
    row = next(
        line for line in text.splitlines() if line.startswith("| LD-matrix tutorial inputs")
    )
    assert "GRCh37 1000 Genomes Phase 3 low-coverage EUR PLINK panel" in row
    assert "489 individuals; chromosomes 1–22" in row
    assert "GENCODE v46 gene BED" in row
    assert "small historical" not in text.lower()
    assert "historical EUR resource" not in text


def test_dataset_catalog_has_counts_sources_and_distinct_previews():
    catalog = json.loads((REPO / "docs/resource_catalog.json").read_text())
    text = render(catalog)
    assert "| Dataset | Spots / cells | Publication / source | Report preview | Download |" in text
    for removed in (
        "Availability",
        "UKB-derived matrices are not publicly downloadable.",
        "prepared sections",
        "Staged; not published",
        "await final publication",
    ):
        assert removed not in text
    counts = {
        "mouse-e16-mosta": 121767,
        "mouse-visium": 7628,
        "human-gastrulation-cs8": 38562,
        "mouse-e11": 6993667,
    }
    for entry in catalog["resources"]:
        assert entry["url"] == f"{catalog['base_url']}/{entry['path']}"
        if entry["id"] not in counts:
            continue
        display = entry["display"]
        assert display["n_spots"] == counts[entry["id"]]
        assert f"{display['n_spots']:,}" in text
        page = REPO / "docs/source" / (entry["tutorial"].removesuffix(".html") + ".md")
        tutorial = page.read_text()
        for source in display["sources"]:
            assert source["label"] != "Publication"
            assert source["url"] in text
            assert source["url"] in tutorial
        for preview in display["report_previews"]:
            assert preview["url"] in tutorial
            assert preview["url"] in text


@pytest.mark.parametrize(
    ("page", "author"),
    [
        ("2d_tutorial/mouse_embryo.md", "Chen et al. · 2022 · Cell"),
        ("3d_tutorial/human_gastrulation.md", "Xiao et al. · 2024 · Cell"),
        ("3d_tutorial/mouse_e11.md", "Qiu et al. · 2024 · Cell"),
    ],
)
def test_tutorial_citation_labels_are_informative(page, author):
    text = (REPO / "docs/source" / page).read_text()
    assert f"[{author}](https://doi.org/" in text.split("## Files needed", 1)[0]
    assert "[Publication]" not in text


@pytest.mark.parametrize("page", sorted(set(TUTORIALS.values())))
def test_required_file_tree_precedes_tutorial_commands(page):
    text = (REPO / "docs/source" / page).read_text()
    tree = re.search(r"## Files needed for this tutorial.*?```text\n(.*?)\n```", text, re.S)
    assert tree is not None
    assert tree.end() < text.index("```bash")
    for required in (
        "resources/",
        "example_data/",
        "snp_gene_weights.h5ad",
        "w_ld/",
        "IQ.sumstats.gz",
    ):
        assert required in tree[1]
    if "human_gastrulation" in page:
        assert "homologs/" not in tree[1]
        assert "sample_paths.txt" in tree[1]
        assert "gwas.yaml" in tree[1]
    elif "mouse_e11" in page:
        for required in (
            "mouse-e11/",
            "AF.sumstats.gz",
            "LDL.sumstats.gz",
            "gwas.yaml",
        ):
            assert required in tree[1]


@pytest.mark.parametrize("page", sorted(set(TUTORIALS.values())))
def test_tutorials_follow_three_step_structure(page):
    text = (REPO / "docs/source" / page).read_text()
    headings = re.findall(r"^## (.+)$", text, re.M)
    assert headings == [
        "Files needed for this tutorial",
        "1. Download",
        "2. Run gsMap",
        "3. View results",
    ]
    for clutter in ("sha256sum", "prepare_e11.py", "preparation_metadata.json", "Resume safely"):
        assert clutter not in text


def test_e11_download_is_ready_to_run():
    text = (REPO / "docs/source/3d_tutorial/mouse_e11.md").read_text()
    assert "datasets/st/3d/mouse-e11/r1/bundle.tar.gz" in text
    assert "no preparation or configuration-file editing is needed" in text
    assert "sample_paths.txt" in text
    assert "gwas.yaml" in text
    assert not (REPO / "docs/source/3d_tutorial/mouse_e11_preparation.md").exists()


def test_report_preview_links_work_independently_of_docs_location():
    expected = {
        "2d_tutorial/10x.md": {"visium-coronal", "visium-sagittal", "visium-kidney"},
        "2d_tutorial/mouse_embryo.md": {"mouse-e16-mosta"},
        "3d_tutorial/human_gastrulation.md": {"human-gastrulation-cs8"},
        "3d_tutorial/mouse_e11.md": {"e11-ordinal"},
    }
    for page, reports in expected.items():
        text = (REPO / "docs/source" / page).read_text()
        links = re.findall(r"\[[^\]]*(?:Report preview|View report)[^\]]*\]\(([^)]+)\)", text)
        assert links
        assert all(link.startswith("https://yanglab.westlake.edu.cn/") for link in links)
        targets = {
            f"https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/{report}/r1/index.html"
            for report in reports
        }
        for base in (
            "https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/docs/r1/",
            "https://gsmap.readthedocs.io/en/latest/",
            "http://localhost:8000/",
            "file:///tmp/docs-build/",
        ):
            page_url = f"{base}{page.removesuffix('.md')}.html"
            assert {urljoin(page_url, link) for link in links} == targets
        assert "gsMap Report preview" in text.split("## Files needed", 1)[0]
        assert re.search(r"gsMap Report preview", text.split("## 3. View results", 1)[1])
        assert "staged" not in text.lower()
    e11 = (REPO / "docs/source/3d_tutorial/mouse_e11.md").read_text()
    assert "gsMap3D/gsmap3d-report/index.html" not in e11
    assert "6,609,635 cells" in e11
    assert "results from this tutorial" in e11
    assert "/reports/e11-ordinal/r1/index.html" in e11
    downloads = (REPO / "docs/source/downloads.md").read_text()
    assert "/reports/e11-ordinal/r1/index.html" in downloads
    assert "gsMap3D/gsmap3d-report/index.html" not in downloads
    output_reference = (REPO / "docs/source/output_files.md").read_text()
    assert "Links marked staged" not in output_reference


@pytest.mark.parametrize("interface", ["cli", "python"])
def test_e11_quick_mode_initializes_documented_settings(tmp_path, monkeypatch, interface):
    """Exercise real CLI/config initialization; no biological pipeline is run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    dataset = tmp_path / "example_data/mouse-e11"
    dataset.mkdir(parents=True)
    import anndata as ad
    import numpy as np

    (dataset / "ST").mkdir()
    inputs = []
    for section in range(17):
        relative = f"ST/Mouse_E11.5_embryo_S{section:02d}.h5ad"
        data = ad.AnnData(np.ones((3, 2)))
        data.obsm["z_correction"] = np.array(
            [[0, 0, section], [1, 0, section], [0, 1, section]], dtype=float
        )
        data.write_h5ad(dataset / relative)
        inputs.append(relative)
    (dataset / "sample_paths.txt").write_text("\n".join(inputs) + "\n")
    shutil.copyfile(
        REPO / "tests/fixtures/tutorial_inputs/mouse-e11/gwas.yaml",
        dataset / "gwas.yaml",
    )
    gwas = tmp_path / "example_data/example-traits"
    gwas.mkdir()
    for trait in ("IQ", "AF", "LDL"):
        (gwas / f"{trait}.sumstats.gz").touch()
    resource = tmp_path / "resources/1000g-hc-grch38-v50-eur"
    (resource / "w_ld").mkdir(parents=True)
    (resource / "snp_gene_weights.h5ad").touch()
    homologs = tmp_path / "resources/homologs"
    homologs.mkdir()
    (homologs / "mouse_human_homologs.txt").touch()
    recorded = []
    monkeypatch.setattr(gsMap.pipeline, "run_quick_mode", recorded.append)
    monkeypatch.setattr(gsMap.pipeline.quick_mode, "run_quick_mode", recorded.append)
    _, code = tutorial_code(REPO, "e11-ordinal", interface)
    if interface == "cli":
        invocation = next(
            line for line in code.replace("\\\n", " ").splitlines() if line.startswith("gsmap ")
        )
        result = CliRunner().invoke(app, shlex.split(invocation)[1:])
        assert result.exit_code == 0, result.output
    else:
        exec(compile(code, TUTORIALS["e11-ordinal"], "exec"), {})
    assert len(recorded) == 1
    config = recorded[0]
    assert isinstance(config, gsMap.config.QuickModeConfig)
    assert config.start_step == "find_latent"
    assert config.stop_step is None
    assert config.annotation == "spatial_domain"
    assert config.annotation_list == ["spatial_domain", "ManualAnnotation"]
    assert config.data_layer == "X"
    assert config.dataset_type == "spatial3D"
    assert config.cross_slice_selection_mode == "ordinal"
    assert config.n_adjacent_slices == 8
    defaults = gsMap.config.QuickModeConfig.__dataclass_fields__
    assert config.spatial_neighbors == defaults["spatial_neighbors"].default
    assert config.adjacent_slice_spatial_neighbors == 200
    assert config.homogeneous_neighbors == defaults["homogeneous_neighbors"].default
    assert config.graph_mode == "symmetric_knn"
    assert config.n_neighbors == 10
    assert config.K == 3
    assert config.random_seed == 2024
    assert config.latent_representation_cell == "emb_cell"
    assert config.latent_representation_niche == "emb_niche"
    assert config.high_quality_cell_qc is False
    assert config.high_quality_neighbor_filter is False
    assert config.cross_slice_marker_score_strategy == "hierarchical_pool"
    assert config.homogeneous_selection_mode == "soft_priority"
    assert config.cell_embedding_similarity_threshold == 0.0
    assert config.spatial_domain_similarity_threshold == 0.6
    assert config.trait_name_list == ["IQ", "AF", "LDL"]
    assert all(path.is_file() for path in config.sumstats_config_dict.values())


def test_matching_mode_is_documented_only_in_reference():
    source = REPO / "docs/source"
    reference = source / "cli_reference.md"
    assert "--cross-slice-selection-mode" in reference.read_text()
    for path in source.rglob("*"):
        if path.suffix in (".md", ".rst") and path != reference:
            assert not re.search(r"cross[-_]slice[-_]selection[-_]mode", path.read_text()), path
    tutorial = (source / "3d_tutorial/mouse_e11.md").read_text()
    for obsolete in ("e11-physical", "physical_window", "find-latent"):
        assert obsolete not in tutorial
    assert "e11-physical" not in TUTORIALS


def test_output_reference_keeps_logs_brief_and_platform_neutral():
    text = (REPO / "docs/source/output_files.md").read_text()
    assert "legacy" not in text.lower()
    logs = text.split("## Execution Logs\n", 1)[1].split("\n## Step 1:", 1)[0]
    assert len(logs.split()) < 230
    for internal in ("slurm", "cgroup", "rebuild_log_index", "symbolic links"):
        assert internal not in logs.lower()
    for useful in ("logs/latest/", "03_spatial_ldsc/{trait}.log", "run_manifest.json"):
        assert useful in logs
    assert "load_marker_scores_memmap_format" in text
    assert "observation mapping" in text
    assert "np.memmap(" not in text


def test_public_docs_do_not_depend_on_review_branch_language():
    paths = [
        REPO / "README.md",
        *(REPO / "docs/source").rglob("*.md"),
        *(REPO / "docs/source").rglob("*.rst"),
    ]
    for path in paths:
        text = path.read_text()
        for internal in (
            "revision/graph-3d",
            "origin/main",
            "development branch",
            "tutorial_setup",
            "migration.md",
            ":doc:`migration`",
        ):
            assert internal not in text, path


def test_customization_complete_commands_use_real_flags_and_required_paths():
    commands = []
    for _, language, code in blocks(REPO / "docs/source/advanced_usage.md"):
        if language == "bash":
            commands.extend(cli_settings(code))
    commands = [(command, values) for command, values in commands if "help" not in values]
    assert len(commands) == 3
    for _, values in commands:
        for key in (
            "workdir",
            "project_name",
            "w_ld_dir",
            "snp_gene_weight_adata_path",
            "sumstats_file",
            "trait_name",
        ):
            assert key in values
    embedding = next(
        values for _, values in commands if values["project_name"] == "custom-embedding"
    )
    assert embedding["start_step"] == "latent2gene"
    assert embedding["data_layer"] == "counts"
    gss = next(values for _, values in commands if values["project_name"] == "custom-gss")
    assert gss["marker_score_format"] == "h5ad"


def test_customization_flag_excerpts_are_valid_quick_mode_options():
    text = (REPO / "docs/source/advanced_usage.md").read_text()
    excerpts = re.findall(r"```text\n(.*?)\n```", text, re.S)
    for excerpt in excerpts:
        if not excerpt.startswith("--"):
            continue
        settings = cli_settings("gsmap quick-mode " + " ".join(excerpt.splitlines()))
        assert len(settings) == 1
