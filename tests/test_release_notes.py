"""Release notes come from CHANGELOG.md and must exist before anything is published."""

import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_notes", ROOT / ".github/scripts/release_notes.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CHANGELOG = """# Changelog

## 2.0.0b2

- Second beta.

### Fixes

- A fix.

## 2.0.0b1 - 2026-11-20

- First beta.

## gsMap3D preview releases

### 0.1.4a2

- Preview.
"""


def test_section_stops_at_next_release_heading():
    assert MODULE.changelog_section(CHANGELOG, "2.0.0b2") == (
        "- Second beta.\n\n### Fixes\n\n- A fix."
    )
    assert MODULE.changelog_section(CHANGELOG, "2.0.0b1") == "- First beta."


@pytest.mark.parametrize("version", ["2.0.0", "2.0.0b", "0.1.4a2"])
def test_missing_section_is_rejected(version):
    with pytest.raises(ValueError, match="no '## "):
        MODULE.changelog_section(CHANGELOG, version)


def test_empty_section_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        MODULE.changelog_section("## 2.0.0\n\n## 1.0.0\n- old\n", "2.0.0")


def test_previous_tag_skips_unrelated_and_newer_tags():
    tags = ["v1.73.8", "v0.1.4a2", "v2.0.0b1", "v2.0.0rc1", "v2.0.0", "1.73.6", "vnext"]
    assert MODULE.previous_tag("v2.0.0b1", tags) == "v1.73.8"
    assert MODULE.previous_tag("v2.0.0", tags) == "v2.0.0rc1"
    assert MODULE.previous_tag("v0.1.0", tags) is None


def test_prerelease_notes_do_not_advertise_docker():
    notes = MODULE.release_notes(
        "v2.0.0b1", True, "JianYang-Lab/gsMap", "- First beta.", "v1.73.8"
    )
    assert 'uv pip install "gsMap[cpu]==2.0.0b1" --torch-backend=cpu' in notes
    assert "docker pull" not in notes
    assert "compare/v1.73.8...v2.0.0b1" in notes
    assert "- First beta." in notes


def test_final_notes_use_lowercase_image_name():
    notes = MODULE.release_notes("v2.0.0", False, "JianYang-Lab/gsMap", "- Stable.")
    assert "docker pull ghcr.io/jianyang-lab/gsmap:2.0.0" in notes
    assert "compare/" not in notes


def test_changelog_covers_the_package_version():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert MODULE.changelog_section((ROOT / "CHANGELOG.md").read_text(), version)
