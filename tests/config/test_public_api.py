"""The documented Python API stays stable: entry points, config fields and defaults."""

import json
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("public_api_snapshot.json")


def test_documented_entry_points_and_config_fields_are_unchanged():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from public_api_snapshot import build_snapshot

    expected = json.loads(SNAPSHOT.read_text())
    current = build_snapshot()
    assert current["entry_points"] == expected["entry_points"]
    assert current["configs"] == expected["configs"]
    assert current["DatasetType"] == expected["DatasetType"]
    # New names may be added to gsMap.config; none may disappear.
    assert set(expected["exports"]) <= set(current["exports"])
