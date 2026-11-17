"""Report generation: Manhattan positions, bundled libraries."""

import numpy as np
import pandas as pd

from gsMap.utils.manhattan_plot import _ManhattanPlot


def test_manhattan_positions_do_not_depend_on_row_order():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "CHR": np.repeat([1, 2, 3], 40),
            "BP": np.concatenate([np.sort(rng.integers(1, 10_000_000, 40)) for _ in range(3)]),
            "P": rng.uniform(1e-8, 1, 120),
            "SNP": [f"rs{i}" for i in range(120)],
            "GENE": [f"g{i}" for i in range(120)],
        }
    )
    ordered = _ManhattanPlot(frame.copy()).data.set_index("SNP")["POSITION"]
    shuffled = _ManhattanPlot(frame.sample(frac=1, random_state=1).reset_index(drop=True))
    positions = shuffled.data.set_index("SNP")["POSITION"].loc[ordered.index]
    pd.testing.assert_series_equal(positions, ordered)
    # Each chromosome starts after the previous chromosome's largest position.
    by_chr = ordered.groupby(frame.set_index("SNP")["CHR"]).agg(["min", "max"])
    assert (by_chr["min"].iloc[1:].values > by_chr["max"].iloc[:-1].values).all()


def test_report_libraries_are_refreshed_only_when_they_differ(tmp_path):
    from gsMap.report.report_data import _copy_js_assets

    _copy_js_assets(tmp_path)
    libraries = sorted(path.name for path in (tmp_path / "js_lib").iterdir())
    assert libraries == ["alpine.min.js", "plotly.min.js"]

    stale = tmp_path / "js_lib" / "alpine.min.js"
    stale.write_text("old library")
    plotly = tmp_path / "js_lib" / "plotly.min.js"
    before = plotly.stat().st_mtime_ns
    _copy_js_assets(tmp_path)
    assert stale.read_text() != "old library"  # replaced
    assert plotly.stat().st_mtime_ns == before  # identical: left alone
