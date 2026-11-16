"""Build a tiny real report, then check labels under file:// in Playwright.

Run --build with the gsMap environment, --check with a Playwright environment.
Only the supplied test directory is written; no existing report is edited.
"""

import argparse
import json
from pathlib import Path


def build(output):
    import pandas as pd
    from PIL import Image

    from gsMap.report.association_overview import export_association_overview
    from gsMap.report.report import _render_report_html
    from gsMap.report.report_data import (
        _copy_js_assets,
        _export_annotation_schemas_js,
        _export_cauchy_js,
    )
    from gsMap.report.spatial_payload import export_sample_spatial_payloads
    from gsMap.report.trait_metadata import trait_payload_js

    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "index.html").exists(), "Use a fresh test directory"
    _copy_js_assets(output)
    js = output / "js_data"
    (js / "gss_stats").mkdir(parents=True)
    traits = ["A B", "A_B"]
    samples = ["slice1", "slice2"]
    labels = {"A B": "Intelligence — cognitive performance", "A_B": "Schizophrenia"}
    meta = dict(
        traits=traits,
        trait_labels=labels,
        samples=samples,
        annotations=["celltype"],
        dataset_type="spatial2D",
        top_corr_genes=10,
        trait_payload_version=2,
        association_overview_version=1,
    )
    (js / "report_meta.js").write_text("window.GSMAP_REPORT_META=" + json.dumps(meta) + ";")
    frame = pd.DataFrame(
        {
            "spot": [f"s{i}" for i in range(8)],
            "sample_name": samples * 4,
            "sx": [0, 0, 1, 1, 0, 0, 1, 1],
            "sy": [0, 0, 0, 0, 1, 1, 1, 1],
            "celltype": ["Brain", "Heart"] * 4,
            "A B": range(8),
            "A_B": range(8, 0, -1),
        }
    )
    index, _ = export_sample_spatial_payloads(
        metadata=frame,
        samples=samples,
        traits=traits,
        annotations=["celltype"],
        js_data_dir=js,
        max_points=100,
    )
    (js / "sample_index.js").write_text("window.GSMAP_SAMPLE_INDEX=" + json.dumps(index) + ";")
    _export_annotation_schemas_js(js, frame, ["celltype"])
    rows = []
    for i, trait in enumerate(traits):
        gene = [{"gene": "TEST", "PCC": 0.9 - i / 10, "trait": trait}]
        (js / "gss_stats" / f"gene_trait_correlation_{trait}.js").write_text(
            trait_payload_js("GSMAP_GENE_TRAIT_CORRELATION", trait, json.dumps(gene))
        )
        snps = dict(
            x=[1, 2],
            y=[i + 1, i + 2],
            gene=["TEST", ""],
            chr=[1, 1],
            snp=["rs1", "rs2"],
            is_top=[1, 0],
            chrom_ticks={"1": 1.5},
        )
        (js / f"manhattan_{trait}.js").write_text(
            trait_payload_js("GSMAP_MANHATTAN", trait, json.dumps(snps))
        )
        for annotation in ["Brain", "Heart"]:
            rows.append(
                dict(
                    trait=trait,
                    annotation=annotation,
                    annotation_name="celltype",
                    type="aggregated",
                    mlog10_p_cauchy=i + 3,
                    mlog10_p_median=i + 2,
                    top_95_quantile=i + 4,
                )
            )
    pd.DataFrame(rows).to_csv(output / "cauchy_results.csv", index=False)
    _export_cauchy_js(output, js)
    frame.to_csv(output / "spot_metadata.csv", index=False)
    export_association_overview(frame, output, js, traits, ["celltype"])
    (js / "umap_data.js").write_text("window.GSMAP_UMAP=null;")
    plots = output / "gene_diagnostic_plots"
    plots.mkdir()
    for trait in traits:
        for sample in samples:
            for mode in ["exp", "gss"]:
                Image.new("RGB", (10, 10), "white").save(
                    plots / f"gene_{trait}_TEST_{sample}_{mode}.png"
                )
    (output / "index.html").write_text(_render_report_html("Trait label test"))


def check(output):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: print(message.text, flush=True) if message.type == "error" else None,
        )
        page.goto((output / "index.html").as_uri(), wait_until="networkidle")
        page.wait_for_function(
            "window.Alpine && Alpine.$data(document.body).dataLoaded && Alpine.$data(document.body).plotlyReady"
        )
        page.wait_for_function("Alpine.$data(document.body).overviewData?.rows.length > 0")
        assert page.evaluate("Alpine.$data(document.body).overviewData.n_spots") == 4
        assert page.evaluate("Alpine.$data(document.body).overviewScope") == "sample"
        page.get_by_role("radio", name="All sections", exact=True).check()
        page.wait_for_function("Alpine.$data(document.body).overviewData?.n_spots === 8")
        page.get_by_role("button", name="Toggle category Brain", exact=True).click()
        assert page.evaluate("Alpine.$data(document.body).selectedAnnotationCategories") == [
            "Brain"
        ]
        page.get_by_role("button", name="Toggle category Heart", exact=True).click()
        assert page.evaluate("Alpine.$data(document.body).selectedAnnotationCategories") == [
            "Brain",
            "Heart",
        ]
        page.get_by_role("button", name="Clear annotation selection", exact=True).click()
        selector = page.locator("#trait-selector")
        assert selector.input_value() == "Intelligence — cognitive performance"
        selector.click()
        selector.fill("schizo")
        page.get_by_role("option", name="Schizophrenia", exact=True).click()
        page.wait_for_function("Alpine.$data(document.body).selectedTrait === 'A_B'")
        page.locator("#sec-gsmap").scroll_into_view_if_needed()
        page.get_by_label("Comparison type", exact=True).select_option("traits")
        try:
            page.wait_for_function(
                "!Alpine.$data(document.body).spatialComparisonBusy", timeout=15000
            )
        except Exception:
            print(
                json.dumps(
                    {
                        "errors": errors,
                        "state": page.evaluate("""() => {
                const a=Alpine.$data(document.body); return {busy:a.spatialComparisonBusy,
                    error:a.spatial2dError, ready:a.plotlyReady};}"""),
                    }
                ),
                flush=True,
            )
            raise
        assert page.locator("#comparison-search-traitA").input_value() == "Schizophrenia"
        assert page.locator("#comparison-search-traitB").input_value().startswith("Intelligence")
        page.evaluate("""async () => {const a=Alpine.$data(document.body);
            a.manhattanActivated=true; await a.loadManhattanData();
            await a.loadCauchyData(); await a.renderCauchyHeatmap(); }""")
        page.wait_for_function("document.getElementById('cauchyHeatmap').data?.length > 0")
        result = page.evaluate("""() => {const a=Alpine.$data(document.body),
            plot=document.getElementById('cauchyHeatmap');
            return {selected:a.selectedTrait,values:a.manhattanData.y,
                ids:plot.data[0].x,labels:plot.layout.xaxis.ticktext,
                comparison:a.spatialComparisonPanelTitle(1)};}""")
        assert result["values"] == [2, 3]
        assert set(result["ids"]) == {"A B", "A_B"}
        assert "Schizophrenia" in result["labels"]
        assert "Intelligence" in result["comparison"]
        page.evaluate("""() => {const a=Alpine.$data(document.body);
            a.cauchyHeatmapTranspose=true;a.renderCauchyHeatmap();}""")
        page.set_viewport_size({"width": 760, "height": 900})
        page.wait_for_timeout(250)
        assert not errors, errors
        print(json.dumps({"passed": True, "file_url": True, **result}, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (build if args.build else check)(args.output.resolve())
