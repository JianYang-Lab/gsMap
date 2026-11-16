from collections import Counter
from html.parser import HTMLParser

import pytest
from jinja2 import Template

from tests.paths import REPO_ROOT

TEMPLATE_PATH = REPO_ROOT / "src/gsMap/report/static/template.html"


class _TemplateAuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.label_targets: list[str] = []
        self.control_targets: list[str] = []
        self.report_card_headers: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.append(element_id)
        if tag == "label" and (label_target := attributes.get("for")):
            self.label_targets.append(label_target)
        if control_target := attributes.get("aria-controls"):
            self.control_targets.append(control_target)
        if tag == "button" and "report-card-header" in (attributes.get("class") or ""):
            self.report_card_headers.append(attributes)


@pytest.fixture
def report_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def rendered_report(report_template: str) -> str:
    return Template(report_template).render(
        title="gsMap report test",
        project_name="test_project",
        generated_at="2026-08-17 00:00:00",
        gsmap_version="test",
    )


def test_report_template_has_unique_ids_and_valid_control_targets(
    rendered_report: str,
) -> None:
    parser = _TemplateAuditParser()
    parser.feed(rendered_report)

    counts = Counter(parser.ids)
    assert [element_id for element_id, count in counts.items() if count > 1] == []
    assert set(parser.control_targets) <= set(parser.ids)
    assert set(parser.label_targets) <= set(parser.ids)


def test_report_sections_are_accessible_buttons(rendered_report: str) -> None:
    parser = _TemplateAuditParser()
    parser.feed(rendered_report)

    assert len(parser.report_card_headers) == 7
    assert all(header.get("type") == "button" for header in parser.report_card_headers)
    assert all(header.get(":aria-expanded") for header in parser.report_card_headers)
    assert all(header.get("aria-controls") for header in parser.report_card_headers)


def test_report_template_includes_responsive_navigation_and_error_state(
    report_template: str,
) -> None:
    assert 'id="report-sidebar"' in report_template
    assert 'class="mobile-sidebar-backdrop' in report_template
    assert 'id="main-content"' in report_template
    assert 'class="report-overview' in report_template
    assert "initializationError" in report_template
    assert "setupSectionObserver()" in report_template
    assert "scrollToSection(section)" in report_template
    assert "setAllSections(false)" in report_template
    assert 'x-init="init()"' not in report_template


def test_report_plot_layout_has_no_fixed_minimum_panel_widths(
    report_template: str,
) -> None:
    assert "min-w-[400px]" not in report_template
    assert "min-w-[500px]" not in report_template
    assert "height: 800px" not in report_template
    assert report_template.count('class="plot-grid"') == 4
    assert "spatial2dViewMode === 'compare' ? 'plot-grid'" in report_template


def test_spatial3d_is_lazy_direct_plotly_and_labels_nonphysical_geometry(
    report_template: str,
) -> None:
    assert "<iframe" not in report_template
    assert "initializeSpatial3d()" in report_template
    assert "GSMAP_SPATIAL_3D_INDEX" in report_template
    assert 'id="spatial3dPlotTrait"' in report_template
    assert 'id="spatial3dPlotAnnotation"' in report_template
    assert "resolveSpatial3dAspectMode" in report_template
    assert "Expanded section-order Z" in report_template
    assert 'x-model="spatial3dAspectMode"' not in report_template
    assert "spatial3dCoordinateInfo?.note" in report_template
    assert "xRange / largestXyRange" in report_template
    assert "yRange / largestXyRange" in report_template


def test_spatial3d_compare_waits_for_layout_then_resizes_and_recenters(
    report_template: str,
) -> None:
    assert "await this.waitForLayoutPaint()" in report_template
    assert "await this.resizeSpatial3dViews({ center: mode === 'compare' })" in report_template
    assert "Plotly.Plots.resize(plot)" in report_template
    assert "'scene.camera': camera" in report_template
    assert "this.fittedSpatial3dCamera" in report_template


def test_3d_fullscreen_has_accessible_button_and_fills_viewport(report_template: str) -> None:
    assert 'id="spatial3dFullscreenButton"' in report_template
    assert ':aria-pressed="spatial3dFullscreen.toString()"' in report_template
    assert "'Exit full screen' : 'Full screen 3D map'" in report_template
    assert '@fullscreenchange.document="onSpatial3dFullscreenChange()"' in report_template
    assert "#sec-spatial3d:fullscreen .plot-frame-3d" in report_template
    assert "document.fullscreenElement || document.body" in report_template


def test_2d_fullscreen_keeps_controls_and_hides_static_overviews(report_template: str) -> None:
    assert 'id="spatial2dFullscreenButton"' in report_template
    assert ':aria-pressed="spatial2dFullscreen.toString()"' in report_template
    assert "'Exit full screen' : 'Full screen 2D map'" in report_template
    assert '@fullscreenchange.document="onSpatial2dFullscreenChange()"' in report_template
    assert "#sec-gsmap:fullscreen .spatial-2d-plot" in report_template
    assert "#sec-gsmap:fullscreen .spatial2d-overviews { display: none; }" in report_template
    assert "max-width: none" in report_template


def test_large_secondary_datasets_are_loaded_near_their_sections(
    report_template: str,
) -> None:
    assert '<script src="js_data/cauchy_results.js"></script>' not in report_template
    assert '<script src="js_data/umap_data.js"></script>' not in report_template
    assert "loadScript('js_data/cauchy_index.js')" in report_template
    assert "loadCauchyShard(entry, cache, order" in report_template
    assert "setupDeferredPlotLoading()" in report_template
    assert "loadUmapData()" in report_template
    assert "loadCauchyData()" in report_template
    assert 'loading="lazy"' in report_template


def test_trait_selector_supports_fuzzy_keyboard_search(report_template: str) -> None:
    assert 'role="combobox"' in report_template
    assert 'id="trait-listbox"' in report_template
    assert "scoreTraitMatch(label, query)" in report_template
    assert "moveTraitSearch(1)" in report_template
    assert "selectActiveTrait()" in report_template


def test_sample_selector_supports_fuzzy_keyboard_search(report_template: str) -> None:
    assert 'id="sample-listbox"' in report_template
    assert "get filteredSamples()" in report_template
    assert "moveSampleSearch(1)" in report_template
    assert "selectActiveSample()" in report_template
    assert "selectSampleFromSearch(sample)" in report_template


def test_spatial2d_has_scalable_controls_and_synchronized_comparison(
    report_template: str,
) -> None:
    assert "spatial2dViewMode: 'compare'" in report_template
    assert "spatial3dViewMode: 'compare'" in report_template
    assert "setSpatial2dViewMode('compare')" in report_template
    assert "setSpatial2dViewMode('annotation')" in report_template
    assert 'aria-label="2D point-size multiplier"' in report_template
    assert 'max="12" step="0.25"' in report_template
    assert 'aria-label="2D point opacity"' not in report_template
    assert "spatial2dOpacity" not in report_template
    assert "bindSpatial2dViewportSync" in report_template
    assert "Color scale is shared across samples" in report_template


def test_spatial_views_offer_scalable_hover_details(report_template: str) -> None:
    assert "marker:{color:points.hoverValues}" in report_template
    assert "bindSpatial3dHover" in report_template
    assert "gl.mouseListener.enabled = false" in report_template
    assert "appendPointAnnotationLines(lines, context, sourceIndex)" in report_template
    assert "setPointHoverContext(key, context)" in report_template
    assert "buildSpatial2dAnnotationHoverText" not in report_template
    assert "Object.entries(data.annotations || {})" in report_template
    assert "<b>${anno}</b> code: %{marker.color:.0f}" not in report_template


def test_umap_uses_readable_geometry_hover_and_canonical_annotation_colors(
    report_template: str,
) -> None:
    assert 'class="umap-frame"' in report_template
    assert "resolveUmapPointSize" in report_template
    assert "scaleanchor: 'y'" in report_template
    assert "strongest trait signals are drawn on top" in report_template
    assert "currentAnnotationLegend" in report_template
    assert "this.getAnnotationColorMap(anno)" in report_template
    continuous_renderer = report_template[
        report_template.index(
            "                renderSingleUmapContinuous(divId"
        ) : report_template.index("// Optimized: render discrete")
    ]
    assert "includeSpots" not in continuous_renderer
    assert "spots:" not in continuous_renderer
    assert "bindPointHoverTooltip" in report_template
    assert "pointer.clientX + gap" in report_template
    assert "_pointHoverContextOrder.length > 4096" in report_template
    assert "customdata: rendering.hoverColors" not in report_template
    assert "customdata: indices.map(() => color)" not in report_template


def test_spatial2d_uses_plane_labels_and_point_colored_offset_tooltips(
    report_template: str,
) -> None:
    assert "data.axis_titles" in report_template
    assert "meta: { hoverContextKey }" in report_template
    assert "hoverKind: 'trait'" in report_template
    assert "hoverKind: 'annotation'" in report_template
    assert "spatial2dTraitHoverColors" not in report_template
    assert "this.bindPointHoverTooltip(plotDiv)" in report_template
    assert "tooltip.style.background = background" in report_template


def test_spatial2d_disables_native_hover_and_automatic_margin_redraws(
    report_template: str,
) -> None:
    renderer = report_template.split(
        "renderSpatial2DPlotSingle(divId, plotType, options = {}) {", 1
    )[1].split("bindSpatial2dViewportSync(sourceId, targetId)", 1)[0]
    assert "hovermode: false" in renderer
    assert "autoexpand: false" in renderer
    assert "this.bindSpatial2dHover(plotDiv)" in renderer
    assert "this.bindPointHoverTooltip(plotDiv)" not in renderer


def test_shared_annotation_selection_filters_all_point_maps(report_template: str) -> None:
    assert "visibleUmapPointIndices()" in report_template
    assert "const annotation = this.spatial3dAnnotationCache?.[annotationName]" in report_template
    assert "this.renderSpatial2DPlotSingle('spatial2dPlotTrait', 'trait')" in report_template
    assert "this.renderSpatial3dViews().catch(error => {" in report_template


def test_enrichment_uses_shared_annotation_color_reference(report_template: str) -> None:
    assert '<script src="js_data/annotation_schemas.js"></script>' in report_template
    assert "window.GSMAP_ANNOTATION_SCHEMAS" in report_template
    assert "getAnnotationColor(row.annotation, row.annotation_name)" in report_template
    assert "annotationColorTrace" in report_template
    assert "Same color as spatial and UMAP views" in report_template
    assert "toggleAnnotationFromEnrichment" not in report_template
    assert "clearAnnotationSelection()" in report_template
    assert "Filtered by sidebar annotation selection." in report_template

    table_filter = report_template[
        report_template.index("get filteredCauchyTable()") : report_template.index(
            "// Wait for Plotly to load"
        )
    ]
    heatmap = report_template[
        report_template.index("                renderCauchyHeatmap() {") : report_template.index(
            "// Hierarchical clustering"
        )
    ]
    assert "isAnnotationCategoryVisible" in table_filter
    assert "cauchyFocusedAnnotation" not in table_filter
    assert "isAnnotationCategoryVisible" in heatmap


def test_enrichment_is_sharded_paginated_and_uses_bounded_browser_caches(
    report_template: str,
) -> None:
    assert "cauchyTablePageSize: 200" in report_template
    assert "get visibleCauchyTable()" in report_template
    assert "_cauchyShardPromises" in report_template
    assert "setBoundedCache(cache, order, entry.key, rows, 4)" in report_template
    assert "const values = new Map(), traitSet = new Set()" in report_template
    assert "enrichmentModel?.source === rows" in report_template
    assert "cauchyHeatmapTraitsPerPage: 24" in report_template
    assert "window.GSMAP_CAUCHY =" not in report_template
    assert "fetch(" not in report_template


def test_spatial3d_keeps_full_display_payload_during_camera_interaction(
    report_template: str,
) -> None:
    assert "spatial3dInteractionPointLimit" not in report_template
    assert "configureSpatial3dInteractionLod" not in report_template
    assert "applySpatial3dInteractionLod" not in report_template
    camera_sync = report_template.split("bindSpatial3dCameraSync(sourceId, targetId) {", 1)[
        1
    ].split("fittedSpatial3dCamera(current = null)", 1)[0]
    assert "plotly_relayouting" in camera_sync
    assert "queueSpatial3dCamera(target, camera)" in camera_sync


def test_spatial_association_views_have_independent_q95_opacity_controls(
    report_template: str,
) -> None:
    assert "spatial3dAssociationMinLogP: 0" in report_template
    assert "spatial2dAssociationMinLogP: 0" in report_template
    assert "spatial3dAssociationOpacityStrength: 0.65" in report_template
    assert "spatial2dAssociationOpacityStrength: 0," in report_template
    assert report_template.count('aria-label="Association opacity gradient strength"') == 1
    assert report_template.count('aria-label="2D association opacity gradient strength"') == 1
    assert "filterAssociationPlotPoints(points, values, thresholdValue = 0)" in report_template
    assert "value, cmin, q95, maximumOpacity, strengthValue, thresholdValue" in report_template
    assert "const binCount = 8" in report_template
    assert "counts[b] ? sum / counts[b] : 0" in report_template
    assert "this.spatial3dViewMode === 'compare' && traitFeature" in report_template
    assert "this.spatial2dViewMode === 'compare' && comparisonTrait?.values" in report_template


def test_report_exposes_lazy_all_sample_trait_overview(report_template: str) -> None:
    assert "One trait across all samples" in report_template
    assert "One sample across all traits" in report_template
    assert "multiSampleTraitPlotUrl" in report_template
    assert "singleSampleMultiTraitPlotUrl" in report_template
    assert (
        "<template x-if=\"allSampleTraitOverviewOpen && multiSamplePlotStatus !== 'error'\">"
    ) in report_template
    assert (
        "<template x-if=\"singleSampleTraitOverviewOpen && singleSamplePlotStatus !== 'error'\">"
    ) in report_template
    assert ':src="multiSampleTraitPlotUrl"' in report_template
    assert ':src="singleSampleMultiTraitPlotUrl"' in report_template
    assert "allSampleTraitOverviewOpen ? multiSampleTraitPlotUrl : ''" not in report_template
    assert (
        "singleSampleTraitOverviewOpen ? singleSampleMultiTraitPlotUrl : ''" not in report_template
    )
    assert "multiSamplePlotStatus === 'error'" in report_template
    assert "singleSamplePlotStatus === 'error'" in report_template
    assert 'decoding="async"' in report_template


def test_run_details_are_collapsed_and_summarized_before_raw_config(
    report_template: str,
) -> None:
    assert "info: false" in report_template
    assert "get runDetailCards()" in report_template
    assert "Graph and neighborhood" in report_template
    assert "Browser visualization" in report_template
    assert "Configuration" in report_template
    assert "Complete machine-readable configuration" not in report_template
    assert "JSON.stringify(reportMeta.configuration || {}, null, 2)" in report_template


def test_spatial_widgets_do_not_repeat_outer_panel_titles(report_template: str) -> None:
    assert "spatial3dLayout(title)" not in report_template
    assert "this.spatial3dLayout(`Trait:" not in report_template
    assert "this.spatial3dLayout(`Annotation:" not in report_template


def test_sidebar_is_the_single_annotation_selection_control(
    report_template: str,
) -> None:
    assert "selectedAnnotationCategories" in report_template
    assert 'id="association-overview"' in report_template
    assert 'aria-label="Clear annotation selection"' in report_template
    assert "toggleAnnotationFromEnrichment" not in report_template
    sidebar = report_template.split('id="report-sidebar"', 1)[1].split("</aside>", 1)[0]
    assert 'id="association-overview"' in sidebar
    assert report_template.count('id="association-overview"') == 1
    assert sidebar.count('name="overview-scope"') == 2
    assert sidebar.count('name="overview-sort"') == 5
    assert 'x-model="overviewScope"' in sidebar
    assert 'x-model="overviewSort"' in sidebar
    assert 'aria-label="Sort annotation overview"' in sidebar

    toggle_start = report_template.index("                toggleAnnotationCategory(category) {")
    toggle_handler = report_template[
        toggle_start : report_template.index(
            "                clearAnnotationSelection() {", toggle_start
        )
    ]
    assert "this.selectedAnnotationCategories.filter(value => value !== key)" in toggle_handler
    assert "[...this.selectedAnnotationCategories, key]" in toggle_handler
    assert "annotationHiddenCategories" not in report_template


def test_annotation_first_layout_and_shared_interactive_legend(report_template: str) -> None:
    section_order = [
        report_template.index("{ key: 'show3d', id: 'sec-spatial3d'"),
        report_template.index("{ key: 'show2dDistribution', id: 'sec-gsmap'"),
        report_template.index("{ key: 'showUMAP', id: 'sec-umap'"),
        report_template.index("{ key: 'showEnrichment', id: 'sec-cauchy'"),
        report_template.index("{ key: 'showGSS', id: 'sec-genes'"),
    ]
    assert section_order == sorted(section_order)
    assert 'style="order: 1"' in report_template
    assert 'style="order: 2"' in report_template
    assert 'style="order: 3"' in report_template
    assert 'style="order: 4"' in report_template
    assert 'id="annotation-selection-help"' in report_template
    assert 'aria-describedby="annotation-selection-help"' in report_template
    for keyword in ["Click", "select", "remove", "Clear"]:
        assert (
            f'<strong class="font-semibold text-primary-700">{keyword}</strong>' in report_template
        )
    assert "toggleAnnotationCategory(row.category)" in report_template
    assert "scheduleAnnotationLegendToggle" not in report_template
    assert "filterCategoricalPlotPoints" in report_template
    assert "visible: this.isAnnotationCategoryVisible(val, annoName)" in report_template


def test_dataset_counts_are_only_in_top_overview(report_template: str) -> None:
    assert report_template.count('x-text="traits.length.toLocaleString()"') == 1
    assert report_template.count('x-text="samples.length.toLocaleString()"') == 1
    assert report_template.count('x-text="annotations.length.toLocaleString()"') == 1
    assert ">Dataset</h3>" not in report_template
