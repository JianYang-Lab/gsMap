"""Profile the 3D viewer of a gsMap report in Chromium; save metrics and fixed-camera images.

The report is served with ``gsmap report-view`` exactly as users open it. Recorded:
load-time long tasks and total blocking time, time until both 3D views are drawn,
frame times while orbiting, latency of the 3D controls and JS heap size. Screenshots of
the 3D views at fixed cameras are saved for appearance comparisons between templates.

Example:
    python tests/helpers/profile_report_3d.py --report results/e11/gsmap_web_report \\
        --python .venv/bin/python --output profile/e11-baseline
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Collect long tasks from navigation start; the page cannot run code earlier.
INIT_SCRIPT = """
window.__gsmapProfile = {longTasks: [], frames: []};
try {
    new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
            window.__gsmapProfile.longTasks.push({start: entry.startTime, duration: entry.duration});
        }
    }).observe({type: 'longtask', buffered: true});
} catch (error) {}
"""

READY_JS = """
() => {
    const app = window.Alpine && Alpine.$data(document.body);
    if (!app || !app.dataLoaded) return false;
    if (!app.has3dWidget) return '2d';
    if (app.spatial3dError) return 'error: ' + app.spatial3dError;
    if (app.spatial3dTraitLoading || app.spatial3dAnnotationLoading) return false;
    const ids = app.spatial3dViewMode === 'trait' ? ['spatial3dPlotTrait']
        : app.spatial3dViewMode === 'annotation' ? ['spatial3dPlotAnnotation']
        : ['spatial3dPlotTrait', 'spatial3dPlotAnnotation'];
    return ids.every(id => {
        const gl = document.getElementById(id)?._fullLayout?.scene?._scene?.glplot;
        return Boolean(gl && gl.__gsMapPointLayer && gl.__gsMapPointLayer.pointCount > 0);
    });
}
"""

FRAMES_JS = """
async (milliseconds) => {
    const stamps = [];
    const end = performance.now() + milliseconds;
    await new Promise(resolve => {
        const step = now => { stamps.push(now); if (now < end) requestAnimationFrame(step); else resolve(); };
        requestAnimationFrame(step);
    });
    return stamps.slice(1).map((value, index) => value - stamps[index]);
}
"""

SETTLE_JS = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"

CAMERA = {
    "eye": {"x": 1.45, "y": -1.35, "z": 0.95},
    "up": {"x": 0, "y": 0, "z": 1},
    "center": {"x": 0, "y": 0, "z": 0},
}


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize_tasks(tasks, start=0.0, end=float("inf")):
    window = [task for task in tasks if start <= task["start"] < end]
    return {
        "count": len(window),
        "longest_ms": round(max((task["duration"] for task in window), default=0.0), 1),
        "total_blocking_ms": round(sum(max(0.0, task["duration"] - 50) for task in window), 1),
    }


def timed_action(page, script, argument=None):
    """Run an in-page async action; report wall time and long tasks while it ran."""
    before = page.evaluate("() => performance.now()")
    page.evaluate(script, argument)
    page.evaluate(SETTLE_JS)
    after = page.evaluate("() => performance.now()")
    tasks = page.evaluate("() => window.__gsmapProfile.longTasks")
    return {"wall_ms": round(after - before, 1), **summarize_tasks(tasks, before, after)}


def screenshot_views(page, output, label):
    page.evaluate(
        """async camera => {
            for (const id of ['spatial3dPlotTrait', 'spatial3dPlotAnnotation']) {
                const plot = document.getElementById(id);
                if (plot?.data && plot.getClientRects().length) {
                    await Plotly.relayout(plot, {'scene.camera': camera});
                }
            }
        }""",
        CAMERA,
    )
    page.evaluate(SETTLE_JS)
    page.wait_for_timeout(500)
    saved = []
    for element_id in ("spatial3dPlotTrait", "spatial3dPlotAnnotation"):
        element = page.locator(f"#{element_id}")
        if element.count() and element.is_visible():
            path = output / f"{label}__{element_id}.png"
            element.screenshot(path=str(path), animations="disabled")
            saved.append(path.name)
    return saved


def orbit(page, frames_ms=2000):
    box = page.locator("#spatial3dPlotTrait").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    start = page.evaluate("() => performance.now()")
    # Start recording without awaiting it, so frames are sampled during the drag.
    page.evaluate(f"ms => {{ window.__gsmapFrames = ({FRAMES_JS})(ms); }}", frames_ms)
    page.mouse.down()
    steps = 60
    for index in range(steps):
        page.mouse.move(x + 3 * (index + 1), y + (index % 2))
        page.wait_for_timeout(frames_ms / steps)
    page.mouse.up()
    intervals = page.evaluate("() => window.__gsmapFrames")
    end = page.evaluate("() => performance.now()")
    tasks = page.evaluate("() => window.__gsmapProfile.longTasks")
    return {
        "frames": len(intervals),
        "frame_ms_median": round(statistics.median(intervals), 1) if intervals else None,
        "frame_ms_p95": round(percentile(intervals, 0.95), 1) if intervals else None,
        "frame_ms_max": round(max(intervals), 1) if intervals else None,
        **summarize_tasks(tasks, start, end),
    }


def heap_mb(cdp):
    metrics = {m["name"]: m["value"] for m in cdp.send("Performance.getMetrics")["metrics"]}
    return round(metrics.get("JSHeapUsedSize", 0) / 2**20, 1)


def profile(page, cdp, url, output, timeout_s):
    result = {}
    t0 = time.monotonic()
    page.goto(url, wait_until="load")
    deadline = time.monotonic() + timeout_s
    state = False
    while time.monotonic() < deadline:
        state = page.evaluate(READY_JS)
        if (
            state is True
            or state == "2d"
            or (isinstance(state, str) and state.startswith("error"))
        ):
            break
        page.wait_for_timeout(100)
    if state == "2d":
        # 2D reports (no 3D viewer): record start-up only, as a regression control.
        page.evaluate(SETTLE_JS)
        ready_at = page.evaluate("() => performance.now()")
        result["wall_until_ready_s"] = round(time.monotonic() - t0, 2)
        page.wait_for_timeout(5000)
        tasks = page.evaluate("() => window.__gsmapProfile.longTasks")
        result["load"] = summarize_tasks(tasks, 0, ready_at + 5000)
        result["heap_after_load_mb"] = heap_mb(cdp)
        result["screenshots"] = []
        return result
    if state is not True:
        raise RuntimeError(f"3D views were not ready: {state}")
    page.evaluate(SETTLE_JS)
    ready_at = page.evaluate("() => performance.now()")
    result["wall_until_3d_ready_s"] = round(time.monotonic() - t0, 2)
    result["page_time_until_3d_ready_ms"] = round(ready_at, 1)
    page.wait_for_timeout(1500)
    tasks = page.evaluate("() => window.__gsmapProfile.longTasks")
    result["load"] = summarize_tasks(tasks, 0, ready_at + 1500)
    result["heap_after_load_mb"] = heap_mb(cdp)
    result["renderer"] = page.evaluate(
        """() => { const gl = document.createElement('canvas').getContext('webgl');
                   const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
                   return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null; }"""
    )
    result["points"] = page.evaluate(
        "() => Alpine.$data(document.body).spatial3dIndex?.n_points ?? null"
    )
    result["screenshots"] = screenshot_views(page, output, "01_default")
    result["orbit"] = orbit(page)
    app = "Alpine.$data(document.body)"
    actions = {
        "opacity_to_1": f"async () => {{ const a={app}; a.spatial3dOpacity=1; await a.renderSpatial3dViews(); }}",
        "opacity_back": f"async () => {{ const a={app}; a.spatial3dOpacity=0.72; await a.renderSpatial3dViews(); }}",
        "threshold_2": f"async () => {{ const a={app}; a.spatial3dAssociationMinLogP=2; await a.renderSpatial3dViews(); }}",
        "threshold_0": f"async () => {{ const a={app}; a.spatial3dAssociationMinLogP=0; await a.renderSpatial3dViews(); }}",
        "point_size_2": f"async () => {{ const a={app}; a.spatial3dPointSize=2; await a.renderSpatial3dViews(); }}",
        "point_size_1": f"async () => {{ const a={app}; a.spatial3dPointSize=1; await a.renderSpatial3dViews(); }}",
    }
    result["controls"] = {}
    for name, script in actions.items():
        result["controls"][name] = timed_action(page, script)
    result["screenshots"] += screenshot_views(page, output, "02_after_controls")
    # Opacity 1 and a signal cutoff, for appearance comparisons of other render paths.
    timed_action(page, actions["opacity_to_1"])
    result["screenshots"] += screenshot_views(page, output, "03_opaque")
    timed_action(page, actions["opacity_back"])
    timed_action(page, actions["threshold_2"])
    result["screenshots"] += screenshot_views(page, output, "04_threshold_2")
    timed_action(page, actions["threshold_0"])
    category = page.evaluate(
        f"() => {{ const a={app}; return a.getAnnotationSchema(a.selectedAnnotationCategory)?.categories?.[0] ?? null; }}"
    )
    if category is not None:
        result["controls"]["category_filter"] = timed_action(
            page,
            f"async c => {{ const a={app}; a.toggleAnnotationCategory(c); "
            "await new Promise(r => setTimeout(r, 50)); await a.renderSpatial3dViews(); }",
            category,
        )
        result["screenshots"] += screenshot_views(page, output, "05_category_filter")
        timed_action(
            page,
            f"async () => {{ const a={app}; a.clearAnnotationSelection(); "
            "await new Promise(r => setTimeout(r, 50)); await a.renderSpatial3dViews(); }",
        )
    traits = page.evaluate(f"() => {app}.traits")
    if len(traits) > 1:
        result["controls"]["trait_switch"] = timed_action(
            page,
            f"async t => {{ const a={app}; a.selectTraitFromSearch(t); "
            "for (let i=0;i<400 && (a.spatial3dTraitLoading || !a.spatial3dTraitCache[t] || "
            "a.spatial3dTraitData!==a.spatial3dTraitCache[t]);i++) "
            "await new Promise(r=>setTimeout(r,25)); await a.renderSpatial3dViews(); }",
            traits[1],
        )
    result["heap_end_mb"] = heap_mb(cdp)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True, help="gsmap_web_report directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="Python with gsMap installed")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds to wait for 3D views")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    port = free_port()
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=f"{repo / 'src'}:{repo}")
    command = [
        str(args.python),
        "-m",
        "gsMap.cli",
        "report-view",
        str(args.report.resolve()),
        "--port",
        str(port),
        "--no-browser",
    ]
    errors = []
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.25)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=not args.headed,
                    args=["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
                )
                page = browser.new_page(
                    viewport={"width": 1500, "height": 1100}, device_scale_factor=1
                )
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script(INIT_SCRIPT)
                cdp = page.context.new_cdp_session(page)
                cdp.send("Performance.enable")
                result = profile(page, cdp, f"http://127.0.0.1:{port}/", args.output, args.timeout)
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=30)
    result["page_errors"] = errors
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "screenshots"}, indent=2))


if __name__ == "__main__":
    main()
