"""Profile a whole gsMap report in Chromium: load, each section, and the main controls.

The report is served with ``gsmap report-view`` exactly as users open it. Each phase (page
load, scrolling every section into view, switching trait, annotation, sample and gene) is
timed until the page settles: no request in flight and no long task for a while. For each
phase the script records long tasks, total blocking time and, with ``--cpu-profile``, CPU
time per script (Tailwind, Plotly, Alpine, report code, data payloads). Screenshots of every
section are saved after the sections have loaded, for appearance comparisons.

Example:
    python tests/helpers/profile_report.py --report results/e16/gsmap_web_report \\
        --python .venv/bin/python --output profile/e16-baseline --cpu-profile
"""

import argparse
import collections
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_report_3d import INIT_SCRIPT, READY_JS, free_port, summarize_tasks

APP = "Alpine.$data(document.body)"
QUIET_MS = 1200


def script_group(url):
    name = url.rsplit("/", 1)[-1].split("?", 1)[0]
    if not url:
        return "(native)"
    if name.startswith("tailwind"):
        return "tailwind"
    if name.startswith("plotly"):
        return "plotly"
    if name.startswith("alpine"):
        return "alpine"
    if "/js_data/" in url:
        return "payload"
    return "report"


class Recorder:
    """Track requests in flight and CPU samples for the current phase."""

    def __init__(self, page, cdp, cpu_profile):
        self.page, self.cdp, self.cpu_profile = page, cdp, cpu_profile
        self.in_flight = set()
        page.on("request", lambda request: self.in_flight.add(request))
        for event in ("requestfinished", "requestfailed"):
            page.on(event, lambda request: self.in_flight.discard(request))
        if cpu_profile:
            cdp.send("Profiler.enable")
            cdp.send("Profiler.setSamplingInterval", {"interval": 500})

    def now(self):
        return self.page.evaluate("() => performance.now()")

    def busy_until(self, since):
        return self.page.evaluate(
            """since => {
                const ends = window.__gsmapProfile.longTasks
                    .map(task => task.start + task.duration).filter(end => end >= since);
                for (const entry of performance.getEntriesByType('resource')) {
                    if (entry.responseEnd >= since) ends.push(entry.responseEnd);
                }
                return Math.max(since, ...ends);
            }""",
            since,
        )

    def phase(self, action, timeout_s=120):
        """Run ``action`` and wait until the page is quiet; return timing for the phase."""
        if self.cpu_profile:
            self.cdp.send("Profiler.start")
        start = self.now()
        action()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.page.wait_for_timeout(200)
            if self.in_flight:
                continue
            if self.now() - self.busy_until(start) >= QUIET_MS:
                break
        end = self.busy_until(start)
        tasks = self.page.evaluate("() => window.__gsmapProfile.longTasks")
        result = {"settled_ms": round(end - start, 1), **summarize_tasks(tasks, start, end + 1)}
        if self.cpu_profile:
            result["cpu"] = summarize_profile(self.cdp.send("Profiler.stop")["profile"])
        return result


def summarize_profile(profile):
    nodes = {node["id"]: node for node in profile["nodes"]}
    parents = {}
    for node in profile["nodes"]:
        for child in node.get("children", []):
            parents[child] = node["id"]
    samples = profile["samples"]
    interval = (profile["endTime"] - profile["startTime"]) / max(1, len(samples)) / 1000
    groups, functions = collections.Counter(), collections.Counter()
    for node_id in samples:
        frame = nodes[node_id]["callFrame"]
        name = frame["functionName"]
        if name in ("(idle)", "(program)", "(garbage collector)"):
            groups[name] += interval
        else:
            groups[script_group(frame["url"])] += interval
        # Inclusive time of report functions (once per sample).
        seen, current = set(), node_id
        while current is not None:
            frame = nodes[current]["callFrame"]
            if script_group(frame["url"]) == "report" and frame["functionName"]:
                key = frame["functionName"]
                if key not in seen:
                    functions[key] += interval
                    seen.add(key)
            current = parents.get(current)
    return {
        "by_script_ms": {key: round(value) for key, value in groups.most_common()},
        "report_functions_ms": {key: round(value) for key, value in functions.most_common(12)},
    }


def wait_for_load(page, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = page.evaluate(READY_JS)
        if state is True or state == "2d":
            return
        if isinstance(state, str) and state.startswith("error"):
            raise RuntimeError(state)
        page.wait_for_timeout(100)
    raise RuntimeError("Report did not finish loading")


def screenshot_sections(page, output, label):
    saved = []
    ids = page.evaluate(
        "() => [...document.querySelectorAll('[id^=sec-]')]"
        ".filter(e => e.getClientRects().length).map(e => e.id)"
    )
    for section_id in ids:
        locator = page.locator(f"#{section_id}")
        locator.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        path = output / f"{label}__{section_id}.png"
        locator.screenshot(path=str(path), animations="disabled")
        saved.append(path.name)
    page.screenshot(path=str(output / f"{label}__top.png"), animations="disabled")
    return saved


def profile(page, recorder, url, output, timeout_s):
    phases = {}
    phases["load"] = recorder.phase(
        lambda: (page.goto(url, wait_until="load"), wait_for_load(page, timeout_s)), timeout_s
    )
    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(300)
    phases["top_screenshot"] = (
        page.screenshot(path=str(output / "00_initial__top.png"), animations="disabled")
        and "00_initial__top.png"
    )
    section_ids = page.evaluate(
        "() => [...document.querySelectorAll('[id^=sec-]')]"
        ".filter(e => e.getClientRects().length).map(e => e.id)"
    )
    for section_id in section_ids:
        phases[f"scroll:{section_id}"] = recorder.phase(
            lambda section_id=section_id: page.evaluate(
                "id => document.getElementById(id).scrollIntoView({block: 'start'})", section_id
            ),
            timeout_s,
        )
    phases["screenshots_loaded"] = screenshot_sections(page, output, "01_loaded")
    state = page.evaluate(
        f"""() => {{ const a = {APP};
            return {{traits: a.traits, samples: a.samples, annotations: a.annotations,
                     genes: [...document.querySelectorAll('#gene-selector option')]
                         .map(o => o.value).filter(Boolean)}}; }}"""
    )
    if len(state["traits"]) > 1:
        for label, trait in (
            ("trait_switch", state["traits"][1]),
            ("trait_back", state["traits"][0]),
        ):
            phases[label] = recorder.phase(
                lambda trait=trait: page.evaluate(f"t => {APP}.selectTraitFromSearch(t)", trait),
                timeout_s,
            )
    if len(state["annotations"]) > 1:
        for label, value in (
            ("annotation_switch", state["annotations"][1]),
            ("annotation_back", state["annotations"][0]),
        ):
            phases[label] = recorder.phase(
                lambda value=value: page.select_option("#annotation-selector", value), timeout_s
            )
    if len(state["samples"]) > 1:
        for label, sample in (
            ("sample_switch", state["samples"][1]),
            ("sample_back", state["samples"][0]),
        ):
            phases[label] = recorder.phase(
                lambda sample=sample: page.evaluate(
                    f"s => {APP}.selectSampleFromSearch(s)", sample
                ),
                timeout_s,
            )
    if len(state["genes"]) > 1:
        phases["gene_switch"] = recorder.phase(
            lambda: page.select_option("#gene-selector", state["genes"][1]), timeout_s
        )
    phases["screenshots_end"] = screenshot_sections(page, output, "02_after_controls")
    return phases


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True, help="gsmap_web_report directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="Python with gsMap installed")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--cpu-profile", action="store_true", help="Record CPU time per script")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    port = free_port()
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=f"{repo / 'src'}:{repo}")
    command = [str(args.python), "-m", "gsMap.cli", "report-view", str(args.report.resolve()),
               "--port", str(port), "--no-browser"]  # fmt: skip
    errors, console_errors = [], []
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(240):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.25)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    args=["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
                )
                page = browser.new_page(
                    viewport={"width": 1500, "height": 1100}, device_scale_factor=1
                )
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "console",
                    lambda message: (
                        message.type == "error" and console_errors.append(message.text)
                    ),
                )
                page.add_init_script(INIT_SCRIPT)
                cdp = page.context.new_cdp_session(page)
                recorder = Recorder(page, cdp, args.cpu_profile)
                result = profile(page, recorder, f"http://127.0.0.1:{port}/", args.output,
                                 args.timeout)  # fmt: skip
                result["heap_end_mb"] = round(
                    page.evaluate("() => performance.memory?.usedJSHeapSize || 0") / 2**20, 1
                )
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=30)
    result["page_errors"] = errors
    result["console_errors"] = console_errors
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, phase in result.items():
        if isinstance(phase, dict) and "settled_ms" in phase:
            cpu = phase.get("cpu", {}).get("by_script_ms", {})
            print(f"{name:28s} settled {phase['settled_ms']:8.0f} ms  longest {phase['longest_ms']:6.0f}"
                  f"  TBT {phase['total_blocking_ms']:6.0f}  {cpu}")  # fmt: skip
    print("errors:", errors, console_errors[:5])


if __name__ == "__main__":
    main()
