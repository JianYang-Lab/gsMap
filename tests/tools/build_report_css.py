"""Rebuild the Tailwind CSS that is inlined into the report template.

The report used to compile Tailwind in the browser with the Play CDN script, scanning the
page on every load and on every class change. The same Tailwind version and theme are now
compiled once, here, from the template's own text; the result is inlined at the end of
``<head>``, where the Play CDN used to insert it, so the cascade is unchanged.

Run after changing classes in ``src/gsMap/report/static/template.html``::

    python tests/tools/build_report_css.py          # rewrite the inlined CSS
    python tests/tools/build_report_css.py --check  # exit 1 if it is out of date

Requires Node.js (``npx``); the Tailwind CLI is fetched by npm on first use.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TAILWIND_VERSION = "3.4.17"
TEMPLATE = Path(__file__).resolve().parents[2] / "src/gsMap/report/static/template.html"
START = '<style id="gsmap-tailwind">'
END = "</style>"

THEME = {
    "extend": {
        "colors": {
            "primary": {
                "50": "#eff6ff",
                "100": "#dbeafe",
                "200": "#bfdbfe",
                "300": "#93c5fd",
                "400": "#60a5fa",
                "500": "#3b82f6",
                "600": "#2563eb",
                "700": "#1d4ed8",
                "800": "#1e40af",
                "900": "#1e3a8a",
            }
        },
        "boxShadow": {
            "panel": "0 1px 2px rgba(15, 23, 42, 0.05), 0 12px 32px rgba(15, 23, 42, 0.06)"
        },
    }
}
# Plotly gives its drag handles these class names. The Play CDN scanned the live page and
# emitted matching utilities; keep them so the generated rules stay a superset of its output.
SAFELIST = [
    f"cursor-{name}"
    for name in (
        "crosshair",
        "move",
        "e-resize",
        "ew-resize",
        "n-resize",
        "ne-resize",
        "ns-resize",
        "nw-resize",
        "s-resize",
        "se-resize",
        "sw-resize",
        "w-resize",
    )
]


def template_without_css(text):
    """Template text with the generated CSS removed, so builds do not scan their own output."""
    start = text.index(START) + len(START)
    end = text.index(END, start)
    return text[:start] + "\n" + text[end:]


def compact(css):
    """Drop comments and indentation; never merge or reorder rules."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = " ".join(line.strip() for line in css.splitlines() if line.strip())
    return re.sub(r"\s*([{};])\s*", r"\1", css)


def build_css(template_text):
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError("npx (Node.js) is required to build the report CSS")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        (work / "content.html").write_text(template_without_css(template_text), encoding="utf-8")
        config = {"content": ["./content.html"], "theme": THEME, "safelist": SAFELIST}
        (work / "tailwind.config.js").write_text(f"module.exports = {json.dumps(config)};\n")
        (work / "input.css").write_text(
            "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n"
        )
        subprocess.run(
            [npx, "--yes", f"tailwindcss@{TAILWIND_VERSION}", "-c", "tailwind.config.js",
             "-i", "input.css", "-o", "output.css"],
            cwd=work, check=True, capture_output=True, text=True,
        )  # fmt: skip
        return compact((work / "output.css").read_text(encoding="utf-8"))


def render(template_text, css):
    start = template_text.index(START) + len(START)
    end = template_text.index(END, start)
    return template_text[:start] + "\n" + css + "\n    " + template_text[end:]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Fail if the inlined CSS is stale")
    args = parser.parse_args()
    text = TEMPLATE.read_text(encoding="utf-8")
    updated = render(text, build_css(text))
    if args.check:
        if updated != text:
            print(f"{TEMPLATE} has stale CSS; run {Path(__file__).name}", file=sys.stderr)
            return 1
        print("Report CSS is up to date")
        return 0
    if updated != text:
        TEMPLATE.write_text(updated, encoding="utf-8")
        print(f"Updated the Tailwind CSS in {TEMPLATE}")
    else:
        print("Report CSS is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
