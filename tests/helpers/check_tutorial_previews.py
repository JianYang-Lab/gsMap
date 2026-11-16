"""Follow deployed tutorial preview links and check that each report renders."""

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright


class PreviewLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, data):
        if self.href:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            if re.search(r"(report preview|view report)", "".join(self.label), re.I):
                self.targets.append(self.href)
            self.href = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_urls = set()
    records = []
    failure = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            for tutorial in (
                "2d_tutorial/10x",
                "2d_tutorial/mouse_embryo",
                "3d_tutorial/human_gastrulation",
                "3d_tutorial/mouse_e11",
            ):
                url = f"{args.docs_url.rstrip('/')}/{tutorial}.html"
                links = PreviewLinks()
                with urlopen(
                    Request(url, headers={"User-Agent": "gsMap-docs-check"}), timeout=60
                ) as response:
                    assert response.status == 200, url
                    links.feed(response.read().decode("utf-8"))
                targets = [urljoin(url, link) for link in links.targets]
                assert targets, f"No preview links on {url}"
                report_urls.update(targets)
                records.append({"tutorial": url, "preview_urls": targets})
            assert len(report_urls) == 6, report_urls
            for index, url in enumerate(sorted(report_urls)):
                page = browser.new_page(viewport={"width": 1500, "height": 1100})
                errors, bad_responses = [], []
                page.on("pageerror", lambda error, items=errors: items.append(str(error)))
                page.on(
                    "response",
                    lambda response, items=bad_responses: (
                        items.append({"url": response.url, "status": response.status})
                        if response.status >= 400 and not response.url.endswith("favicon.ico")
                        else None
                    ),
                )
                record = {
                    "report": url,
                    "errors": errors,
                    "bad_responses": bad_responses,
                    "rendered": False,
                }
                records.append(record)
                response = page.goto(url, wait_until="networkidle", timeout=90000)
                assert response.status == 200, url
                # Check visible plots, not framework-private application state:
                # deployed report versions need not expose an identical Alpine proxy.
                page.wait_for_function(
                    """() => [...document.querySelectorAll(
                        'canvas, .js-plotly-plot .main-svg'
                    )].some(node => {
                        const rect = node.getBoundingClientRect();
                        return rect.width > 0 && rect.height > 0
                            && getComputedStyle(node).visibility !== 'hidden';
                    })""",
                    timeout=60000,
                )
                record.update(title=page.title(), rendered=True)
                page.screenshot(path=str(args.output / f"preview_{index}.png"))
                assert not errors, record
                assert not bad_responses, record
                print(json.dumps(record), flush=True)
                page.close()
            browser.close()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "browser.json").write_text(
            json.dumps(
                {"passed": failure is None, "failure": failure, "checks": records}, indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
