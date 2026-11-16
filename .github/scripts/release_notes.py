"""Build GitHub release notes for a version tag from CHANGELOG.md.

The CHANGELOG section for the tagged version is required: the release workflow
runs ``--check`` before anything is published, so a release cannot ship without
curated notes.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

DOCS_INSTALL_URL = (
    "https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/install.html"
)


def changelog_section(text, version):
    """Return the body of the ``## <version>`` section, without its heading."""
    heading = re.compile(rf"^## v?{re.escape(version)}(?:\s.*)?$")
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if heading.match(line.strip()):
            break
    else:
        raise ValueError(f"CHANGELOG.md has no '## {version}' section")
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    section = "\n".join(body).strip()
    if not section:
        raise ValueError(f"CHANGELOG.md section '## {version}' is empty")
    return section


def previous_tag(tag, tags):
    """Highest public ``v*`` tag below ``tag``, ignoring non-PEP 440 tags."""
    current = Version(tag[1:])
    earlier = []
    for candidate in tags:
        if not candidate.startswith("v"):
            continue
        try:
            parsed = Version(candidate[1:])
        except InvalidVersion:
            continue
        if parsed < current and parsed.local is None:
            earlier.append((parsed, candidate))
    return max(earlier)[1] if earlier else None


def release_notes(tag, prerelease, repository, section, previous=None):
    version = tag[1:]
    image = f"ghcr.io/{repository.lower()}"
    lines = [
        f"# gsMap {tag}",
        "",
        "## 📦 Installation",
        "",
        "Choose one compute backend:",
        "",
        "```bash",
        f'uv pip install "gsMap[cpu]=={version}" --torch-backend=cpu',
        f'uv pip install "gsMap[cuda128]=={version}" --torch-backend=cu128',
        f'uv pip install "gsMap[cuda130]=={version}" --torch-backend=cu130',
        "```",
        "",
        f"See the [installation guide]({DOCS_INSTALL_URL}) for other backends.",
        "",
    ]
    if prerelease:
        lines += [
            "This is a pre-release for testing. `pip install gsMap` without a version does not "
            "select it, and stable Docker tags are unchanged.",
            "",
        ]
    else:
        lines += ["### Docker", "", "```bash", f"docker pull {image}:{version}", "```", ""]
    lines += ["## 📋 What's Changed", "", section, ""]
    if previous:
        lines += [
            f"**Full changelog:** https://github.com/{repository}/compare/{previous}...{tag}",
            "",
        ]
    return "\n".join(lines)


def git_tags():
    result = subprocess.run(
        ["git", "tag", "--list", "v*"], check=True, capture_output=True, text=True
    )
    return result.stdout.split()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, e.g. v2.0.0b1")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--check", action="store_true", help="Only validate the section")
    parser.add_argument("--prerelease", choices=["true", "false"], default="false")
    parser.add_argument("--repository", default="JianYang-Lab/gsMap")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    section = changelog_section(args.changelog.read_text(), args.tag[1:])
    if args.check:
        print(f"CHANGELOG.md has release notes for {args.tag}")
        return
    notes = release_notes(
        args.tag,
        args.prerelease == "true",
        args.repository,
        section,
        previous_tag(args.tag, git_tags()),
    )
    if args.output:
        args.output.write_text(notes)
    else:
        sys.stdout.write(notes)


if __name__ == "__main__":
    main()
