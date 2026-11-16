from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from urllib.parse import urlsplit

project = "gsMap"
copyright = "2024-2026, JianYang-Lab"
author = "Liyang, Wenhao"
try:
    release = _distribution_version("gsMap")
except PackageNotFoundError:  # building from a checkout without installing gsMap
    release = "2.x"
version = release

# Downloads, tutorial commands and report previews share this resource root.
# Change this value or pass
# -D gsmap_resource_base_url=https://your-server.example/data/gsMap to Sphinx.
gsmap_resource_base_url = "https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1"
# Existing source pages use this prefix; rewrite it when building the docs.
_source_resource_base_url = "https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1"


def _configure_resource_url(app, config):
    from sphinx.errors import ConfigError

    value = config.gsmap_resource_base_url
    if not isinstance(value, str):
        raise ConfigError("gsmap_resource_base_url must be an absolute HTTP(S) URL")
    value = value.rstrip("/")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.port != 0
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(
                character.isspace() or character in "\\\"'<>`;$&|(){}*" for character in value
            )
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError(
            "gsmap_resource_base_url must be an absolute HTTP(S) URL "
            "without credentials, whitespace, a query or a fragment"
        )
    config.gsmap_resource_base_url = value


def _replace_resource_urls(app, docname, source):
    # Rewrite before parsing so literal URLs inside shell code blocks and
    # Markdown/RST links are updated together. Other domains stay unchanged.
    source[0] = source[0].replace(
        f"{_source_resource_base_url}/", f"{app.config.gsmap_resource_base_url}/"
    )


def setup(app):
    app.add_config_value("gsmap_resource_base_url", gsmap_resource_base_url, "env", types=[str])
    app.connect("config-inited", _configure_resource_url)
    app.connect("source-read", _replace_resource_urls)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "nbsphinx",
    "myst_parser",
    # "sphinx_charts.charts",
    "sphinxcontrib.jquery",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'classic'
# html_theme = 'sphinx_rtd_theme'
# html_theme = "pydata_sphinx_theme"
html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_show_sourcelink = False
html_css_files = ["custom.css"]

# One Mermaid theme for all documentation diagrams. Semantic node classes in
# individual flowcharts distinguish work, buffers and data, not page themes.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
mermaid_height = "auto"
mermaid_fullscreen = True
mermaid_fullscreen_button = "Expand"
mermaid_init_config = {
    "startOnLoad": False,
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Arial, sans-serif",
        "fontSize": "14px",
        "primaryColor": "#edf4fc",
        "primaryTextColor": "#19324d",
        "primaryBorderColor": "#517ba5",
        "secondaryColor": "#fff4df",
        "tertiaryColor": "#eef2f6",
        "lineColor": "#64748b",
        "textColor": "#19324d",
        "clusterBkg": "#f8fafc",
        "clusterBorder": "#cbd5e1",
        "titleColor": "#19324d",
        "edgeLabelBackground": "#ffffff",
    },
    "flowchart": {
        "wrappingWidth": 320,
        "nodeSpacing": 20,
        "rankSpacing": 28,
        "subGraphTitleMargin": {"top": 8, "bottom": 12},
    },
}

html_theme_options = {
    # "light_css_variables": {
    #     "color-brand-primary": "#7C4DFF",
    #     "color-brand-content": "#7C4DFF",
    #     "color-code-background": "#f5f5f5",
    # },
}

# add plotly.js to the build
# html_js_files = [
#     "https://cdn.plot.ly/plotly-latest.min.js",
# ]
