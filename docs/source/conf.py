project = "gsMap"
copyright = "2024, Liyang, Wenhao"
author = "Liyang, Wenhao"
# release = gsMap.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
    "nbsphinx",
    "myst_parser",
    # "sphinx_charts.charts",
    "sphinxcontrib.jquery",
    "sphinx_inline_tabs",
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

html_theme_options = {
    # Shown at the top of every page of the 1.x documentation.
    "announcement": (
        "<strong>gsMap 1.x is deprecated</strong> and receives bug fixes only; pin it with "
        "<code>pip install \"gsMap&lt;2\"</code>. For new analyses, see the "
        '<a href="https://jianyang-lab.github.io/gsMap/">gsMap 2.x documentation</a>.'
    ),
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
