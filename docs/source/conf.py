# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "GFloat"
copyright = "2023, Andrew Fitzgibbon"
author = "Andrew Fitzgibbon"
release = "0.0.4"
version = "0.0.4"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "member-order": "bysource",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
