# Configuration file for the Sphinx documentation builder.

project = "NVision"
copyright = "2026, NVision Team"
author = "NVision Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
static_path = ["_static"]
