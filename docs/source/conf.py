"""Configuration for Sphinx documentation build.

It is recommended to use tox to run the build (see tox.ini):
`tox -e docs-clean` and `tox -e docs-update`,
or directly: `sphinx-build -n -W --keep-going docs/source docs/_build`
"""
from aiida_lammps import __version__

project = "AiiDA LAMMPS"
copyright = "2021, AiiDA Team"
author = "AiiDA Team"
version = __version__

extensions = [
    # read Markdown files
    "myst_parser",
    # specify sitemap in single file (not toctrees)
    "sphinx_external_toc",
]

html_theme = "furo"
html_title = f"v{__version__}"
html_logo = "static/logo.png"
html_theme_options = {
    "announcement": "This documentation is in development!",
}
