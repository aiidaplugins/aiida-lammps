"""Configuration for Sphinx documentation build.

It is recommended to use tox to run the build (see tox.ini):
`tox -e docs-clean` and `tox -e docs-update`,
or directly: `sphinx-build -n -W --keep-going docs/source docs/_build`
"""
from aiida_lammps import __version__

PROJECT = 'AiiDA LAMMPS'
COPYRIGHT = '2021, AiiDA Team'
AUTHOR = 'AiiDA Team'
VERSION = __version__

extensions = [
    # read Markdown files
    'myst_parser',
    # specify sitemap in single file (not toctrees)
    'sphinx_external_toc',
]

HTML_THEME = 'furo'
HTML_TITLE = f'v{__version__}'
HTML_LOGO = 'static/logo.png'
html_theme_options = {
    'announcement': 'This documentation is in development!',
}
