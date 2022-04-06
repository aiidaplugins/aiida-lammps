"""Configuration for Sphinx documentation build.

It is recommended to use tox to run the build (see tox.ini):
`tox -e docs-clean` and `tox -e docs-update`,
or directly: `sphinx-build -n -W --keep-going docs/source docs/_build`
"""
import os
import subprocess
import sys

from aiida.manage.configuration import load_documentation_profile

from aiida_lammps import __version__

# -- AiiDA-related setup --------------------------------------------------

# Load the dummy profile even if we are running locally, this way the
# documentation will succeed even if the current
# default profile of the AiiDA installation does not use a Django backend.
load_documentation_profile()

PROJECT = "AiiDA LAMMPS"
COPYRIGHT = "2021, AiiDA Team"
AUTHOR = "AiiDA Team"
VERSION = __version__

extensions = [
    # read Markdown files
    "myst_parser",
    # specify sitemap in single file (not toctrees)
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "aiida": ("https://aiida-core.readthedocs.io/en/latest", None),
    "click": ("https://click.palletsprojects.com/", None),
    "flask": ("http://flask.pocoo.org/docs/latest/", None),
    "flask_restful": ("https://flask-restful.readthedocs.io/en/latest/", None),
    "kiwipy": ("https://kiwipy.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "plumpy": ("https://plumpy.readthedocs.io/en/latest/", None),
    "pymatgen": ("https://pymatgen.org/", None),
}

suppress_warnings = ["etoc.toctree"]

html_theme = "furo"  # pylint: disable=invalid-name
html_title = f"v{__version__}"  # pylint: disable=invalid-name
html_logo = "static/logo.png"  # pylint: disable=invalid-name
html_theme_options = {
    "announcement": "This documentation is in development!",
}


def run_apidoc(_):
    """Runs sphinx-apidoc when building the documentation.

    Needs to be done in conf.py in order to include the APIdoc in the
    build on readthedocs.

    See also https://github.com/rtfd/readthedocs.org/issues/1139
    """
    source_dir = os.path.abspath(os.path.dirname(__file__))
    apidoc_dir = os.path.join(source_dir, "reference", "apidoc")
    package_dir = os.path.join(source_dir, os.pardir, os.pardir, "aiida_lammps")

    # In #1139, they suggest the route below, but this ended up
    # calling sphinx-build, not sphinx-apidoc
    # from sphinx.apidoc import main
    # main([None, '-e', '-o', apidoc_dir, package_dir, '--force'])

    cmd_path = "sphinx-apidoc"
    if hasattr(sys, "real_prefix"):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(
            os.path.join(
                sys.prefix,
                "bin",
                "sphinx-apidoc",
            )
        )

    options = [
        "-o",
        apidoc_dir,
        package_dir,
        "--private",
        "--force",
        "--no-headings",
        "--module-first",
        "--no-toc",
        "--maxdepth",
        "4",
    ]

    # See https://stackoverflow.com/a/30144019
    env = os.environ.copy()
    env[
        "SPHINX_APIDOC_OPTIONS"
    ] = "members,special-members,private-members,undoc-members,show-inheritance"
    subprocess.check_call([cmd_path] + options, env=env)


def setup(app):
    """Run the apidoc."""
    if os.environ.get("RUN_APIDOC", None) != "False":
        app.connect("builder-inited", run_apidoc)


# We should ignore any python built-in exception, for instance
# Warnings to ignore when using the -n (nitpicky) option
nitpicky = True
with open("nitpick-exceptions") as handle:
    nitpick_ignore = [
        tuple(line.strip().split(None, 1))
        for line in handle.readlines()
        if line.strip() and not line.startswith("#")
    ]
