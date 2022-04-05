"""Validate consistency of versions and dependencies.

Validates consistency of setup.json and

 * environment.yml
 * version in aiida_lammps/__init__.py
"""
import json
import os
import sys

import click

FILENAME_SETUP_JSON = "setup.json"
SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]
ROOT_DIR = os.path.join(SCRIPT_PATH, os.pardir)
FILEPATH_SETUP_JSON = os.path.join(ROOT_DIR, FILENAME_SETUP_JSON)


def get_setup_json():
    """Return the `setup.json` as a python dictionary."""
    with open(FILEPATH_SETUP_JSON, "r") as handle:
        setup_json = json.load(handle)  # , object_pairs_hook=OrderedDict)

    return setup_json


@click.group()
def cli():
    """Command line interface for pre-commit checks."""
    pass


@cli.command("version")
def validate_version():
    """Check that version numbers match.

    Check version number in setup.json and aiida_lammos/__init__.py and make sure
    they match.
    """
    # Get version from python package
    sys.path.insert(0, ROOT_DIR)
    import aiida_lammps  # pylint: disable=wrong-import-position

    version = aiida_lammps.__version__

    setup_content = get_setup_json()
    if version != setup_content["version"]:
        click.echo("Version number mismatch detected:")
        click.echo(
            "Version number in '{}': {}".format(
                FILENAME_SETUP_JSON, setup_content["version"]
            )
        )
        click.echo(
            "Version number in '{}/__init__.py': {}".format("aiida_lammps", version)
        )
        click.echo(
            "Updating version in '{}' to: {}".format(FILENAME_SETUP_JSON, version)
        )

        setup_content["version"] = version
        with open(FILEPATH_SETUP_JSON, "w") as fil:
            # Write with indentation of two spaces and explicitly define separators to not have spaces at end of lines
            json.dump(setup_content, fil, indent=2, separators=(",", ": "))

        sys.exit(1)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
