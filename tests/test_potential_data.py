"""Test the functionality of the lammps potential data object"""
import os

import pytest
import yaml

from aiida_lammps.data.potential import LammpsPotentialData
from aiida_lammps.parsers import inputfile
from .utils import TEST_DIR


@pytest.mark.parametrize(
    "potential_type",
    ["tersoff", "eam_alloy", "meam", "morse"],
)
def test_lammps_potentials_init(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the LAMMPS potential data type."""

    potential_information = get_lammps_potential_data(potential_type)
    node = LammpsPotentialData.get_or_create(
        source=potential_information["filename"],
        filename=potential_information["filename"],
        **potential_information["parameters"],
    )

    reference_file = os.path.join(
        TEST_DIR,
        "test_lammps_potential_data",
        f"test_init_{potential_type}.yaml",
    )

    with open(reference_file) as handler:
        reference_values = yaml.load(handler, yaml.SafeLoader)

    _attributes = ["md5", "pair_style", "species", "atom_style", "default_units"]

    for _attribute in _attributes:
        _msg = f'attribute "{_attribute}" does not match between reference and current value'
        assert reference_values[_attribute] == node.base.attributes.get(
            _attribute
        ), _msg


@pytest.mark.parametrize(
    "potential_type",
    ["tersoff", "eam_alloy", "meam", "morse"],
)
def test_lammps_potentials_files(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the LAMMPS potential data type."""

    potential_information = get_lammps_potential_data(potential_type)
    node = LammpsPotentialData.get_or_create(
        source=potential_information["filename"],
        filename=potential_information["filename"],
        **potential_information["parameters"],
    )

    _msg = "content of the files differ"
    assert node.get_content().split("\n") == potential_information[
        "potential_data"
    ].split("\n"), _msg


@pytest.mark.parametrize(
    "potential_type",
    ["tersoff", "eam_alloy", "meam", "morse"],
)
def test_lammps_potentials_input_block(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the LAMMPS potential data type."""

    potential_information = get_lammps_potential_data(potential_type)
    node = LammpsPotentialData.get_or_create(
        source=potential_information["filename"],
        filename=potential_information["filename"],
        **potential_information["parameters"],
    )

    potential_block = inputfile.write_potential_block(
        parameters_potential={},
        potential_file="potential.dat",
        potential=node,
        structure=potential_information["structure"],
    )

    reference_file = os.path.join(
        TEST_DIR,
        "test_lammps_potential_data",
        f"test_init_{potential_type}_block.txt",
    )

    with open(reference_file) as handler:
        reference_value = handler.read()

    assert potential_block == reference_value, "content of the potential blocks differ"
