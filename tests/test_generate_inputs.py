"""Functions to tests the input file generation"""

import io
import os

import pytest
import yaml

from aiida_lammps.common import input_generator
from aiida_lammps.data.lammps_potential import LammpsPotentialData
from .utils import TEST_DIR


@pytest.mark.parametrize(
    "potential_type",
    ["eam_alloy"],
)
def test_input_generate_minimize(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the generation of the input file for minimize calculations"""
    # pylint: disable=too-many-locals

    parameters_file = os.path.join(
        TEST_DIR,
        "test_generate_inputs",
        "parameters_minimize.yaml",
    )
    # Dictionary with parameters for controlling aiida-lammps
    with open(parameters_file) as handler:
        parameters = yaml.load(handler, yaml.SafeLoader)

    input_generator.validate_input_parameters(parameters)
    # Generate the potential
    potential_information = get_lammps_potential_data(potential_type)
    potential = LammpsPotentialData.get_or_create(
        source=potential_information["filename"],
        filename=potential_information["filename"],
        **potential_information["parameters"],
    )
    # Generating the structure
    structure = potential_information["structure"]
    # Generating the input file
    input_file = input_generator.generate_input_file(
        parameters=parameters,
        potential=potential,
        structure=structure,
        trajectory_filename="temp.dump",
        restart_filename="restart.aiida",
        potential_filename="potential.dat",
        structure_filename="structure.dat",
    )
    reference_file = os.path.join(
        TEST_DIR,
        "test_generate_inputs",
        f"test_generate_input_{potential_type}_minimize.txt",
    )

    with io.open(reference_file, "r") as handler:
        reference_value = handler.read()

    assert input_file == reference_value, "the content of the files differ"


@pytest.mark.parametrize(
    "potential_type",
    ["eam_alloy"],
)
def test_input_generate_md(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the generation of the input file for MD calculations"""
    # pylint: disable=too-many-locals

    parameters_file = os.path.join(
        TEST_DIR,
        "test_generate_inputs",
        "parameters_md.yaml",
    )
    # Dictionary with parameters for controlling aiida-lammps
    with open(parameters_file) as handler:
        parameters = yaml.load(handler, yaml.SafeLoader)

    input_generator.validate_input_parameters(parameters)
    # Generate the potential
    potential_information = get_lammps_potential_data(potential_type)
    potential = LammpsPotentialData.get_or_create(
        source=potential_information["filename"],
        filename=potential_information["filename"],
        **potential_information["parameters"],
    )
    # Generating the structure
    structure = potential_information["structure"]
    # Generating the input file
    input_file = input_generator.generate_input_file(
        parameters=parameters,
        potential=potential,
        structure=structure,
        trajectory_filename="temp.dump",
        restart_filename="restart.aiida",
        potential_filename="potential.dat",
        structure_filename="structure.dat",
    )
    reference_file = os.path.join(
        TEST_DIR,
        "test_generate_inputs",
        f"test_generate_input_{potential_type}_md.txt",
    )
    with io.open(reference_file, "r") as handler:
        reference_value = handler.read()

    assert input_file == reference_value, "the content of the files differ"
