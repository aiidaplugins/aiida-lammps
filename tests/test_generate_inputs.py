"""Functions to tests the input file generation"""
import os

import pytest

from aiida_lammps.data.potential import LammpsPotentialData
from aiida_lammps.parsers import inputfile
from .utils import TEST_DIR


@pytest.mark.parametrize(
    "potential_type",
    ["eam_alloy"],
)
def test_input_generate_minimize(
    db_test_app,  # pylint: disable=unused-argument
    parameters_minimize,
    get_lammps_potential_data,
    potential_type,
    file_regression,
):
    """Test the generation of the input file for minimize calculations"""
    # pylint: disable=too-many-locals

    inputfile.validate_input_parameters(parameters_minimize)
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
    input_file = inputfile.generate_input_file(
        parameters=parameters_minimize,
        potential=potential,
        structure=structure,
        trajectory_filename="temp.dump",
        restart_filename="restart.aiida",
        potential_filename="potential.dat",
        structure_filename="structure.dat",
    )

    file_regression.check(input_file)


@pytest.mark.parametrize(
    "potential_type,restart",
    [
        ("eam_alloy", None),
        ("eam_alloy", "input_aiida_lammps.restart"),
    ],
)
def test_input_generate_md(
    db_test_app,  # pylint: disable=unused-argument
    parameters_md_npt,
    get_lammps_potential_data,
    potential_type,
    restart,
    file_regression,
):
    """Test the generation of the input file for MD calculations"""
    # pylint: disable=too-many-locals

    inputfile.validate_input_parameters(parameters_md_npt)
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
    input_file = inputfile.generate_input_file(
        parameters=parameters_md_npt,
        potential=potential,
        structure=structure,
        trajectory_filename="temp.dump",
        restart_filename="restart.aiida",
        potential_filename="potential.dat",
        structure_filename="structure.dat",
        read_restart_filename=restart,
    )

    file_regression.check(input_file)


@pytest.mark.parametrize(
    "print_final,print_intermediate,num_steps",
    [
        (True, False, None),
        (True, True, 100),
        (True, True, None),
        (False, True, None),
        (False, True, 100),
    ],
)
def test_input_generate_restart(
    db_test_app,  # pylint: disable=unused-argument
    parameters_md_npt,
    print_final,
    print_intermediate,
    num_steps,
    data_regression,
):
    """Test the generation of the input file for MD calculations"""
    # pylint: disable=too-many-locals, too-many-arguments

    if "restart" not in parameters_md_npt:
        parameters_md_npt["restart"] = {}
    parameters_md_npt["restart"]["print_final"] = print_final
    parameters_md_npt["restart"]["print_intermediate"] = print_intermediate
    if num_steps:
        parameters_md_npt["restart"]["num_steps"] = num_steps

    inputfile.validate_input_parameters(parameters_md_npt)

    # Generating the input file
    input_file = inputfile.write_restart_block(
        parameters_restart=parameters_md_npt["restart"],
        restart_filename="restart.aiida",
        max_number_steps=1000,
    )

    data_regression.check(input_file)


#    if print_final:
#        assert "final" in input_file, "no final restart information generated"
#        _msg = "reference value for the final restart does not match"
#        assert input_file["final"] == restart_data["final"], _msg
#    else:
#        assert input_file["final"] == ""
#
#    if print_intermediate:
#        assert (
#            "intermediate" in input_file
#        ), "no intermediate restart information generated"
#        _msg = "reference value for the intermediate restart does not match"
#        assert input_file["intermediate"] == restart_data["intermediate"], _msg
#    else:
#        assert input_file["intermediate"] == ""
