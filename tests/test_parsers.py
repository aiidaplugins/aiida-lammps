"""
Tests to aiida-lammps parsers.
"""
import os

from aiida_lammps.parsers.parse_raw import parse_final_data, parse_outputfile
from .utils import TEST_DIR


def test_parser_out(data_regression):
    """
    Test the parser for the ``lammps.out`` file.
    """
    filename = os.path.join(
        TEST_DIR,
        "input_files",
        "parsers",
        "lammps.out",
    )

    parsed_data = parse_outputfile(filename=filename)
    data_regression.check(parsed_data)


def test_parse_final_variables():
    """
    Test the parser for the final variables
    """
    filename = os.path.join(
        TEST_DIR,
        "input_files",
        "parsers",
        "aiida_lammps.yaml",
    )

    parsed_data = parse_final_data(filename=filename)

    assert isinstance(parsed_data, dict), "the parsed data is not of the correct format"

    assert "final_step" in parsed_data, "no step information present"
    assert "final_etotal" in parsed_data, "no total energy information present"
