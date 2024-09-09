"""
Tests to aiida-lammps parsers.
"""

import os

from aiida.plugins import ParserFactory

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


def test_base_parser_error(generate_calc_job_node, data_regression, fixture_localhost):
    """Test the parser when an error is found in the output file."""
    node = generate_calc_job_node(
        computer=fixture_localhost,
        entry_point_name="lammps.base",
        test_name="parsers/fixtures/base/error",
    )
    parser = ParserFactory("lammps.base")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)
    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.exit_status == 309, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})
