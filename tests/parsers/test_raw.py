"""Tests for the :mod:`aiida_lammps.parsers.raw` module."""

# pylint: disable=redefined-outer-name
from aiida.plugins import ParserFactory


def test_default(generate_calc_job_node, data_regression, fixture_localhost):
    """Test parsing a default output case."""
    node = generate_calc_job_node(
        computer=fixture_localhost,
        entry_point_name="lammps.raw",
        test_name="parsers/fixtures/raw/default",
    )
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})

def test_double_thermo_style(generate_calc_job_node, data_regression, fixture_localhost):
    """Test parsing a double thermo_style output case."""
    node = generate_calc_job_node(
        computer=fixture_localhost,
        entry_point_name="lammps.raw",
        test_name="parsers/fixtures/raw/thermo_style",
    )
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})


def test_alt_timing_info(generate_calc_job_node, data_regression, fixture_localhost):
    """Test parsing an alt output case."""
    node = generate_calc_job_node(
        computer=fixture_localhost,
        entry_point_name="lammps.raw",
        test_name="parsers/fixtures/raw/alt",
    )
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})


def test_raw_parser_error(generate_calc_job_node, data_regression, fixture_localhost):
    """Test the parser when an error is found in the output file."""
    node = generate_calc_job_node(
        computer=fixture_localhost,
        entry_point_name="lammps.raw",
        test_name="parsers/fixtures/raw/error",
    )
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)
    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.exit_status == 309, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})
