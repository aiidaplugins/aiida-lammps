"""Tests for the :mod:`aiida_lammps.parsers.raw` module."""
# pylint: disable=redefined-outer-name
from aiida.plugins import ParserFactory


def test_default(generate_calc_job_node, data_regression):
    """Test parsing a default output case."""
    node = generate_calc_job_node("lammps.raw", "default")
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})


def test_alt_timing_info(generate_calc_job_node, data_regression):
    """Test parsing an alt output case."""
    node = generate_calc_job_node("lammps.raw", "alt")
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})
