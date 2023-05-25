"""Tests for the :mod:`aiida_lammps.parsers.raw` module."""
# pylint: disable=redefined-outer-name
from aiida.orm import SinglefileData
from aiida.plugins import ParserFactory

from aiida_lammps.calculations.raw import LammpsRawCalculation


def test_default(generate_calc_job_node, data_regression):
    """Test parsing a default output case."""
    node = generate_calc_job_node("lammps.raw", "default")
    parser = ParserFactory("lammps.raw")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    data_regression.check({"results": results["results"].get_dict()})
