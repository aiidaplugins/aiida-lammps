"""Test the structure generation in aiida-lammps"""
from aiida_lammps.parsers.utils import generate_lammps_structure
import pytest


@pytest.mark.parametrize(
    "structure",
    ["Fe", "pyrite", "fes_cubic-zincblende", "greigite"],
)
def test_generate(db_test_app, get_structure_data, structure, file_regression):  # pylint: disable=unused-argument
    """Test the structure generation in aiida-lammps"""
    structure = get_structure_data(structure)
    text, _ = generate_lammps_structure(structure, round_dp=8)
    file_regression.check(text)
