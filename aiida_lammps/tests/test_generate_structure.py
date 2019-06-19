import pytest
from aiida_lammps.common.generate_structure import generate_lammps_structure


@pytest.mark.parametrize('structure', [
    "pyrite",
    "fes_cubic-zincblende"
])
def test_generate(db_test_app, get_structure_data, structure, file_regression):
    structure = get_structure_data(structure)
    file_regression.check(generate_lammps_structure(structure))
