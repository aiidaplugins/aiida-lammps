import pytest
import six
from aiida_lammps.common.generate_structure import generate_lammps_structure


@pytest.mark.parametrize('structure', [
    "Fe",
    "pyrite",
    "fes_cubic-zincblende",
    "greigite"
])
def test_generate(db_test_app, get_structure_data, structure, file_regression):
    structure = get_structure_data(structure)
    file_regression.check(six.ensure_text(
        generate_lammps_structure(structure, round_dp=8)))
