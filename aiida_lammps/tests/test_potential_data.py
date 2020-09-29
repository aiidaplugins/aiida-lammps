import pytest

from aiida_lammps.data.potential import EmpiricalPotential


def test_list_potentials():
    assert set(EmpiricalPotential.list_types()).issuperset(
        ["eam", "lennard_jones", "reaxff", "tersoff"]
    )


def test_load_type():
    EmpiricalPotential.load_type("eam")


@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_init(db_test_app, get_potential_data, potential_type, data_regression):
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(type=potential.type, data=potential.data)
    data_regression.check(node.attributes)


@pytest.mark.parametrize("potential_type", ["tersoff"])
def test_potential_files(
    db_test_app, get_potential_data, potential_type, file_regression
):
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(type=potential.type, data=potential.data)
    file_regression.check(node.get_object_content("potential.pot", "r"))


@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_input_lines(db_test_app, get_potential_data, potential_type, file_regression):
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(type=potential.type, data=potential.data)
    file_regression.check(node.get_input_lines())
