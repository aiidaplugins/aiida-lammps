"""Test the functionality of the lammps potential data object"""

import pytest
from aiida_lammps.data.potential import EmpiricalPotential


def test_list_potentials():
    """Test that all the supported potential types are recognized."""
    assert set(EmpiricalPotential.list_types()).issuperset(
        ['eam', 'lennard_jones', 'reaxff', 'tersoff'])


def test_load_type():
    """Test that an specific potential can be loaded"""
    EmpiricalPotential.load_type('eam')


@pytest.mark.parametrize('potential_type',
                         ['lennard-jones', 'tersoff', 'eam', 'reaxff'])
def test_init(
    db_test_app,  # pylint: disable=unused-argument
    get_potential_data,
    potential_type,
    data_regression,
):
    """Test that the potential can be generated"""
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(type=potential.type, data=potential.data)  # pylint: disable=no-value-for-parameter
    data_regression.check(node.attributes)


@pytest.mark.parametrize('potential_type', ['tersoff'])
def test_potential_files(
    db_test_app,  # pylint: disable=unused-argument
    get_potential_data,
    potential_type,
    file_regression,
):
    """Test that one can read the potential content."""
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(type=potential.type, data=potential.data)  # pylint: disable=no-value-for-parameter
    file_regression.check(node.get_object_content('potential.pot', 'r'))


@pytest.mark.parametrize('potential_type',
                         ['lennard-jones', 'tersoff', 'eam', 'reaxff'])
def test_input_lines(
    db_test_app,  # pylint: disable=unused-argument
    get_potential_data,
    potential_type,
    file_regression,
):
    """Test that one can get the potential lines for a given aiida-lammps potential"""
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(type=potential.type, data=potential.data)  # pylint: disable=no-value-for-parameter
    file_regression.check(node.get_input_lines())
