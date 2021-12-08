"""Test the functionality of the lammps potential data object"""

import io
import os

import pytest
import yaml
from aiida_lammps.tests.utils import TEST_DIR
from aiida_lammps.data.potential import EmpiricalPotential
from aiida_lammps.data.lammps_potential import LammpsPotentialData
from aiida_lammps.common import input_generator


def test_list_potentials():
    """Test that all the supported potential types are recognized."""
    assert set(EmpiricalPotential.list_types()).issuperset(
        ['eam', 'lennard_jones', 'reaxff', 'tersoff'])


def test_load_type():
    """Test that an specific potential can be loaded"""
    EmpiricalPotential.load_type('eam')


@pytest.mark.parametrize(
    'potential_type',
    ['lennard-jones', 'tersoff', 'eam', 'reaxff'],
)
def test_init(
    db_test_app,  # pylint: disable=unused-argument
    get_potential_data,
    potential_type,
    data_regression,
):
    """Test that the potential can be generated"""
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(
        potential_type=potential.type,
        data=potential.data,
    )
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
    node = EmpiricalPotential(
        potential_type=potential.type,
        data=potential.data,
    )
    file_regression.check(node.get_object_content('potential.pot', 'r'))


@pytest.mark.parametrize(
    'potential_type',
    ['lennard-jones', 'tersoff', 'eam', 'reaxff'],
)
def test_input_lines(
    db_test_app,  # pylint: disable=unused-argument
    get_potential_data,
    potential_type,
    file_regression,
):
    """Test that one can get the potential lines for a given aiida-lammps potential"""
    potential = get_potential_data(potential_type)
    node = EmpiricalPotential(
        potential_type=potential.type,
        data=potential.data,
    )
    file_regression.check(node.get_input_lines())


@pytest.mark.parametrize(
    'potential_type',
    ['tersoff', 'eam_alloy', 'meam', 'morse'],
)
def test_lammps_potentials_init(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the LAMMPS potential data type."""

    potential_information = get_lammps_potential_data(
        potential_type.replace('_', '/'))
    node = LammpsPotentialData.get_or_create(
        source=potential_information['filename'],
        filename=potential_information['filename'],
        **potential_information['parameters'],
    )

    reference_file = os.path.join(
        TEST_DIR,
        'test_lammps_potential_data',
        f'test_init_{potential_type}.yaml',
    )

    with io.open(reference_file, 'r') as handler:
        reference_values = yaml.load(handler, yaml.SafeLoader)

    _attributes = [
        'md5', 'pair_style', 'species', 'atom_style', 'default_units'
    ]

    for _attribute in _attributes:
        _msg = f'attribute "{_attribute}" does not match between reference and current value'
        assert reference_values[_attribute] == node.get_attribute(
            _attribute), _msg


@pytest.mark.parametrize(
    'potential_type',
    ['tersoff', 'eam_alloy', 'meam', 'morse'],
)
def test_lammps_potentials_files(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the LAMMPS potential data type."""

    potential_information = get_lammps_potential_data(
        potential_type.replace('_', '/'))
    node = LammpsPotentialData.get_or_create(
        source=potential_information['filename'],
        filename=potential_information['filename'],
        **potential_information['parameters'],
    )

    _msg = 'content of the files differ'
    assert node.get_content().split(
        '\n') == potential_information['potential_data'].split('\n'), _msg


@pytest.mark.parametrize(
    'potential_type',
    ['tersoff', 'eam_alloy', 'meam', 'morse'],
)
def test_lammps_potentials_input_block(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the LAMMPS potential data type."""

    potential_information = get_lammps_potential_data(
        potential_type.replace('_', '/'))
    node = LammpsPotentialData.get_or_create(
        source=potential_information['filename'],
        filename=potential_information['filename'],
        **potential_information['parameters'],
    )

    potential_block = input_generator.write_potential_block(
        parameters_potential={},
        potential_file='temp.pot',
        potential=node,
        structure=potential_information['structure'],
    )

    reference_file = os.path.join(
        TEST_DIR,
        'test_lammps_potential_data',
        f'test_init_{potential_type}_block.txt',
    )

    with io.open(reference_file, 'r') as handler:
        reference_value = handler.read()

    assert potential_block == reference_value, 'content of the potential blocks differ'
