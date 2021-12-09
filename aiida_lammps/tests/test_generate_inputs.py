"""Functions to tests the input file generation"""

import io
import os
import yaml
import pytest
from aiida_lammps.tests.utils import TEST_DIR
from aiida_lammps.data.lammps_potential import LammpsPotentialData
from aiida_lammps.common import input_generator


@pytest.mark.parametrize(
    'potential_type',
    ['eam_alloy'],
)
def test_input_generate_minimize(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the generation of the input file for minimize calculations"""
    # pylint: disable=too-many-locals

    parameters_file = os.path.join(
        TEST_DIR,
        'test_generate_inputs',
        'parameters_minimize.yaml',
    )
    # Dictionary with parameters for controlling aiida-lammps
    with open(parameters_file) as handler:
        parameters = yaml.load(handler, yaml.SafeLoader)

    input_generator.validate_input_parameters(parameters)
    # Generate the potential
    potential_information = get_lammps_potential_data(potential_type)
    potential = LammpsPotentialData.get_or_create(
        source=potential_information['filename'],
        filename=potential_information['filename'],
        **potential_information['parameters'],
    )
    # Generating the structure
    structure = potential_information['structure']
    # Generate the input blocks
    control_block = input_generator.write_control_block(
        parameters_control=parameters['control'])
    compute_block = input_generator.write_compute_block(
        parameters_compute=parameters['compute'])
    thermo_block, fixed_thermo = input_generator.write_thermo_block(
        parameters_thermo=parameters['thermo'],
        parameters_compute=parameters['compute'],
    )
    minimize_block = input_generator.write_minimize_block(
        parameters_minimize=parameters['minimize'])
    structure_block, group_lists = input_generator.write_structure_block(
        parameters_structure=parameters['structure'],
        structure=structure,
        structure_filename='structure.dat',
    )
    fix_block = input_generator.write_fix_block(
        parameters_fix=parameters['fix'],
        group_names=group_lists,
    )
    potential_block = input_generator.write_potential_block(
        parameters_potential=parameters['potential'],
        potential_file='potential.dat',
        potential=potential,
        structure=structure,
    )
    dump_block = input_generator.write_dump_block(
        parameters_dump=parameters['dump'],
        parameters_compute=parameters['compute'],
        trajectory_filename='temp.dump',
        atom_style='atom',
    )
    restart_block = input_generator.write_restart_block(
        restart_filename='restart.aiida')
    final_block = input_generator.write_final_variables_block(
        fixed_thermo=fixed_thermo)
    # Printing the potential
    input_file = control_block+structure_block+potential_block+fix_block+\
        compute_block+thermo_block+dump_block+minimize_block+final_block+\
        restart_block
    print(input_file)
    reference_file = os.path.join(
        TEST_DIR,
        'test_generate_inputs',
        f'test_generate_input_{potential_type}_minimize.txt',
    )

    with io.open(reference_file, 'r') as handler:
        reference_value = handler.read()

    assert input_file == reference_value, 'the content of the files differ'


@pytest.mark.parametrize(
    'potential_type',
    ['eam_alloy'],
)
def test_input_generate_md(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
):
    """Test the generation of the input file for MD calculations"""
    # pylint: disable=too-many-locals

    parameters_file = os.path.join(
        TEST_DIR,
        'test_generate_inputs',
        'parameters_md.yaml',
    )
    # Dictionary with parameters for controlling aiida-lammps
    with open(parameters_file) as handler:
        parameters = yaml.load(handler, yaml.SafeLoader)

    input_generator.validate_input_parameters(parameters)
    # Generate the potential
    potential_information = get_lammps_potential_data(potential_type)
    potential = LammpsPotentialData.get_or_create(
        source=potential_information['filename'],
        filename=potential_information['filename'],
        **potential_information['parameters'],
    )
    # Generating the structure
    structure = potential_information['structure']
    # Generate the input blocks
    control_block = input_generator.write_control_block(
        parameters_control=parameters['control'])
    compute_block = input_generator.write_compute_block(
        parameters_compute=parameters['compute'])
    thermo_block, fixed_thermo = input_generator.write_thermo_block(
        parameters_thermo=parameters['thermo'],
        parameters_compute=parameters['compute'],
    )
    md_block = input_generator.write_minimize_block(
        parameters_minimize=parameters['md'])
    structure_block, group_lists = input_generator.write_structure_block(
        parameters_structure=parameters['structure'],
        structure=structure,
        structure_filename='structure.dat',
    )
    fix_block = input_generator.write_fix_block(
        parameters_fix=parameters['fix'],
        group_names=group_lists,
    )
    potential_block = input_generator.write_potential_block(
        parameters_potential=parameters['potential'],
        potential_file='potential.dat',
        potential=potential,
        structure=structure,
    )
    dump_block = input_generator.write_dump_block(
        parameters_dump=parameters['dump'],
        parameters_compute=parameters['compute'],
        trajectory_filename='temp.dump',
        atom_style='atom',
    )
    restart_block = input_generator.write_restart_block(
        restart_filename='restart.aiida')
    final_block = input_generator.write_final_variables_block(
        fixed_thermo=fixed_thermo)
    # Printing the potential
    input_file = control_block+structure_block+potential_block+fix_block+\
        compute_block+thermo_block+dump_block+md_block+final_block+\
        restart_block
    reference_file = os.path.join(
        TEST_DIR,
        'test_generate_inputs',
        f'test_generate_input_{potential_type}_md.txt',
    )
    print(input_file)
    with io.open(reference_file, 'r') as handler:
        reference_value = handler.read()

    assert input_file == reference_value, 'the content of the files differ'
