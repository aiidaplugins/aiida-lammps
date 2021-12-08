"""Functions to tests the input file generation"""

import io
import os

import pytest
from aiida_lammps.tests.utils import TEST_DIR

from aiida_lammps.data.lammps_potential import LammpsPotentialData
import aiida_lammps.common.input_generator as input_generator

@pytest.mark.parametrize(
    'potential_type',
    ['tersoff', 'eam_alloy', 'meam', 'morse'],
    'structure_type',
    ['Fe'],
)
def test_input_generate(
    db_test_app,  # pylint: disable=unused-argument
    get_lammps_potential_data,
    potential_type,
    get_structure_data,
    structure_type,
):
    """Test the generation of the input file"""

    # Dictionary with parameters for controlling aiida-lammps
    parameters = {
        'control': {
            'units':'metal',
            'timestep': 1e-5,
        },
        'compute':{
            'pe/atom': [{'type':[{'keyword':" ", 'value':" "}], "group":'all'}],
            'ke/atom': [{'type':[{'keyword':" ", 'value':" "}], "group":'all'}],
            'stress/atom': [{'type':['NULL'], "group":'all'}],
            'pressure': [{'type':['thermo_temp'], 'group':'all'}],
        },
        'md':{
            'integration':{
                'style': 'npt',
                'constraints': {
                    'temp':[300,300,100],
                    'iso': [0.0, 0.0, 1000.0],
                }
            },
            'max_number_steps': 5000,
            'velocity': [{'create':{'temp': 300}, 'group':'all'}]
        },
        'fix':{
            'box/relax': [{"type":['iso', 0.0, 'vmax', 0.001], 'group':'all'}]
        },
        'structure': {'atom_style': 'atomic'},
        'potential': {},
        'thermo':{
            'printing_rate': 100,
            'thermo_printing':{
                'step':True,
                'pe':True,
                'ke': True,
                'press': True,
                'pxx': True,
                'pyy': True,
                'pzz': True,
                }
            },
        'dump': {'dump_rate': 1000}
    }

    input_generator.validate_input_parameters(parameters)
    # Generating the structure
    structure = get_structure_data(structure_type)
    # Generate the potential
    potential_information = get_lammps_potential_data(potential_type)
    potential = LammpsPotentialData.get_or_create(
        source=potential_information['filename'],
        filename=potential_information['filename'],
        **potential_information['parameters'],
    )
    # Generate the input blocks
    control_block = input_generator.write_control_block(parameters_control=parameters['control'])
    compute_block = input_generator.write_compute_block(parameters_compute=parameters['compute'])
    thermo_block, fixed_thermo = input_generator.write_thermo_block(
        parameters_thermo=parameters['thermo'],
        parameters_compute=parameters['compute'],
    )
    md_block = input_generator.write_md_block(parameters_md=parameters['md'])
    structure_block, group_lists = input_generator.write_structure_block(
        parameters_structure=parameters['structure'],
        structure=structure,
        structure_filename='temp.structure',
    )
    fix_block = input_generator.write_fix_block(
        parameters_fix=parameters['fix'],
        group_names=group_lists,
    )
    potential_block = input_generator.write_potential_block(
        parameters_potential=parameters['potential'],
        potential_file='EAM_WCo.txt',
        potential=potential,
        structure=structure,
    )
    dump_block = input_generator.write_dump_block(
        parameters_dump=parameters['dump'],
        parameters_compute=parameters['compute'],
        trajectory_filename='temp.dump',
        atom_style='atom',
    )
    restart_block = input_generator.write_restart_block(restart_filename='restart.aiida')
    final_block = input_generator.write_final_variables_block(fixed_thermo=fixed_thermo)
    # Printing the potential
    input_file = control_block+structure_block+potential_block+fix_block+\
        compute_block+thermo_block+dump_block+md_block+final_block+restart_block

    reference_file = os.path.join(
        TEST_DIR,
        'test_generate_inputs',
        f'test_generate_input_{potential_type}.txt',
    )

    with io.open(reference_file, 'r') as handler:
        reference_value = handler.read()
    
    assert input_file == reference_value, 'the content of the files differ'