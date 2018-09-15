import glob
import os
import sys
import numpy as np
import pytest

import aiida_lammps.tests.utils as tests
from aiida_lammps.tests import TEST_DIR


@pytest.fixture(scope='function')
def example_fe_calc(new_database_with_daemon, new_workdir):
    from aiida.orm import DataFactory
    StructureData = DataFactory('structure')
    ParameterData = DataFactory('parameter')
    # Fe BCC
    cell = [[2.848116, 0.000000, 0.000000],
            [0.000000, 2.848116, 0.000000],
            [0.000000, 0.000000, 2.848116]]

    scaled_positions = [(0.0000000, 0.0000000, 0.0000000),
                        (0.5000000, 0.5000000, 0.5000000)]

    symbols = ['Fe', 'Fe']

    structure = StructureData(cell=cell)
    positions = np.dot(scaled_positions, cell)

    for i, scaled_position in enumerate(scaled_positions):
        structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                              symbols=symbols[i])

    eam_path = os.path.join(TEST_DIR, 'input_files', 'Fe_mm.eam.fs')
    eam_data = {'type': 'fs',
                'file_contents': open(eam_path).readlines()}

    potential_dict = {'pair_style': 'eam', 'data': eam_data}
    potential = ParameterData(dict=potential_dict)

    parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                      'pressure': 0.0,  # kbars
                      'vmax': 0.000001,  # Angstrom^3
                      'energy_tolerance': 1.0e-25,  # eV
                      'force_tolerance': 1.0e-25,  # eV angstrom
                      'max_evaluations': 1000000,
                      'max_iterations': 500000}
    parameters = ParameterData(dict=parameters_opt)

    computer = tests.get_computer(workdir=new_workdir, configure=True)

    from aiida.orm import Code
    code = Code(
        input_plugin_name='lammps.optimize',
        remote_computer_exec=[computer, tests.get_path_to_executable('lammps')],
    )
    code.store()

    calc = code.new_calc()
    calc.set_withmpi(False)
    calc.set_resources({"num_machines": 1, "num_mpiprocs_per_machine": 1})

    calc.label = "test lammps calculation"
    calc.description = "A much longer description"
    calc.use_structure(structure)
    calc.use_potential(potential)

    calc.use_parameters(parameters)

    input_dict = {
        "options": {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1
            },
            "withmpi": False,
            "max_wallclock_seconds": 60
        },
        "structure": structure,
        "potential": potential,
        "parameters": parameters,
        "code": code
    }

    return calc, input_dict


def test_example_fe_submission(example_fe_calc):

    calc, input_dict = example_fe_calc

    from aiida.common.folders import SandboxFolder

    # output input files and scripts to temporary folder
    with SandboxFolder() as folder:
        subfolder, script_filename = calc.submit_test(folder=folder)
        print("inputs created successfully at {}".format(subfolder.abspath))
        print([
            os.path.basename(p)
            for p in glob.glob(os.path.join(subfolder.abspath, "*"))
        ])
        assert subfolder.isfile('input.data')
        assert subfolder.isfile('input.in')
        assert subfolder.isfile('potential.pot')


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    tests.aiida_version() < tests.cmp_version('1.0.0a1') and tests.is_sqla_backend(),
    reason='Error in obtaining authinfo for computer configuration')
def test_example_fe_process(example_fe_calc):
    calc, input_dict = example_fe_calc
    process = calc.process()

    calcnode = tests.run_get_node(process, input_dict)

    print(calcnode.get_inputs_dict())
    assert set(calcnode.get_inputs_dict().keys()).issuperset(
        ['parameters', 'structure', 'potential'])

    print(calcnode.get_outputs_dict())

    sys.stdout.write(tests.get_calc_log(calcnode))

    from aiida.common.datastructures import calc_states
    assert calcnode.get_state() == calc_states.FINISHED

