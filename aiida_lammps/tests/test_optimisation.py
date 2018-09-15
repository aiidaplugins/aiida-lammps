import glob
import os
import numpy as np

import aiida_lammps.tests.utils as tests
from aiida_lammps.tests import TEST_DIR


def test_example_fe_submission(new_database, new_workdir):
    from aiida.orm import DataFactory
    StructureData = DataFactory('structure')
    ParameterData = DataFactory('parameter')
    from aiida.common.folders import SandboxFolder

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

    potential = {'pair_style': 'eam', 'data': eam_data}

    parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                      'pressure': 0.0,  # kbars
                      'vmax': 0.000001,  # Angstrom^3
                      'energy_tolerance': 1.0e-25,  # eV
                      'force_tolerance': 1.0e-25,  # eV angstrom
                      'max_evaluations': 1000000,
                      'max_iterations': 500000}

    computer = tests.get_computer(workdir=new_workdir)

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
    calc.use_potential(ParameterData(dict=potential))

    calc.use_parameters(ParameterData(dict=parameters_opt))

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
