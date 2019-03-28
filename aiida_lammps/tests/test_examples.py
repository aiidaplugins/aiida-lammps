import glob
import os
import sys

import numpy as np
import pytest

import aiida_lammps.tests.utils as tests
from aiida_lammps.common.reaxff_convert import read_reaxff_file
from aiida_lammps.utils import aiida_version, cmp_version
from aiida.orm import Dict

def eam_data():
    cell = [[2.848116, 0.000000, 0.000000],
            [0.000000, 2.848116, 0.000000],
            [0.000000, 0.000000, 2.848116]]

    scaled_positions = [(0.0000000, 0.0000000, 0.0000000),
                        (0.5000000, 0.5000000, 0.5000000)]

    symbols = ['Fe', 'Fe']

    struct_dict = {"cell": cell,
                   "symbols": symbols,
                   "scaled_positions": scaled_positions}

    eam_path = os.path.join(tests.TEST_DIR, 'input_files', 'Fe_mm.eam.fs')

    data = {'type': 'fs',
            'file_contents': open(eam_path).readlines()}

    potential_dict = {'pair_style': 'eam', 'data': data}

    output_dict = {"energy": -8.2448702,
                   "infiles": ['input.data', 'input.in', 'potential.pot', 'input.units'],
                   "warnings": []}

    return struct_dict, potential_dict, output_dict


def lj_data():
    cell = [[3.987594, 0.000000, 0.000000],
            [-1.993797, 3.453358, 0.000000],
            [0.000000, 0.000000, 6.538394]]

    symbols = ['Ar'] * 2
    scaled_positions = [(0.33333, 0.66666, 0.25000),
                        (0.66667, 0.33333, 0.75000)]

    struct_dict = {"cell": cell,
                   "symbols": symbols,
                   "scaled_positions": scaled_positions}

    # Example LJ parameters for Argon. These may not be accurate at all
    potential_dict = {
        'pair_style': 'lennard_jones',
        #                 epsilon,  sigma, cutoff
        'data': {'1  1': '0.01029   3.4    2.5',
                 # '2  2':   '1.0      1.0    2.5',
                 # '1  2':   '1.0      1.0    2.5'
                 }
    }

    output_dict = {"energy": 0.0,  # TODO should LJ energy be 0?
                   "infiles": ['input.data', 'input.in', 'input.units'],
                   "warnings": []}

    return struct_dict, potential_dict, output_dict


def tersoff_data():
    cell = [[3.1900000572, 0, 0],
            [-1.5950000286, 2.762621076, 0],
            [0.0, 0, 5.1890001297]]

    scaled_positions = [(0.6666669, 0.3333334, 0.0000000),
                        (0.3333331, 0.6666663, 0.5000000),
                        (0.6666669, 0.3333334, 0.3750000),
                        (0.3333331, 0.6666663, 0.8750000)]

    symbols = ['Ga', 'Ga', 'N', 'N']

    struct_dict = {"cell": cell,
                   "symbols": symbols,
                   "scaled_positions": scaled_positions}

    tersoff_gan = {
        'Ga Ga Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199',
        'N  N  N': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77',
        'Ga Ga N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
        'Ga N  N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
        'N  Ga Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
        'N  Ga N ': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000',
        'N  N  Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
        'Ga N  Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000'}

    potential_dict = {'pair_style': 'tersoff',
                      'data': tersoff_gan}
    
    output_dict = {"energy": -18.110852,
                   "infiles": ['input.data', 'input.in', 'potential.pot', 'input.units'],
                   "warnings": []}
    return struct_dict, potential_dict, output_dict


def reaxff_data():
    # pyrite
    cell = [[5.38, 0.000000, 0.000000],
            [0.000000, 5.38, 0.000000],
            [0.000000, 0.000000, 5.38]]

    scaled_positions = [[0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.338, 0.338, 0.338],
                        [0.662, 0.662, 0.662],
                        [0.162, 0.662, 0.838],
                        [0.838, 0.338, 0.162],
                        [0.662, 0.838, 0.162],
                        [0.338, 0.162, 0.838],
                        [0.838, 0.162, 0.662],
                        [0.162, 0.838, 0.338]]

    symbols = ['Fe'] * 4 + ['S'] * 8

    struct_dict = {"cell": cell,
                   "symbols": symbols,
                   "scaled_positions": scaled_positions}

    reaxff_path = os.path.join(tests.TEST_DIR, 'input_files', 'FeCrOSCH.reaxff')
    reaxff_params = {
        'file_contents': open(reaxff_path).readlines(),
        'safezone': 1.6,
    }

    potential_dict = {'pair_style': 'reaxff', 'data': reaxff_params}

    output_dict = {"energy": -1030.3543,
                   "units": "real",
                   "infiles": ['input.data', 'input.in', 'potential.pot', 'input.units'],
                   "warnings": ['Warning: changed valency_val to valency_boc for X']}

    return struct_dict, potential_dict, output_dict


def reaxff_data_param_dict():
    # pyrite
    cell = [[5.38, 0.000000, 0.000000],
            [0.000000, 5.38, 0.000000],
            [0.000000, 0.000000, 5.38]]

    scaled_positions = [[0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.338, 0.338, 0.338],
                        [0.662, 0.662, 0.662],
                        [0.162, 0.662, 0.838],
                        [0.838, 0.338, 0.162],
                        [0.662, 0.838, 0.162],
                        [0.338, 0.162, 0.838],
                        [0.838, 0.162, 0.662],
                        [0.162, 0.838, 0.338]]

    symbols = ['Fe'] * 4 + ['S'] * 8

    struct_dict = {"cell": cell,
                   "symbols": symbols,
                   "scaled_positions": scaled_positions}

    reaxff_path = os.path.join(tests.TEST_DIR, 'input_files', 'FeCrOSCH.reaxff')
    reaxff_params = read_reaxff_file(reaxff_path)
    reaxff_params['safezone'] = 1.6

    potential_dict = {'pair_style': 'reaxff', 'data': reaxff_params}

    output_dict = {"energy": -1030.3543,
                   "units": "real",
                   "infiles": ['input.data', 'input.in', 'potential.pot', 'input.units'],
                   "warnings": ['Warning: changed valency_val to valency_boc for X']}

    return struct_dict, potential_dict, output_dict


def setup_calc(workdir, configure, struct_dict, potential_dict, ctype, units='metal'):

    from aiida.plugins import DataFactory
    StructureData = DataFactory('structure')

    computer = tests.get_computer(workdir=workdir, configure=configure)

    structure = StructureData(cell=struct_dict["cell"])

    for scaled_position, symbols in zip(struct_dict["scaled_positions"], struct_dict["symbols"]):
        structure.append_atom(position=np.dot(scaled_position, struct_dict["cell"]).tolist(),
                              symbols=symbols)

    potential = Dict(dict=potential_dict)

    if ctype == "optimisation":
        parameters_opt = {
            'lammps_version': tests.lammps_version(),
            'units': units,
            'relax': {
                'type': 'iso',
                'pressure': 0.0,
                'vmax': 0.001,
            },
            "minimize": {
                'style': 'cg',
                'energy_tolerance': 1.0e-25,
                'force_tolerance': 1.0e-25,
                'max_evaluations': 100000,
                'max_iterations': 50000}
        }

        code_plugin = 'lammps.optimize'
    elif ctype == "md":
        parameters_opt = {
            'lammps_version': tests.lammps_version(),
            'units': units,
            'timestep': 0.001,
            'integration': {
                'style': 'nvt',
                'constraints': {
                    'temp': [300, 300, 0.5]
                }
            },
            "neighbor": [0.3, "bin"],
            "neigh_modify": {"every": 1, "delay": 0, "check": False},
            'equilibrium_steps': 100,
            'total_steps': 1000,
            'dump_rate': 1}
        code_plugin = 'lammps.md'
    else:
        raise NotImplementedError

    parameters = Dict(dict=parameters_opt)

    from aiida.orm import Code

    code = Code(
        input_plugin_name=code_plugin,
        remote_computer_exec=[computer, tests.get_path_to_executable(tests.TEST_EXECUTABLE)],
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


@pytest.mark.parametrize('data_func', [
    lj_data,
    tersoff_data,
    eam_data,
    reaxff_data,
    reaxff_data_param_dict
])
def test_opt_submission(new_database, new_workdir, data_func):
    struct_dict, potential_dict, output_dict = data_func()

    calc, input_dict = setup_calc(new_workdir, False,
                                  struct_dict, potential_dict, 'optimisation',
                                  output_dict.get('units', 'metal'))

    from aiida.common.folders import SandboxFolder

    # output input files and scripts to temporary folder
    with SandboxFolder() as folder:
        subfolder, script_filename = calc.submit_test(folder=folder)
        print("inputs created successfully at {}".format(subfolder.abspath))
        print([
            os.path.basename(p)
            for p in glob.glob(os.path.join(subfolder.abspath, "*"))
        ])
        for infile in output_dict['infiles']:
            assert subfolder.isfile(infile)
            print('---')
            print(infile)
            print('---')
            with subfolder.open(infile) as f:
                print(f.read())


@pytest.mark.parametrize('data_func', [
    lj_data,
    tersoff_data,
    eam_data,
    reaxff_data
])
def test_md_submission(new_database, new_workdir, data_func):
    struct_dict, potential_dict, output_dict = data_func()

    calc, input_dict = setup_calc(new_workdir, False,
                                  struct_dict, potential_dict, 'md',
                                  output_dict.get('units', 'metal'))

    from aiida.common.folders import SandboxFolder

    # output input files and scripts to temporary folder
    with SandboxFolder() as folder:
        subfolder, script_filename = calc.submit_test(folder=folder)
        print("inputs created successfully at {}".format(subfolder.abspath))
        print([
            os.path.basename(p)
            for p in glob.glob(os.path.join(subfolder.abspath, "*"))
        ])
        for infile in output_dict['infiles']:
            assert subfolder.isfile(infile)
            print('---')
            print(infile)
            print('---')
            with subfolder.open(infile) as f:
                print(f.read())


@pytest.mark.lammps_call
@pytest.mark.timeout(120)
@pytest.mark.skipif(
    aiida_version() < cmp_version('1.0.0a1') and tests.is_sqla_backend(),
    reason='Error in obtaining authinfo for computer configuration')
@pytest.mark.parametrize('data_func', [
    lj_data,
    tersoff_data,
    eam_data,
    reaxff_data,
    reaxff_data_param_dict
])
def test_opt_process(new_database_with_daemon, new_workdir, data_func):
    struct_dict, potential_dict, output_dict = data_func()

    calc, input_dict = setup_calc(new_workdir, True,
                                  struct_dict, potential_dict, 'optimisation',
                                  output_dict.get('units', 'metal'))

    process = calc.process()

    calcnode = tests.run_get_node(process, input_dict)

    sys.stdout.write(tests.get_calc_log(calcnode))

    print(calcnode.get_inputs_dict())
    assert set(calcnode.get_inputs_dict().keys()).issuperset(
        ['parameters', 'structure', 'potential'])

    print(calcnode.get_outputs_dict())
    assert set(calcnode.get_outputs_dict().keys()).issuperset(
        ['output_parameters', 'output_array', 'output_structure'])

    #from aiida.common.datastructures import calc_states
    #assert calcnode.get_state() == calc_states.FINISHED

    pdict = calcnode.out.output_parameters.get_dict()
    assert set(pdict.keys()).issuperset(
        ['energy', 'warnings', 'energy_units', 'force_units', 'parser_class', 'parser_version'])
    assert pdict['warnings'] == output_dict["warnings"]
    assert pdict['energy'] == pytest.approx(output_dict['energy'])

    assert set(calcnode.out.output_array.get_arraynames()).issuperset(
        ['stress', 'forces']
    )


@pytest.mark.lammps_call
@pytest.mark.timeout(180)
@pytest.mark.skipif(
    aiida_version() < cmp_version('1.0.0a1') and tests.is_sqla_backend(),
    reason='Error in obtaining authinfo for computer configuration')
@pytest.mark.parametrize('data_func', [
    lj_data,
    tersoff_data,
    eam_data,
    reaxff_data
])
def test_md_process(new_database_with_daemon, new_workdir, data_func):
    struct_dict, potential_dict, output_dict = data_func()

    calc, input_dict = setup_calc(new_workdir, True,
                                  struct_dict, potential_dict, 'md',
                                  output_dict.get('units', 'metal'))

    process = calc.process()

    calcnode = tests.run_get_node(process, input_dict)

    sys.stdout.write(tests.get_calc_log(calcnode))

    print(calcnode.get_inputs_dict())
    print(calcnode.get_outputs_dict())

    assert set(calcnode.get_inputs_dict().keys()).issuperset(
        ['parameters', 'structure', 'potential'])

    assert set(calcnode.get_outputs_dict().keys()).issuperset(
        ['output_parameters', 'trajectory_data'])

    # from aiida.common.datastructures import calc_states
    # assert calcnode.get_state() == calc_states.FINISHED

    pdict = calcnode.out.output_parameters.get_dict()
    assert set(pdict.keys()).issuperset(
        ['warnings', 'parser_class', 'parser_version'])
    assert pdict['warnings'] == output_dict["warnings"]

    print(dict(calcnode.out.trajectory_data.iterarrays()))

    # assert set(calcnode.out.trajectory_data.get_arraynames()).issuperset(
    #     ['stress', 'forces']
    # )
