import pytest

from aiida.engine import run_get_node
from aiida.cmdline.utils.common import get_calcjob_report
from aiida.orm import Dict

import aiida_lammps.tests.utils as tests


def get_calc_parameters(plugin_name, units):

    if plugin_name == 'lammps.optimize':
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

    elif plugin_name == "lammps.md":
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
    else:
        raise ValueError(plugin_name)

    return Dict(dict=parameters_opt)


@pytest.mark.parametrize('potential_type', [
    "lennard-jones",
    "tersoff",
    "eam",
    "reaxff",
])
def test_optimize_submission(db_test_app, get_potential_data, potential_type):
    calc_plugin = 'lammps.optimize'
    code = db_test_app.get_or_create_code(calc_plugin)
    potential_data = get_potential_data(potential_type)
    parameters = get_calc_parameters(calc_plugin, potential_data.units)
    builder = code.get_builder()
    builder._update({
        "metadata": tests.get_default_metadata(),
        "code": code,
        "structure": potential_data.structure,
        "potential": potential_data.potential,
        "parameters": parameters,
    })

    with db_test_app.sandbox_folder() as folder:
        calc_info = db_test_app.generate_calcinfo(calc_plugin, folder, builder)

        assert calc_info.codes_info[0].cmdline_params == ['-in', 'input.in']
        assert sorted(folder.get_content_list()) == sorted(potential_data.input_files)


@pytest.mark.parametrize('potential_type', [
    "lennard-jones",
    "tersoff",
    "eam",
    "reaxff",
])
def test_md_submission(db_test_app, get_potential_data, potential_type):
    calc_plugin = 'lammps.md'
    code = db_test_app.get_or_create_code(calc_plugin)
    potential_data = get_potential_data(potential_type)
    parameters = get_calc_parameters(calc_plugin, potential_data.units)
    builder = code.get_builder()
    builder._update({
        "metadata": tests.get_default_metadata(),
        "code": code,
        "structure": potential_data.structure,
        "potential": potential_data.potential,
        "parameters": parameters,
    })

    with db_test_app.sandbox_folder() as folder:
        calc_info = db_test_app.generate_calcinfo(calc_plugin, folder, builder)

        assert calc_info.codes_info[0].cmdline_params == ['-in', 'input.in']
        assert sorted(folder.get_content_list()) == sorted(potential_data.input_files)


@pytest.mark.lammps_call
@pytest.mark.parametrize('potential_type', [
    "lennard-jones",
    "tersoff",
    "eam",
    "reaxff",
])
def test_optimize_process(db_test_app, get_potential_data, potential_type):
    calc_plugin = 'lammps.optimize'
    code = db_test_app.get_or_create_code(calc_plugin)
    potential_data = get_potential_data(potential_type)
    parameters = get_calc_parameters(calc_plugin, potential_data.units)
    builder = code.get_builder()
    builder._update({
        "metadata": tests.get_default_metadata(),
        "code": code,
        "structure": potential_data.structure,
        "potential": potential_data.potential,
        "parameters": parameters,
    })

    output = run_get_node(builder)
    calc_node = output.node

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception("finished with exit message: {}".format(calc_node.exit_message))

    link_labels = calc_node.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(
        ['results', 'arrays', 'structure'])

    pdict = calc_node.outputs.results.get_dict()
    assert set(pdict.keys()).issuperset(
        ['energy', 'warnings', 'energy_units', 'force_units', 'parser_class', 'parser_version'])
    assert pdict['warnings'].strip() == potential_data.output["warnings"]
    assert pdict['energy'] == pytest.approx(potential_data.output['energy'])

    assert set(calc_node.outputs.arrays.get_arraynames()).issuperset(
        ['stress', 'forces']
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize('potential_type', [
    "lennard-jones",
    "tersoff",
    "eam",
    "reaxff",
])
def test_md_process(db_test_app, get_potential_data, potential_type):
    calc_plugin = 'lammps.md'
    code = db_test_app.get_or_create_code(calc_plugin)
    potential_data = get_potential_data(potential_type)
    parameters = get_calc_parameters(calc_plugin, potential_data.units)
    builder = code.get_builder()
    builder._update({
        "metadata": tests.get_default_metadata(),
        "code": code,
        "structure": potential_data.structure,
        "potential": potential_data.potential,
        "parameters": parameters,
    })

    output = run_get_node(builder)
    calc_node = output.node

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception("finished with exit message: {}".format(calc_node.exit_message))

    link_labels = calc_node.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(
        ['results', 'trajectory_data'])

    pdict = calc_node.outputs.results.get_dict()
    assert set(pdict.keys()).issuperset(
        ['warnings', 'parser_class', 'parser_version'])
    assert pdict['warnings'].strip() == potential_data.output["warnings"]

    assert set(calc_node.outputs.trajectory_data.get_arraynames()).issuperset(
        ['cells', 'positions', 'steps', 'times']
    )
