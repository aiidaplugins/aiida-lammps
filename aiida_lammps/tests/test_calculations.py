import pytest

from aiida.engine import run_get_node
from aiida.cmdline.utils.common import get_calcjob_report
from aiida.orm import Dict
from aiida.plugins import CalculationFactory, DataFactory

import aiida_lammps.tests.utils as tests


def get_lammps_version(code):
    exec_path = code.get_remote_exec_path()
    return tests.lammps_version(exec_path)


def get_calc_parameters(lammps_version, plugin_name, units, potential_type):

    if potential_type == "reaxff":
        output_variables = ["temp", "etotal", "c_reax[1]"]
        thermo_keywords = ["c_reax[1]"]
    else:
        output_variables = ["temp", "etotal"]
        thermo_keywords = []

    if plugin_name == "lammps.force":
        parameters_opt = {
            "lammps_version": lammps_version,
            "output_variables": output_variables,
            "thermo_keywords": thermo_keywords,
        }
    elif plugin_name == "lammps.optimize":
        parameters_opt = {
            "lammps_version": lammps_version,
            "units": units,
            "relax": {"type": "iso", "pressure": 0.0, "vmax": 0.001},
            "minimize": {
                "style": "cg",
                "energy_tolerance": 1.0e-25,
                "force_tolerance": 1.0e-25,
                "max_evaluations": 100000,
                "max_iterations": 50000,
            },
            "output_variables": output_variables,
            "thermo_keywords": thermo_keywords,
        }

    elif plugin_name == "lammps.md":
        parameters_opt = {
            "lammps_version": lammps_version,
            "units": units,
            "rand_seed": 12345,
            "timestep": 0.001,
            "integration": {"style": "nvt", "constraints": {"temp": [300, 300, 0.5]}},
            "neighbor": [0.3, "bin"],
            "neigh_modify": {"every": 1, "delay": 0, "check": False},
            "equilibrium_steps": 100,
            "total_steps": 1000,
            "dump_rate": 10,
            "restart": 100,
            "output_variables": output_variables,
            "thermo_keywords": thermo_keywords,
        }
    elif plugin_name == "lammps.md.multi":
        parameters_opt = {
            "lammps_version": lammps_version,
            "units": units,
            "timestep": 0.001,
            "neighbor": [0.3, "bin"],
            "neigh_modify": {"every": 1, "delay": 0, "check": False},
            "system_variables": output_variables,
            "thermo_keywords": thermo_keywords,
            "velocity": [
                {
                    "style": "create",
                    "args": [300, 12345],
                    "keywords": {"dist": "gaussian", "mom": "yes"},
                },
                {"style": "scale", "args": [300]},
            ],
            "stages": [
                {
                    "name": "thermalise",
                    "steps": 100,
                    "integration": {
                        "style": "nvt",
                        "constraints": {"temp": [300, 300, 0.5]},
                    },
                    "dump_rate": 10,
                },
                {
                    "name": "equilibrate",
                    "steps": 900,
                    "integration": {
                        "style": "nvt",
                        "constraints": {"temp": [300, 300, 0.5]},
                    },
                    "dump_rate": 100,
                    "restart": 200,
                    "computes": [{"id": "cna", "style": "cna/atom", "args": [3.0]}],
                    "atom_variables": ["c_cna"],
                },
            ],
        }
    else:
        raise ValueError(plugin_name)

    return Dict(dict=parameters_opt)


@pytest.mark.parametrize(
    "potential_type,calc_type",
    [
        ("lennard-jones", "lammps.force"),
        ("tersoff", "lammps.optimize"),
        ("eam", "lammps.md"),
        ("reaxff", "lammps.md.multi"),
    ],
)
def test_input_creation(
    db_test_app, get_potential_data, calc_type, potential_type, file_regression
):
    pot_data = get_potential_data(potential_type)
    potential_data = DataFactory("lammps.potential")(
        type=pot_data.type, data=pot_data.data
    )
    parameter_data = get_calc_parameters(
        "17 Aug 2017", calc_type, potential_data.default_units, potential_type
    )

    calc = CalculationFactory(calc_type)
    content = calc.create_main_input_content(
        parameter_data,
        potential_data,
        kind_symbols=["A", "B"],
        structure_filename="input.data",
        trajectory_filename="output.traj",
        info_filename="sys_info.txt",
        restart_filename="calc.restart",
    )
    file_regression.check(content)


@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_force_submission(db_test_app, get_potential_data, potential_type):
    calc_plugin = "lammps.force"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    with db_test_app.sandbox_folder() as folder:
        calc_info = db_test_app.generate_calcinfo(calc_plugin, folder, builder)

        assert calc_info.codes_info[0].cmdline_params == ["-in", "input.in"]
        assert set(folder.get_content_list()).issuperset(["input.data", "input.in"])


@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_optimize_submission(db_test_app, get_potential_data, potential_type):
    calc_plugin = "lammps.optimize"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    print(parameters.attributes)
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    with db_test_app.sandbox_folder() as folder:
        calc_info = db_test_app.generate_calcinfo(calc_plugin, folder, builder)

        assert calc_info.codes_info[0].cmdline_params == ["-in", "input.in"]
        assert set(folder.get_content_list()).issuperset(["input.data", "input.in"])


@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_md_submission(db_test_app, get_potential_data, potential_type):
    calc_plugin = "lammps.md"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    with db_test_app.sandbox_folder() as folder:
        calc_info = db_test_app.generate_calcinfo(calc_plugin, folder, builder)

        assert calc_info.codes_info[0].cmdline_params == ["-in", "input.in"]
        assert set(folder.get_content_list()).issuperset(["input.data", "input.in"])


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_force_process(
    db_test_app, get_potential_data, potential_type, data_regression
):
    calc_plugin = "lammps.force"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    output = run_get_node(builder)
    calc_node = output.node

    # raise ValueError(calc_node.get_object_content('input.in'))
    # raise ValueError(calc_node.outputs.retrieved.get_object_content('_scheduler-stdout.txt'))
    # raise ValueError(calc_node.outputs.retrieved.get_object_content('trajectory.lammpstrj'))

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception("finished with exit message: {}".format(calc_node.exit_message))

    link_labels = calc_node.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(["results", "arrays"])

    pdict = calc_node.outputs.results.get_dict()
    # assert pdict["warnings"].strip() == pot_data.output["warnings"]
    # assert pdict["energy"] == pytest.approx(pot_data.output["initial_energy"])
    pdict.pop("parser_version")
    pdict["warnings"] = (
        pdict["warnings"]
        .strip()
        .replace("Warning: changed valency_val to valency_boc for X", "")
    )
    data_regression.check(
        {
            "results": tests.recursive_round(pdict, 1),
            "arrays": calc_node.outputs.arrays.attributes,
        }
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_optimize_process(
    db_test_app, get_potential_data, potential_type, data_regression
):
    calc_plugin = "lammps.optimize"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    output = run_get_node(builder)
    calc_node = output.node

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception("finished with exit message: {}".format(calc_node.exit_message))

    link_labels = calc_node.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(["results", "trajectory_data", "structure"])

    pdict = calc_node.outputs.results.get_dict()
    # assert pdict["warnings"].strip() == pot_data.output["warnings"]
    # assert pdict["energy"] == pytest.approx(pot_data.output["energy"])

    pdict.pop("parser_version")
    pdict["warnings"] = (
        pdict["warnings"]
        .strip()
        .replace("Warning: changed valency_val to valency_boc for X", "")
    )
    trajectory_data = calc_node.outputs.trajectory_data.attributes
    # optimization steps may differ between lammps versions
    trajectory_data = {
        k: [None] + v[1:] if k.startswith("array|") else v
        for k, v in trajectory_data.items()
    }
    data_regression.check(
        {
            "results": tests.recursive_round(pdict, 1),
            "trajectory_data": trajectory_data,
            "structure": {"kind_names": calc_node.outputs.structure.get_kind_names()}
            # "structure": tests.recursive_round(
            #     calc_node.outputs.structure.attributes, 1, apply_lists=True
            # ),
        }
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize("potential_type", ["lennard-jones", "tersoff", "eam"])
def test_md_process(db_test_app, get_potential_data, potential_type, data_regression):
    calc_plugin = "lammps.md"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    output = run_get_node(builder)
    calc_node = output.node

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception("finished with exit message: {}".format(calc_node.exit_message))

    link_labels = calc_node.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(["results", "trajectory_data", "system_data"])

    pdict = calc_node.outputs.results.get_dict()
    # assert set(pdict.keys()).issuperset(["warnings", "parser_class", "parser_version"])
    # assert pdict["warnings"].strip() == pot_data.output["warnings"]

    pdict.pop("parser_version")
    pdict["warnings"] = (
        pdict["warnings"]
        .strip()
        .replace("Warning: changed valency_val to valency_boc for X", "")
    )
    pdict["energy"] = round(pdict["energy"], 1)
    data_regression.check(
        {
            "results": pdict,
            "system_data": calc_node.outputs.system_data.attributes,
            "trajectory_data": calc_node.outputs.trajectory_data.attributes,
        }
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "potential_type", ["lennard-jones", "tersoff", "eam", "reaxff"]
)
def test_md_multi_process(
    db_test_app, get_potential_data, potential_type, data_regression
):
    calc_plugin = "lammps.md.multi"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(type=pot_data.type, data=pot_data.data)
    parameters = get_calc_parameters(
        get_lammps_version(code), calc_plugin, potential.default_units, potential_type
    )
    builder = code.get_builder()
    builder._update(
        {
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    output = run_get_node(builder)
    calc_node = output.node

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception("finished with exit message: {}".format(calc_node.exit_message))

    link_labels = calc_node.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(
        ["results", "trajectory__thermalise", "trajectory__equilibrate", "system_data"]
    )

    pdict = calc_node.outputs.results.get_dict()
    # assert set(pdict.keys()).issuperset(["warnings", "parser_class", "parser_version"])
    # assert pdict["warnings"].strip() == pot_data.output["warnings"]

    pdict.pop("parser_version")
    pdict["warnings"] = (
        pdict["warnings"]
        .strip()
        .replace("Warning: changed valency_val to valency_boc for X", "")
    )
    pdict["energy"] = round(pdict["energy"], 1)
    data_regression.check(
        {
            "results": pdict,
            "system_data": calc_node.outputs.system_data.attributes,
            "trajectory__thermalise": calc_node.outputs.trajectory__thermalise.attributes,
            "trajectory__equilibrate": calc_node.outputs.trajectory__equilibrate.attributes,
        }
    )
