"""Test the aiida-lammps calculations."""
import io
import textwrap

from aiida import orm
from aiida.cmdline.utils.common import get_calcjob_report
from aiida.common import AttributeDict
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory, DataFactory
import numpy as np
import pytest

from aiida_lammps.fixtures.calculations import (
    md_parameters_npt,
    md_parameters_nve,
    md_parameters_nvt,
    md_reference_data_npt,
    md_reference_data_nve,
    md_reference_data_nvt,
    minimize_groups_reference_data,
    minimize_parameters,
    minimize_parameters_groups,
    minimize_reference_data,
)
from aiida_lammps.fixtures.data import generate_structure  # noqa: F401
from aiida_lammps.fixtures.data import get_potential_fe_eam  # noqa: F401
from . import utils as tests


def get_lammps_version(code):
    """Get the version of the lammps code"""
    exec_path = code.get_remote_exec_path()
    return tests.lammps_version(exec_path)


def get_calc_parameters(lammps_version, plugin_name, units, potential_type):
    """
    Get the calculation parameters for a given test

    :param lammps_version: version of the lammps code
    :type lammps_version: str
    :param plugin_name: name of the plugin
    :type plugin_name: str
    :param units: name of the units for the calculation
    :type units: str
    :param potential_type: name of the potential type
    :type potential_type: str
    :raises ValueError: [description]
    :return: dictionary with the calculation parameters
    :rtype: orm.Dict
    """

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
            "thermo_keywords": thermo_keywords,
            "velocity": [
                {
                    "style": "create",
                    "args": [300, 12345],
                    "keywords": {"dist": "gaussian", "mom": True},
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
                    "output_atom": {"dump_rate": 10},
                    "output_system": {"dump_rate": 100, "variables": output_variables},
                },
                {
                    "name": "equilibrate",
                    "steps": 400,
                    "integration": {
                        "style": "nvt",
                        "constraints": {"temp": [300, 300, 0.5]},
                    },
                    "computes": [{"id": "cna", "style": "cna/atom", "args": [3.0]}],
                    "output_atom": {
                        "dump_rate": 100,
                        "average_rate": 10,
                        "ave_variables": ["xu", "yu", "zu"],
                        "variables": ["c_cna"],
                    },
                    "output_system": {
                        "dump_rate": 10,
                        "average_rate": 2,
                        "ave_variables": output_variables,
                    },
                    "restart_rate": 200,
                },
            ],
        }
    else:
        raise ValueError(plugin_name)

    return orm.Dict(dict=parameters_opt)


def sanitize_results(results_dict, round_dp_all=None, round_energy=None):
    """Sanitize the results dictionary for test regression."""
    results_dict.pop("parser_version")
    results_dict.pop("warnings")
    results_dict.pop("steps_per_second", None)
    results_dict.pop("total_wall_time", None)
    if round_energy and "energy" in results_dict:
        results_dict["energy"] = round(results_dict["energy"], round_energy)
    if round_dp_all:
        results_dict = tests.recursive_round(results_dict, round_dp_all)
    return results_dict


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
    db_test_app,  # pylint: disable=unused-argument
    get_potential_data,
    calc_type,
    potential_type,
    file_regression,
):
    """
    Test the generation of the input file for lammps
    """
    pot_data = get_potential_data(potential_type)
    potential_data = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameter_data = get_calc_parameters(
        "17 Aug 2017",
        calc_type,
        potential_data.default_units,
        potential_type,
    )

    calc = CalculationFactory(calc_type)
    content = calc.create_main_input_content(
        parameter_data,
        potential_data,
        kind_symbols=["A", "B"],
        structure_filename="input.data",
        trajectory_filename="output.traj",
        system_filename="sys_info.txt",
        restart_filename="calc.restart",
    )
    file_regression.check(content)


@pytest.mark.parametrize(
    "potential_type",
    ["lennard-jones", "tersoff", "eam", "reaxff"],
)
def test_force_submission(
    db_test_app,
    get_potential_data,
    potential_type,
):
    """
    Test the submission of the force-type calculations.
    """
    calc_plugin = "lammps.force"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameters = get_calc_parameters(
        get_lammps_version(code),
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
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
    "potential_type",
    ["lennard-jones", "tersoff", "eam", "reaxff"],
)
def test_optimize_submission(db_test_app, get_potential_data, potential_type):
    """
    Test the submission of the optimize type of calculation
    """
    calc_plugin = "lammps.optimize"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameters = get_calc_parameters(
        get_lammps_version(code),
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
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
    "potential_type",
    ["lennard-jones", "tersoff", "eam", "reaxff"],
)
def test_md_submission(db_test_app, get_potential_data, potential_type):
    """Test the submission of the md type of calculation"""
    calc_plugin = "lammps.md"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameters = get_calc_parameters(
        get_lammps_version(code),
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
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
    "potential_type",
    ["lennard-jones", "tersoff", "eam", "reaxff"],
)
def test_force_process(
    db_test_app,
    get_potential_data,
    potential_type,
    data_regression,
):
    """Test the functionality of the force calculation type"""
    calc_plugin = "lammps.force"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameters = get_calc_parameters(
        get_lammps_version(code),
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
            "metadata": tests.get_default_metadata(),
            "code": code,
            "structure": pot_data.structure,
            "potential": potential,
            "parameters": parameters,
        }
    )

    output = run_get_node(builder)
    calc_node = output.node

    # raise ValueError(calc_node.base.repository.get_object_content('input.in'))
    # raise ValueError(calc_node.outputs.retrieved.base.repository.get_object_content('_scheduler-stdout.txt'))
    # raise ValueError(calc_node.outputs.retrieved.base.repository.get_object_content('trajectory.lammpstrj'))

    if not calc_node.is_finished_ok:
        print(calc_node.attributes)
        print(get_calcjob_report(calc_node))
        raise Exception(f"finished with exit message: {calc_node.exit_message}")

    link_labels = calc_node.base.links.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(["results", "arrays"])

    data_regression.check(
        {
            "results": sanitize_results(calc_node.outputs.results.get_dict(), 1),
            "arrays": calc_node.outputs.arrays.base.attributes.all,
        }
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "potential_type",
    ["lennard-jones", "tersoff", "eam", "reaxff"],
)
def test_optimize_process(
    db_test_app,
    get_potential_data,
    potential_type,
    data_regression,
):
    """Test the functionality of the optimization calculation type"""
    calc_plugin = "lammps.optimize"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameters = get_calc_parameters(
        get_lammps_version(code),
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
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
        raise Exception(f"finished with exit message: {calc_node.exit_message}")

    link_labels = calc_node.base.links.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(["results", "trajectory_data", "structure"])

    trajectory_data = calc_node.outputs.trajectory_data.base.attributes.all
    # optimization steps may differ between lammps versions
    trajectory_data = {k: v for k, v in trajectory_data.items() if k != "number_steps"}
    data_regression.check(
        {
            "results": sanitize_results(calc_node.outputs.results.get_dict(), 1),
            "trajectory_data": trajectory_data,
            "structure": {"kind_names": calc_node.outputs.structure.get_kind_names()}
            # "structure": tests.recursive_round(
            #     calc_node.outputs.structure.attributes, 1, apply_lists=True
            # ),
        }
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "potential_type",
    ["lennard-jones", "tersoff", "eam"],
)
def test_md_process(
    db_test_app,
    get_potential_data,
    potential_type,
    data_regression,
):
    """Test the functionality of the md calculation type"""
    calc_plugin = "lammps.md"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    version = get_lammps_version(code)
    version_year = version[-4:]
    parameters = get_calc_parameters(
        version,
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
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
        raise Exception(f"finished with exit message: {calc_node.exit_message}")

    link_labels = calc_node.base.links.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(["results", "trajectory_data", "system_data"])

    data_regression.check(
        {
            "results": sanitize_results(
                calc_node.outputs.results.get_dict(),
                round_energy=1,
            ),
            "system_data": calc_node.outputs.system_data.base.attributes.all,
            "trajectory_data": calc_node.outputs.trajectory_data.base.attributes.all,
        },
        basename=f"test_md_process-{potential_type}-{version_year}",
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "potential_type",
    ["lennard-jones", "tersoff", "eam", "reaxff"],
)
def test_md_multi_process(
    db_test_app,
    get_potential_data,
    potential_type,
    data_regression,
):
    """Test the functionality of the multi-stage md calculation type"""
    calc_plugin = "lammps.md.multi"
    code = db_test_app.get_or_create_code(calc_plugin)
    pot_data = get_potential_data(potential_type)
    potential = DataFactory("lammps.potential")(
        potential_type=pot_data.type,
        data=pot_data.data,
    )
    parameters = get_calc_parameters(
        get_lammps_version(code),
        calc_plugin,
        potential.default_units,
        potential_type,
    )
    builder = code.get_builder()
    builder._update(
        {  # pylint: disable=protected-access
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
        raise Exception(f"finished with exit message: {calc_node.exit_message}")

    link_labels = calc_node.base.links.get_outgoing().all_link_labels()
    assert set(link_labels).issuperset(
        [
            "results",
            "retrieved",
            "trajectory__thermalise",
            "trajectory__equilibrate",
            "system__thermalise",
            "system__equilibrate",
        ]
    )

    data_regression.check(
        {
            "retrieved": calc_node.outputs.retrieved.base.repository.list_object_names(),
            "results": sanitize_results(
                calc_node.outputs.results.get_dict(), round_energy=1
            ),
            "system__thermalise": calc_node.outputs.system.thermalise.base.attributes.all,
            "system__equilibrate": calc_node.outputs.system.equilibrate.base.attributes.all,
            "trajectory__thermalise": calc_node.outputs.trajectory.thermalise.base.attributes.all,
            "trajectory__equilibrate": calc_node.outputs.trajectory.equilibrate.base.attributes.all,
        }
    )


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "parameters,reference_data",
    [
        (minimize_parameters(), minimize_reference_data()),
        (minimize_parameters_groups(), minimize_groups_reference_data()),
        (md_parameters_nve(), md_reference_data_nve()),
        (md_parameters_nvt(), md_reference_data_nvt()),
        (md_parameters_npt(), md_reference_data_npt()),
    ],
)
def test_lammps_base(
    db_test_app,
    generate_structure,  # pylint: disable=redefined-outer-name  # noqa: F811
    get_potential_fe_eam,  # pylint: disable=redefined-outer-name  # noqa: F811
    parameters,
    reference_data,
):
    """
    Set of tests for the lammps.base calculation

    :param db_test_app: set of function to setup test profiles
    :param generate_structure: structure used for the tests
    :type generate_structure: orm.StructureDate
    """

    calc_plugin = "lammps.base"
    code = db_test_app.get_or_create_code(calc_plugin)

    calculation = CalculationFactory(calc_plugin)

    inputs = AttributeDict()
    inputs.code = code
    inputs.metadata = tests.get_default_metadata()
    inputs.structure = generate_structure
    inputs.potential = get_potential_fe_eam
    inputs.parameters = orm.Dict(dict=parameters)

    results, node = run_get_node(calculation, **inputs)

    assert node.exit_status == 0, "calculation ended in non-zero state"

    assert "results" in results, 'the "results" node not present'

    for key, value in reference_data.results.items():
        _msg = f'key "{key}" not present'
        assert key in results["results"], _msg
        _msg = f'value for "{key}" does not match'
        _results = results["results"].get_dict()
        if isinstance(_results[key], (int, float)):
            assert np.isclose(
                value,
                _results[key],
                rtol=1e-03,
            ), _msg
        if isinstance(_results[key], dict):
            for sub_key, sub_value in reference_data.results[key].items():
                assert sub_key in _results[key], f'key "{sub_key}" not present'
                if sub_key != "steps_per_second":
                    assert (
                        sub_value == _results[key][sub_key]
                    ), f'value for key "{sub_key}" doe not match'

    assert (
        "time_dependent_computes" in results
    ), 'the "time_dependent_computes" node is not present'
    for key, value in reference_data.time_dependent_computes.items():
        _msg = f'key "{key}" not present'
        assert key in results["time_dependent_computes"].get_arraynames(), _msg
        _msg = f'arrays for "{key}" do not match'
        assert np.allclose(
            value,
            results["time_dependent_computes"].get_array(key),
            rtol=1e-02,
        ), _msg

    assert "trajectories" in results, 'the "trajectories" node is not present'
    _attributes = results["trajectories"].base.attributes.all
    for key, value in reference_data.trajectories["attributes"].items():
        assert key in _attributes, f'the key "{key}" is not present'
        assert value == _attributes[key], f'the values for "{key}" do not match'
    for key, value in reference_data.trajectories["step_data"].items():
        _step_data = results["trajectories"].get_step_data(key).atom_fields
        for sub_key, sub_value in value.items():
            _msg = f'key "{sub_key}" not present'
            assert sub_key in _step_data, _msg
            _msg = f'data for key "{key}" does not match'
            if sub_key != "element":
                assert np.allclose(
                    np.asarray(sub_value, dtype=float),
                    np.asarray(_step_data[sub_key], dtype=float),
                    rtol=1e-02,
                ), _msg
            else:
                assert sub_value == _step_data[sub_key], _msg


def test_lammps_base_script(generate_calc_job, aiida_local_code_factory):
    """Test the ``BaseLammpsCalculation`` with the ``script`` input."""
    from aiida_lammps.calculations.lammps.base import BaseLammpsCalculation

    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    with pytest.raises(
        ValueError,
        match=r"Unless `script` is specified the inputs .* have to be specified.",
    ):
        generate_calc_job("lammps.base", inputs)

    content = textwrap.dedent(
        """
        "velocity      all create 1.44 87287 loop geom
        "pair_style    lj/cut 2.5
        "pair_coeff    1 1 1.0 1.0 2.5
        "neighbor      0.3 bin
        "neigh_modify  delay 0 every 20 check no
        "fix           1 all nve
        "run           10000
        """
    )
    stream = io.StringIO(content)
    script = orm.SinglefileData(stream)

    inputs["script"] = script
    tmp_path, calc_info = generate_calc_job("lammps.base", inputs)
    assert (tmp_path / BaseLammpsCalculation._INPUT_FILENAME).read_text() == content


def test_lammps_base_settings_invalid(generate_calc_job, aiida_local_code_factory):
    """Test the validation of the ``settings`` input."""
    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "settings": orm.Dict({"additional_cmdline_params": ["--option", 1]}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    with pytest.raises(
        ValueError,
        match=r"Invalid value for `additional_cmdline_params`, should be list of strings but got.*",
    ):
        generate_calc_job("lammps.base", inputs)


def test_lammps_base_settings(generate_calc_job, aiida_local_code_factory):
    """Test the ``BaseLammpsCalculation`` with the ``settings`` input."""
    from aiida_lammps.calculations.lammps.base import BaseLammpsCalculation

    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "settings": orm.Dict({"additional_cmdline_params": ["--option", "value"]}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    _, calc_info = generate_calc_job("lammps.base", inputs)
    assert calc_info.codes_info[0].cmdline_params[-2:] == ["--option", "value"]
