"""Test the aiida-lammps calculations."""
import copy

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory
import numpy as np
import pytest

from . import utils as tests


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "parameters",
    [
        ("parameters_minimize"),
        ("parameters_minimize_groups"),
        ("parameters_md_nve"),
        ("parameters_md_nvt"),
        ("parameters_md_npt"),
    ],
)
def test_lammps_base(
    db_test_app,
    generate_structure,
    get_potential_fe_eam,
    parameters,
    request,
    data_regression,
    ndarrays_regression,
):
    """
    Set of tests for the lammps.base calculation

    :param db_test_app: set of function to setup test profiles
    :param generate_structure: structure used for the tests
    :type generate_structure: orm.StructureDate
    """
    # pylint: disable=too-many-arguments, too-many-locals

    calc_plugin = "lammps.base"
    code = db_test_app.get_or_create_code(calc_plugin)

    calculation = CalculationFactory(calc_plugin)

    inputs = AttributeDict()
    inputs.code = code
    inputs.metadata = tests.get_default_metadata()
    inputs.structure = generate_structure
    inputs.potential = get_potential_fe_eam
    inputs.parameters = orm.Dict(dict=request.getfixturevalue(parameters))

    results, node = run_get_node(calculation, **inputs)

    assert node.exit_status == 0, "calculation ended in non-zero state"

    assert "results" in results, 'the "results" node not present'

    _results = results["results"].get_dict()
    if (
        "compute_variables" in _results
        and "steps_per_second" in _results["compute_variables"]
    ):
        del _results["compute_variables"]["steps_per_second"]

    assert "trajectories" in results, 'the "trajectories" node is not present'

    _trajectories_steps = {
        key: results["trajectories"].get_step_data(key).atom_fields
        for key in range(len(results["trajectories"].time_steps))
    }

    data_regression.check(
        tests.recursive_round(
            {
                "results": _results,
                "trajectories_attributes": results["trajectories"].base.attributes.all,
                "trajectories_steps": _trajectories_steps,
            },
            2,
            apply_lists=True,
        )
    )

    assert (
        "time_dependent_computes" in results
    ), 'the "time_dependent_computes" node is not present'

    _msg = "No time dependent computes obtained even when expected"
    assert len(results["time_dependent_computes"].get_arraynames()) > 0, _msg

    _time_dependent_computes = {
        key: results["time_dependent_computes"].get_array(key)
        for key in results["time_dependent_computes"].get_arraynames()
    }

    ndarrays_regression.check(_time_dependent_computes)


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "parameters,restart_parameters",
    [
        (
            "parameters_md_npt",
            "parameters_restart_full",
        ),
        (
            "parameters_md_npt",
            "parameters_restart_full_no_storage",
        ),
        (
            "parameters_md_npt",
            "parameters_restart_final",
        ),
        (
            "parameters_md_npt",
            "parameters_restart_intermediate",
        ),
    ],
)
def test_lammps_restart_generation(
    db_test_app,
    generate_structure,  # pylint: disable=redefined-outer-name  # noqa: F811
    get_potential_fe_eam,  # pylint: disable=redefined-outer-name  # noqa: F811
    parameters,
    restart_parameters,
    request,
):
    """Test the generation of the restart file as well as running using a previous restartfile"""
    calc_plugin = "lammps.base"
    code = db_test_app.get_or_create_code(calc_plugin)

    calculation = CalculationFactory(calc_plugin)

    parameters = request.getfixturevalue(parameters)

    restart_parameters = request.getfixturevalue(restart_parameters)

    parameters.restart = restart_parameters.restart

    inputs = AttributeDict()
    inputs.code = code
    inputs.metadata = tests.get_default_metadata()
    inputs.structure = generate_structure
    inputs.potential = get_potential_fe_eam
    inputs.parameters = orm.Dict(dict=parameters)

    if "settings" in restart_parameters:
        inputs.settings = orm.Dict(dict=restart_parameters.settings)

    results, node = run_get_node(calculation, **inputs)

    assert node.exit_status == 0, "calculation ended in non-zero state"

    for _node in ["results", "retrieved", "remote_folder"]:
        assert _node in results, f"the '{_node}' node it not present"

    # Check that the restartfile is stored
    if restart_parameters.get("settings", {}).get("store_restart", False):
        assert "restartfile" in results, "The restartfile is not found"
        if restart_parameters.restart.get("print_final", False):
            _msg = "The restartfile is not in the retrieved folder"
            assert (
                node.get_option("restart_filename")
                in results["retrieved"].base.repository.list_object_names()
            ), _msg
    else:
        # Check that if the file was not asked to be stored that it is not stored
        assert (
            not "restartfile" in results
        ), "The restartfile is stored even when it was not requested"
        if restart_parameters.restart.get("print_final", False):
            _msg = "The restartfile is in the retrieved folder even when it was not requested"
            assert (
                not node.get_option("restart_filename")
                in results["retrieved"].base.repository.list_object_names()
            ), _msg

    # Check that the final restartfile is printed
    if restart_parameters.restart.get("print_final", False):
        _msg = "The restartfile was not created by the lammps calculation"
        assert (
            node.get_option("restart_filename") in results["remote_folder"].listdir()
        ), _msg

    # Check that the intermediate restartfiles are printed
    if restart_parameters.restart.get("print_intermediate", False):
        restartfiles = [
            entry
            for entry in results["remote_folder"].listdir()
            if node.get_option("restart_filename") in entry
        ]
        _msg = (
            "The intermediate restartfiles were not created by the lammps calculation"
        )
        assert len(restartfiles) > 0, _msg

    # Remove the velocity if present so that the simulation is not shaken at the start. This allows
    # for comparison between parameters from initial and final steps
    _parameters = copy.deepcopy(parameters)
    if "velocity" in _parameters["md"]:
        del _parameters["md"]["velocity"]

    # Set the parameters for the restart calculation
    inputs_restart = AttributeDict()
    inputs_restart.code = code
    inputs_restart.metadata = tests.get_default_metadata()
    inputs_restart.structure = generate_structure
    inputs_restart.potential = get_potential_fe_eam
    inputs_restart.parameters = orm.Dict(dict=_parameters)

    # Add the appropriate restart input for the calculation
    if restart_parameters.get("settings", {}).get("store_restart", False):
        inputs_restart.input_restartfile = results["restartfile"]
    else:
        inputs_restart.parent_folder = results["remote_folder"]

    # Add the appropriate restart setting
    if "settings" in restart_parameters:
        inputs_restart.settings = orm.Dict(dict=restart_parameters.settings)

    # run the restart calculation
    results_restart, node_restart = run_get_node(calculation, **inputs_restart)

    # Check that the restart calculation ended properly
    assert node_restart.exit_status == 0, "calculation ended in non-zero state"

    # Check that the last step of the initial calculation and the initial step of the restart match
    # This should happen if one runs with the same version of lammps in the same hardware
    for _name in results_restart["time_dependent_computes"].get_arraynames():
        if _name.lower() not in ["step"]:
            _msg = f"The initial value of the restart compute '{_name}' is not equal to the last step of the input"
            assert np.isclose(
                results_restart["time_dependent_computes"].get_array(_name)[0],
                results["time_dependent_computes"].get_array(_name)[-1],
            ), _msg

    # Check that the initial structure from the relax and the final structure from the input match
    _msg = "The initial cell from the restart does not match the final cell"
    assert np.allclose(
        results_restart["trajectories"].get_step_structure(0).get_ase().cell,
        results["trajectories"].get_step_structure(-1).get_ase().cell,
    ), _msg

    _msg = "The atomic positions from the restart does not match the final atomic positions"
    assert np.allclose(
        results_restart["trajectories"]
        .get_step_structure(0)
        .get_ase()
        .get_scaled_positions(),
        results["trajectories"].get_step_structure(-1).get_ase().get_scaled_positions(),
    ), _msg


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


def test_lammps_base_settings(
    generate_calc_job,
    aiida_local_code_factory,
    parameters_minimize,
    get_potential_fe_eam,
    generate_structure,
):
    """Test the ``LammpsBaseCalculation`` with the ``settings`` input."""

    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "parameters": orm.Dict(parameters_minimize),
        "potential": get_potential_fe_eam,
        "structure": generate_structure,
        "settings": orm.Dict({"additional_cmdline_params": ["--option", "value"]}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    _, calc_info = generate_calc_job("lammps.base", inputs)
    assert calc_info.codes_info[0].cmdline_params[-2:] == ["--option", "value"]
