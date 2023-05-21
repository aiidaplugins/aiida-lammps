"""Test the aiida-lammps calculations."""
import copy
import io
import textwrap

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory
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
from aiida_lammps.fixtures.data import generate_structure, get_potential_fe_eam
from aiida_lammps.fixtures.inputs import (
    parameters_restart_final,
    parameters_restart_full,
    parameters_restart_full_no_storage,
    parameters_restart_intermediate,
)
from . import utils as tests


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "parameters,reference_data",
    [
        ("minimize_parameters", "minimize_reference_data"),
        ("minimize_parameters_groups", "minimize_groups_reference_data"),
        ("md_parameters_nve", "md_reference_data_nve"),
        ("md_parameters_nvt", "md_reference_data_nvt"),
        ("md_parameters_npt", "md_reference_data_npt"),
    ],
)
def test_lammps_base(
    db_test_app,
    generate_structure,  # pylint: disable=redefined-outer-name  # noqa: F811
    get_potential_fe_eam,  # pylint: disable=redefined-outer-name  # noqa: F811
    parameters,
    reference_data,
    request,
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

    reference_data = request.getfixturevalue(reference_data)

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

    _msg = "No time dependet computes obtained even when expected"
    assert len(results["time_dependent_computes"].get_arraynames()) > 0, _msg

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
    """Test the ``LammpsBaseCalculation`` with the ``script`` input."""
    from aiida_lammps.calculations.base import LammpsBaseCalculation

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
    assert (tmp_path / LammpsBaseCalculation._INPUT_FILENAME).read_text() == content


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "parameters,restart_parameters",
    [
        (
            "md_parameters_npt",
            "parameters_restart_full",
        ),
        (
            "md_parameters_npt",
            "parameters_restart_full_no_storage",
        ),
        (
            "md_parameters_npt",
            "parameters_restart_final",
        ),
        (
            "md_parameters_npt",
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

    # Remove the velocity if pressent so that the simulation is not shaken at the start. This allows
    # for comparizon between parameters from initial and final steps
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


def test_lammps_base_settings(generate_calc_job, aiida_local_code_factory):
    """Test the ``LammpsBaseCalculation`` with the ``settings`` input."""
    from aiida_lammps.calculations.base import LammpsBaseCalculation

    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "settings": orm.Dict({"additional_cmdline_params": ["--option", "value"]}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    _, calc_info = generate_calc_job("lammps.base", inputs)
    assert calc_info.codes_info[0].cmdline_params[-2:] == ["--option", "value"]


def test_lammps_base_files_invalid(generate_calc_job, aiida_local_code_factory):
    """Test the ``files`` input valdiation.

    The list of filenames that will be used to write to the working directory needs to be unique.
    """
    # Create two ``SinglefileData`` nodes without specifying an explicit filename. This will cause the default to be
    # used, and so both will have the same filename, which should trigger the validation error.
    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "files": {
            "file_a": orm.SinglefileData(io.StringIO("content")),
            "file_b": orm.SinglefileData(io.StringIO("content")),
        },
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    with pytest.raises(
        ValueError,
        match=r"The list of filenames of the ``files`` input is not unique:.*",
    ):
        generate_calc_job("lammps.base", inputs)


def test_lammps_base_filenames_invalid(generate_calc_job, aiida_local_code_factory):
    """Test the ``filenames`` input valdiation.

    The list of filenames that will be used to write to the working directory needs to be unique.
    """
    # Create two ``SinglefileData`` nodes with unique filenames but override them using the ``filenames`` input to use
    # the same filename, and so both will have the same filename, which should trigger the validation error.
    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "files": {
            "file_a": orm.SinglefileData(io.StringIO("content"), filename="file_a.txt"),
            "file_b": orm.SinglefileData(io.StringIO("content"), filename="file_b.txt"),
        },
        "filenames": {
            "file_a": "file.txt",
            "file_b": "file.txt",
        },
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    with pytest.raises(
        ValueError,
        match=r"The list of filenames of the ``files`` input is not unique:.*",
    ):
        generate_calc_job("lammps.base", inputs)


def test_lammps_base_files(generate_calc_job, aiida_local_code_factory):
    """Test the ``files`` input."""
    inputs = {
        "code": aiida_local_code_factory("lammps.base", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "files": {
            "file_a": orm.SinglefileData(
                io.StringIO("content a"), filename="file_a.txt"
            ),
            "file_b": orm.SinglefileData(
                io.StringIO("content b"), filename="file_b.txt"
            ),
        },
        "filenames": {"file_b": "custom_filename.txt"},
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    tmp_path, calc_info = generate_calc_job("lammps.base", inputs)
    assert sorted(calc_info.provenance_exclude_list) == [
        "custom_filename.txt",
        "file_a.txt",
    ]
    assert (tmp_path / "file_a.txt").read_text() == "content a"
    assert (tmp_path / "custom_filename.txt").read_text() == "content b"
