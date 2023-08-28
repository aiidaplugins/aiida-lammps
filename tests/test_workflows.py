"""Tests for the workflows in aiida-lammps"""
# pylint: disable=redefined-outer-name
from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.engine import ProcessHandlerReport, run_get_node
from aiida.plugins import WorkflowFactory
from plumpy import ProcessState
import pytest

from aiida_lammps.calculations.base import LammpsBaseCalculation
from aiida_lammps.workflows.base import LammpsBaseWorkChain
from .utils import get_default_metadata, recursive_round


@pytest.fixture
def generate_workchain_base(
    generate_workchain,
    generate_inputs_minimize,
    generate_calc_job_node,
):
    """Generate a LammpsBaseWorkChain node"""

    def _generate_workchain_base(
        exit_code=None,
        inputs=None,
        return_inputs=False,
        lammps_base_outputs=None,
    ):

        entry_point = "lammps.base"

        if inputs is None:
            inputs = {"lammps": generate_inputs_minimize()}

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        lammps_base_node = generate_calc_job_node(
            inputs={
                "parameters": orm.Dict(),
                "metadata": get_default_metadata(),
            }
        )
        process.ctx.iteration = 1
        process.ctx.children = [lammps_base_node]

        if lammps_base_outputs is not None:
            for link_label, output_node in lammps_base_outputs.items():
                output_node.base.links.add_incoming(
                    lammps_base_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()

        if exit_code is not None:
            lammps_base_node.set_process_state(ProcessState.FINISHED)
            lammps_base_node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_base


@pytest.fixture
def generate_workchain_relax(
    generate_workchain,
    generate_inputs_minimize,
):
    """Generate a LammpsRelaxWorkChain node."""

    def _generate_workchain_relax(
        exit_code=None,
        inputs=None,
        return_inputs=False,
        lammps_base_outputs=None,
    ):

        entry_point = "lammps.relax"

        if inputs is None:

            _inputs = generate_inputs_minimize()
            _parameters = _inputs["parameters"].get_dict()

            del _parameters["minimize"]
            _inputs["parameters"] = orm.Dict(_parameters)

            inputs = {"lammps": _inputs}

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_relax


@pytest.fixture
def generate_workchain_md(
    generate_workchain,
    generate_inputs_md,
):
    """Generate a LammpsMDWorkChain node."""

    def _generate_workchain_md(
        exit_code=None,
        inputs=None,
        return_inputs=False,
        lammps_base_outputs=None,
    ):
        entry_point = "lammps.md"
        if inputs is None:
            _inputs = generate_inputs_md()
            _parameters = _inputs["parameters"].get_dict()

            del _parameters["md"]
            _inputs["parameters"] = orm.Dict(_parameters)

            inputs = {"lammps": _inputs}
        process = generate_workchain(entry_point, inputs)
        return process

    return _generate_workchain_md


def test_setup(generate_workchain_base):
    """Test `LammpsBaseWorkChain.setup`."""
    process = generate_workchain_base()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)


def test_handle_unrecoverable_failure(generate_workchain_base):
    """Test `LammpsBaseWorkChain.handle_unrecoverable_failure`."""
    process = generate_workchain_base(
        exit_code=LammpsBaseCalculation.exit_codes.ERROR_NO_RETRIEVED_FOLDER  # pylint: disable=no-member
    )
    process.setup()

    result = process.handle_unrecoverable_failure(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert (
        result.exit_code
        == LammpsBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE  # pylint: disable=no-member
    )

    result = process.inspect_process()
    assert (
        result
        == LammpsBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE  # pylint: disable=no-member
    )


def test_handle_out_of_walltime(
    generate_workchain_base,
    fixture_localhost,
    generate_remote_data,
):
    """Test `LammpsBaseWorkChain.handle_out_of_walltime`."""
    remote_data = generate_remote_data(computer=fixture_localhost, remote_path="/tmp")
    process = generate_workchain_base(
        exit_code=LammpsBaseCalculation.exit_codes.ERROR_OUT_OF_WALLTIME,  # pylint: disable=no-member
        lammps_base_outputs={"remote_folder": remote_data},
    )

    process.setup()

    _walltime = process.ctx.inputs.metadata.options["max_wallclock_seconds"]

    result = process.handle_out_of_walltime(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert process.ctx.inputs.metadata.options["max_wallclock_seconds"] > _walltime
    assert result.do_break

    result = process.inspect_process()
    assert result.status == 0


def test_handle_out_of_walltime_from_trajectory(
    generate_workchain_base,
    fixture_localhost,
    generate_remote_data,
    generate_lammps_trajectory,
):
    """Test `LammpsBaseWorkChain.handle_out_of_walltime`."""
    remote_data = generate_remote_data(computer=fixture_localhost, remote_path="/tmp")
    trajectory = generate_lammps_trajectory(computer=fixture_localhost)
    process = generate_workchain_base(
        exit_code=LammpsBaseCalculation.exit_codes.ERROR_OUT_OF_WALLTIME,  # pylint: disable=no-member
        lammps_base_outputs={
            "remote_folder": remote_data,
            "trajectory": trajectory,
            "structure": trajectory.get_step_structure(-1),
        },
    )

    process.setup()

    _walltime = process.ctx.inputs.metadata.options["max_wallclock_seconds"]

    result = process.handle_out_of_walltime(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert process.ctx.inputs.metadata.options["max_wallclock_seconds"] == _walltime
    assert (
        process.ctx.inputs.structure.get_pymatgen()
        == trajectory.get_step_structure(-1).get_pymatgen()
    )
    assert result.do_break

    result = process.inspect_process()
    assert result.status == 0


def test_handle_minimization_not_converged(
    generate_workchain_base,
    fixture_localhost,
    generate_remote_data,
    generate_singlefile_data,
):
    """Test `LammpsBaseWorkChain.handle_minimization_not_converged`."""
    remote_data = generate_remote_data(computer=fixture_localhost, remote_path="/tmp")
    restartfile = generate_singlefile_data(computer=fixture_localhost)
    process = generate_workchain_base(
        exit_code=LammpsBaseCalculation.exit_codes.ERROR_ENERGY_NOT_CONVERGED,  # pylint: disable=no-member
        lammps_base_outputs={"remote_folder": remote_data, "restartfile": restartfile},
    )

    process.setup()

    result = process.handle_minimization_not_converged(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert "input_restartfile" in process.ctx.inputs
    assert result.do_break

    result = process.inspect_process()
    assert result.status == 0


def test_handle_minimization_not_converged_from_trajectrory(
    generate_workchain_base,
    fixture_localhost,
    generate_remote_data,
    generate_lammps_trajectory,
):
    """Test `LammpsBaseWorkChain.handle_minimization_not_converged`."""
    remote_data = generate_remote_data(computer=fixture_localhost, remote_path="/tmp")
    trajectory = generate_lammps_trajectory(computer=fixture_localhost)
    process = generate_workchain_base(
        exit_code=LammpsBaseCalculation.exit_codes.ERROR_ENERGY_NOT_CONVERGED,  # pylint: disable=no-member
        lammps_base_outputs={
            "remote_folder": remote_data,
            "trajectory": trajectory,
            "structure": trajectory.get_step_structure(-1),
        },
    )

    process.setup()

    result = process.handle_minimization_not_converged(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert (
        process.ctx.inputs.structure.get_pymatgen()
        == trajectory.get_step_structure(-1).get_pymatgen()
    )
    assert result.do_break

    result = process.inspect_process()
    assert result.status == 0


@pytest.mark.parametrize(
    "parameters_relax,fix_references",
    [
        (
            {
                "volume": True,
                "positions": True,
                "shape": True,
                "target_pressure": {"x": 0, "y": 0, "z": 0, "xy": 0, "xz": 0, "yz": 0},
            },
            {
                "box/relax": [
                    {
                        "group": "all",
                        "type": ["x", 0, "y", 0, "z", 0, "xy", 0, "xz", 0, "yz", 0],
                    }
                ]
            },
        ),
        (
            {
                "volume": True,
                "positions": True,
                "shape": False,
                "target_pressure": {"x": 0, "y": 0, "z": 0, "xy": 0, "xz": 0, "yz": 0},
            },
            {"box/relax": [{"group": "all", "type": ["iso", 0]}]},
        ),
        ({"volume": False, "positions": True, "shape": False}, {}),
    ],
)
def test_setup_relax(
    parameters_relax,
    fix_references,
    generate_workchain_relax,
    generate_inputs_minimize,
):
    """Test `LammpsRelaxWorkChain.setup`."""

    inputs = {"lammps": generate_inputs_minimize()}
    inputs["relax"] = AttributeDict()
    for key, value in parameters_relax.items():
        if key == "target_pressure":
            inputs["relax"][key] = orm.Dict(dict=value)
        else:
            inputs["relax"][key] = orm.Bool(value)
    process = generate_workchain_relax(inputs=inputs)
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)
    assert "minimize" in process.ctx.inputs.lammps.parameters
    assert "md" not in process.ctx.inputs.lammps.parameters
    assert process.ctx.inputs.lammps.parameters["fix"] == fix_references


def test_setup_md(generate_workchain_md):
    """Test `LammpsMDWorkChain` setup."""
    process = generate_workchain_md()
    process.setup()

    assert "md" in process.ctx.inputs.lammps.parameters
    assert "minimize" not in process.ctx.inputs.lammps.parameters
    assert isinstance(process.ctx.inputs, AttributeDict)


@pytest.mark.lammps_call
@pytest.mark.parametrize(
    "parameters_relax",
    [
        (
            {
                "volume": True,
                "positions": True,
                "shape": True,
                "target_pressure": {"x": 0, "y": 0, "z": 0, "xy": 0, "xz": 0, "yz": 0},
            }
        ),
        (
            {
                "volume": True,
                "positions": True,
                "shape": False,
                "target_pressure": {"x": 0, "y": 0, "z": 0, "xy": 0, "xz": 0, "yz": 0},
            }
        ),
        ({"volume": False, "positions": True, "shape": False}),
    ],
)
def test_relax_workchain(
    db_test_app,
    generate_structure,
    get_potential_fe_eam,
    generate_inputs_minimize,
    parameters_relax,
    data_regression,
    ndarrays_regression,
):
    """Test running the relaxation workchain."""
    work_plugin = "lammps.relax"
    code = db_test_app.get_or_create_code("lammps.base")

    calculation = WorkflowFactory(work_plugin)

    inputs = AttributeDict()
    inputs.lammps = AttributeDict()
    inputs.lammps.code = code
    inputs.lammps.metadata = get_default_metadata()
    inputs.lammps.structure = generate_structure
    inputs.lammps.potential = get_potential_fe_eam
    inputs.lammps.parameters = orm.Dict(dict=generate_inputs_minimize()["parameters"])
    inputs.relax = AttributeDict()
    for key, value in parameters_relax.items():
        if key == "target_pressure":
            inputs["relax"][key] = orm.Dict(dict=value)
        else:
            inputs["relax"][key] = orm.Bool(value)
    results, node = run_get_node(calculation, **inputs)

    assert node.exit_status == 0, "calculation ended in non-zero state"

    assert "results" in results, 'the "results" node not present'

    _results = results["results"].get_dict()
    if (
        "compute_variables" in _results
        and "steps_per_second" in _results["compute_variables"]
    ):
        del _results["compute_variables"]["steps_per_second"]

    # Removing the line of code that produces the warning as it changes with the lammps version
    if "compute_variables" in _results and "warnings" in _results["compute_variables"]:
        for index, entry in enumerate(_results["compute_variables"]["warnings"]):
            _results["compute_variables"]["warnings"][index] = (
                entry.split("(src")[0].strip() if "(src" in entry else entry
            )

    assert "trajectories" in results, 'the "trajectories" node is not present'

    _trajectories_steps = {
        key: results["trajectories"].get_step_data(key).atom_fields
        for key in range(len(results["trajectories"].time_steps))
    }

    data_regression.check(
        recursive_round(
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
