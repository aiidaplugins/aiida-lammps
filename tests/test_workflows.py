"""Tests for the workflows in aiida-lammps"""
from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.engine import ProcessHandlerReport
from plumpy import ProcessState
import pytest

from aiida_lammps.calculations.base import LammpsBaseCalculation
from aiida_lammps.workflows.base import LammpsBaseWorkChain
from .utils import get_default_metadata


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
    generate_lammps_results,
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
