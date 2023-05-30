from aiida import orm
from aiida.common import AttributeDict, LinkType
from plumpy import ProcessState
import pytest


@pytest.fixture
def generate_workchain_base(
    generate_workchain, generate_inputs_minimize, generate_calc_job_node
):
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

        lammps_base_node = generate_calc_job_node(inputs={"parameters": orm.Dict()})
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
