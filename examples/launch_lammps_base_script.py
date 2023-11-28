"""Run a LAMMPS calculation from a pre-defined complete input script.

The example input script is taken from https://www.lammps.org/inputs/in.lj.txt and is the first example script for the
official benchmarks of LAMMPS. It is a simple NVE simulation of a Lennard-Jones liquid. When passing a complete input
script for the ``script`` input of the ``LammpsBaseCalculation``, other inputs, such as ``structure`` and ``potential``
no longer have to be specified.
"""
import io
import textwrap

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory


def main(
    script: orm.SinglefileData,
    options: AttributeDict,
    code: orm.Code,
) -> orm.Node:
    """Submission of the calculation.

    :param script: complete input script to be used in the calculation
    :type script: orm.SinglefileData
    :param options: options to control the submission parameters
    :type options: AttributeDict
    :param code: code describing the ``LAMMPS`` calculation
    :type code: orm.Code
    :return: node containing the ``LAMMPS`` calculation
    :rtype: orm.Node
    """
    calculation = CalculationFactory("lammps.base")

    builder = calculation.get_builder()
    builder.code = code
    builder.script = script
    builder.metadata.options = options

    _, node = run_get_node(calculation, **builder)

    return node


if __name__ == "__main__":
    # Get the lammps code defined in AiiDA database
    CODE = orm.load_code("lammps-23.06.2022@localhost")
    # Define the parameters for the resources requested for the calculation
    OPTIONS = AttributeDict()
    OPTIONS.resources = AttributeDict()
    # Total number of machines used
    OPTIONS.resources.num_machines = 1
    # Total number of mpi processes
    OPTIONS.resources.tot_num_mpiprocs = 2

    # Define the complete input script that should be run
    SCRIPT = orm.SinglefileData(
        io.StringIO(
            textwrap.dedent(
                """
                # 3d Lennard-Jones melt

                variable      x index 1
                variable      y index 1
                variable      z index 1

                variable      xx equal 20*$x
                variable      yy equal 20*$y
                variable      zz equal 20*$z

                units         lj
                atom_style    atomic

                lattice       fcc 0.8442
                region        box block 0 ${xx} 0 ${yy} 0 ${zz}
                create_box    1 box
                create_atoms  1 box
                mass          1 1.0

                velocity      all create 1.44 87287 loop geom

                pair_style    lj/cut 2.5
                pair_coeff    1 1 1.0 1.0 2.5

                neighbor      0.3 bin
                neigh_modify  delay 0 every 20 check no

                fix           1 all nve

                run           100
                """
            )
        )
    )

    # Run the aiida-lammps calculation
    submission_node = main(
        script=SCRIPT,
        options=OPTIONS,
        code=CODE,
    )

    print(f"Calculation node: {submission_node}")
