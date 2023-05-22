"""Run a LAMMPS calculation with additional input files

The example input script is taken from https://www.lammps.org/inputs/in.rhodo.txt and is an example benchmark script for
the official benchmarks of LAMMPS. It is a simple MD simulation of a protein. It requires an additional input file in
the working directory ``data.rhodo``. This example shows how to add such additional input files.
"""
import io
import textwrap

from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory
import numpy as np

from aiida_lammps.data.potential import LammpsPotentialData


def main(
    script: orm.SinglefileData,
    data: orm.SinglefileData,
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
    builder.files = {
        "data": data,
    }
    builder.filenames = {
        "data": "data.rhodo",
    }
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
                # Rhodopsin model

                units           real
                neigh_modify    delay 5 every 1

                atom_style      full
                bond_style      harmonic
                angle_style     charmm
                dihedral_style  charmm
                improper_style  harmonic
                pair_style      lj/charmm/coul/long 8.0 10.0
                pair_modify     mix arithmetic
                kspace_style    pppm 1e-4

                read_data       data.rhodo

                fix             1 all shake 0.0001 5 0 m 1.0 a 232
                fix             2 all npt temp 300.0 300.0 100.0 &
                        z 0.0 0.0 1000.0 mtk no pchain 0 tchain 1

                special_bonds   charmm

                thermo          50
                thermo_style    multi
                timestep        2.0

                run     100
                """
            )
        )
    )

    # Define the additional input data file. Note that for brevity's sake, it is not complete.
    DATA = orm.SinglefileData(
        io.StringIO(
            textwrap.dedent(
                """
                LAMMPS data file from restart file: timestep = 5000, procs = 1

                32000 atoms
                27723 bonds
                40467 angles
                56829 dihedrals
                1034 impropers
                ...
                """
            )
        )
    )

    # Run the aiida-lammps calculation
    submission_node = main(
        script=SCRIPT,
        data=data,
        options=OPTIONS,
        code=CODE,
    )

    print(f"Calculation node: {submission_node}")
