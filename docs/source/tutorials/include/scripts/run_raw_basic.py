"""Run a LAMMPS calculation with additional input files

The example input script is taken from https://www.lammps.org/inputs/in.rhodo.txt and is an example benchmark script for
the official benchmarks of LAMMPS. It is a simple MD simulation of a protein. It requires an additional input file in
the working directory ``data.rhodo``. This example shows how to add such additional input files.
"""
import io
import textwrap

from aiida.engine import run
from aiida.orm import SinglefileData, load_code
from aiida.plugins import CalculationFactory

script = SinglefileData(
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
data = SinglefileData(
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

builder = CalculationFactory("lammps.raw").get_builder()
builder.code = load_code("lammps@localhost")
builder.script = script
builder.files = {"data": data}
builder.filenames = {"data": "data.rhodo"}
builder.metadata.options = {"resources": {"num_machines": 1}}
results, node = run.run_get_node(builder)

print(
    f"Calculation: {node.process_class}<{node.pk}> {node.process_state.value} [{node.exit_status}]"
)
print(f"Results: {results}")
assert node.is_finished_ok, f"{node} failed: [{node.exit_status}] {node.exit_message}"
