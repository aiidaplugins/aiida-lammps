from aiida.common.extendeddicts import AttributeDict
from aiida.engine import run_get_node
from aiida.orm import Code, Dict, StructureData
from aiida.plugins import CalculationFactory
import numpy as np

from aiida_lammps.data.potential import EmpiricalPotential

if __name__ == "__main__":

    from aiida import load_profile  # noqa: F401

    load_profile()

    codename = "lammps_optimize@stern"

    ############################
    #  Define input parameters #
    ############################

    # Fe BCC
    cell = [
        [2.848116, 0.000000, 0.000000],
        [0.000000, 2.848116, 0.000000],
        [0.000000, 0.000000, 2.848116],
    ]

    scaled_positions = [
        (0.0000000, 0.0000000, 0.0000000),
        (0.5000000, 0.5000000, 0.5000000),
    ]

    symbols = ["Fe", "Fe"]

    structure = StructureData(cell=cell)
    positions = np.dot(scaled_positions, cell)

    for i, scaled_position in enumerate(scaled_positions):
        structure.append_atom(
            position=np.dot(scaled_position, cell).tolist(), symbols=symbols[i]
        )

    structure.store()

    with open("Fe_mm.eam.fs") as handle:
        eam_data = {"type": "fs", "file_contents": handle.readlines()}

    potential = {"pair_style": "eam", "data": eam_data}

    lammps_machine = {"num_machines": 1, "parallel_env": "mpi*", "tot_num_mpiprocs": 16}

    parameters_opt = {
        "units": "metal",
        "relax": {
            "type": "tri",  # iso/aniso/tri
            "pressure": 0.0,  # bars
            "vmax": 0.000001,  # Angstrom^3
        },
        "minimize": {
            "style": "cg",
            "energy_tolerance": 1.0e-25,  # eV
            "force_tolerance": 1.0e-25,  # eV angstrom
            "max_evaluations": 100000,
            "max_iterations": 50000,
        },
    }

    LammpsOptimizeCalculation = CalculationFactory("lammps.optimize")
    inputs = LammpsOptimizeCalculation.get_builder()

    # Computer options
    options = AttributeDict()
    options.account = ""
    options.qos = ""
    options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "parallel_env": "localmpi",
        "tot_num_mpiprocs": 1,
    }
    # options.queue_name = 'iqtc04.q'
    options.max_wallclock_seconds = 3600
    inputs.metadata.options = options

    # Setup code
    inputs.code = Code.get_from_string(codename)

    # setup nodes
    inputs.structure = structure
    inputs.potential = EmpiricalPotential(
        type=potential["pair_style"], data=potential["data"]
    )

    print(inputs.potential.get_potential_file())
    print(inputs.potential.atom_style)
    print(inputs.potential.default_units)

    inputs.parameters = Dict(dict=parameters_opt)

    # run calculation
    result, node = run_get_node(LammpsOptimizeCalculation, **inputs)
    print("results:", result)
    print("node:", node)

    # submit to deamon
    # submit(LammpsOptimizeCalculation, **inputs)
