# This calculation requires also phonopy plugin to work

from aiida.common.extendeddicts import AttributeDict
from aiida.engine import run_get_node
from aiida.orm import Code, Dict, StructureData
from aiida.plugins import CalculationFactory
import numpy as np

if __name__ == "__main__":

    from aiida import load_profile  # noqa: F401

    load_profile()

    codename = "lammps_combinate@stern"

    ############################
    #  Define input parameters #
    ############################

    a = 5.404
    cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]

    symbols = ["Si"] * 8
    scaled_positions = [
        (0.875, 0.875, 0.875),
        (0.875, 0.375, 0.375),
        (0.375, 0.875, 0.375),
        (0.375, 0.375, 0.875),
        (0.125, 0.125, 0.125),
        (0.125, 0.625, 0.625),
        (0.625, 0.125, 0.625),
        (0.625, 0.625, 0.125),
    ]

    structure = StructureData(cell=cell)
    positions = np.dot(scaled_positions, cell)

    for i, scaled_position in enumerate(scaled_positions):
        structure.append_atom(
            position=np.dot(scaled_position, cell).tolist(), symbols=symbols[i]
        )

    structure.store()

    # Silicon(C) Tersoff
    tersoff_si = {
        "Si  Si  Si ": "3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8"
    }

    potential = {"pair_style": "tersoff", "data": tersoff_si}

    dynaphopy_parameters = {
        "supercell": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "primitive": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        "mesh": [40, 40, 40],
        "md_commensurate": False,
        "md_supercell": [2, 2, 2],
    }

    from aiida.orm import load_node

    force_constants = load_node(
        38
    )  # Loads node that contains the harmonic force constants (Array data)

    machine = {"num_machines": 1, "parallel_env": "mpi*", "tot_num_mpiprocs": 16}

    parameters_md = {
        "timestep": 0.001,
        "temperature": 300,
        "thermostat_variable": 0.5,
        "equilibrium_steps": 2000,
        "total_steps": 2000,
        "dump_rate": 1,
    }

    CombinateCalculation = CalculationFactory("lammps.force")
    inputs = CombinateCalculation.get_builder()

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
    inputs.potential = Dict(dict=potential)
    inputs.force_constants(force_constants)
    inputs.parameters_dynaphopy(Dict(dict=dynaphopy_parameters))

    # run calculation
    result, node = run_get_node(CombinateCalculation, **inputs)
    print("results:", result)
    print("node:", node)

    # submit to deamon
    # submit(LammpsOptimizeCalculation, **inputs)
