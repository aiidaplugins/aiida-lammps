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

    # GaN
    cell = [
        [3.1900000572, 0, 0],
        [-1.5950000286, 2.762621076, 0],
        [0.0, 0, 5.1890001297],
    ]

    scaled_positions = [
        (0.6666669, 0.3333334, 0.0000000),
        (0.3333331, 0.6666663, 0.5000000),
        (0.6666669, 0.3333334, 0.3750000),
        (0.3333331, 0.6666663, 0.8750000),
    ]

    symbols = ["Ga", "Ga", "N", "N"]

    structure = StructureData(cell=cell)
    positions = np.dot(scaled_positions, cell)

    for i, scaled_position in enumerate(scaled_positions):
        structure.append_atom(
            position=np.dot(scaled_position, cell).tolist(), symbols=symbols[i]
        )

    structure.store()

    # GaN Tersoff
    tersoff_gan = {
        "Ga Ga Ga": "1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199",
        "N  N  N": "1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77",
        "Ga Ga N": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000",
        "Ga N  N": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44",
        "N  Ga Ga": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44",
        "N  Ga N ": "1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000",
        "N  N  Ga": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000",
        "Ga N  Ga": "1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000",
    }

    # Silicon(C) Tersoff
    # tersoff_si = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}

    potential = {"pair_style": "tersoff", "data": tersoff_gan}

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
            "max_evaluations": 1000000,
            "max_iterations": 500000,
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

    inputs.parameters = Dict(dict=parameters_opt)

    print(inputs.potential.get_potential_file())
    print(inputs.potential.atom_style)
    print(inputs.potential.default_units)

    # run calculation
    result, node = run_get_node(LammpsOptimizeCalculation, **inputs)
    print("results:", result)
    print("node:", node)

    # submit to deamon
    # submit(LammpsOptimizeCalculation, **inputs)
