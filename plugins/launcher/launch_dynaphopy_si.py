from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

import numpy as np


StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

codename = 'dynaphopy@stern'

############################
#  Define input parameters #
############################

a = 5.404
cell = [[a, 0, 0],
        [0, a, 0],
        [0, 0, a]]

symbols=['Si'] * 8
scaled_positions = [(0.875,  0.875,  0.875),
                    (0.875,  0.375,  0.375),
                    (0.375,  0.875,  0.375),
                    (0.375,  0.375,  0.875),
                    (0.125,  0.125,  0.125),
                    (0.125,  0.625,  0.625),
                    (0.625,  0.125,  0.625),
                    (0.625,  0.625,  0.125)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])

structure.store()

dynaphopy_parameters ={'supercell': [[2, 0, 0],
                                     [0, 2, 0],
                                     [0, 0, 2]],
                       'primitive':  [[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0]],
                       'mesh': [40, 40, 40],
                       'md_commensurate': True,
                       'temperature': 300}  # Temperature can be omitted (If ommited calculated from Max.-Boltz.)


dynaphopy_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


from aiida.orm import load_node
force_constants = load_node(20569)  # Loads node that contains the harmonic force constants (Array data)
trajectory = load_node(20528)       # Loads node that constains the MD trajectory (TrajectoryData)


codename = codename
code = Code.get_from_string(codename)
calc = code.new_calc(max_wallclock_seconds=3600,
                     resources=dynaphopy_machine)

calc.label = "test dynaphopy calculation"
calc.description = "A much longer description"

calc.use_code(code)

calc.use_structure(structure)
calc.use_parameters(ParameterData(dict=dynaphopy_parameters))
calc.use_force_constants(force_constants)
calc.use_trajectory(trajectory)

calc.store_all()


calc.submit()
print "submitted calculation with PK={}".format(calc.dbnode.pk)
