from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

import numpy as np


StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

codename = 'lammps_md@boston'

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

# Silicon(C) Tersoff
tersoff_si = {'Si  Si  Si ': '3.0 1.0 1.7322 1.0039e5 16.218 -0.59826 0.78734 1.0999e-6  1.7322  471.18  2.85  0.15  2.4799  1830.8'}


potential ={'pair_style': 'tersoff',
                          'data': tersoff_si}

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}


parameters_md = {'timestep': 0.001,
                 'temperature' : 300,
                 'thermostat_variable': 0.5,
                 'equilibrium_steps': 100,
                 'total_steps': 2000,
                 'dump_rate': 1}


code = Code.get_from_string(codename)

calc = code.new_calc(max_wallclock_seconds=3600,
                     resources=lammps_machine)

calc.label = "test lammps calculation"
calc.description = "A much longer description"
calc.use_code(code)
calc.use_structure(structure)
calc.use_potential(ParameterData(dict=potential))

calc.use_parameters(ParameterData(dict=parameters_md))


test_only = False

if test_only:  # It will not be submitted
    import os
    subfolder, script_filename = calc.submit_test()
    print "Test_submit for calculation (uuid='{}')".format(calc.uuid)
    print "Submit file in {}".format(os.path.join(
                                     os.path.relpath(subfolder.abspath),
                                     script_filename))
else:
    calc.store_all()
    print "created calculation; calc=Calculation(uuid='{}') # ID={}".format(
        calc.uuid, calc.dbnode.pk)
    calc.submit()
    print "submitted calculation; calc=Calculation(uuid='{}') # ID={}".format(
        calc.uuid, calc.dbnode.pk)
