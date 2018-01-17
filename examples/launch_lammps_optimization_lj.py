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

cell = [[ 3.987594, 0.000000, 0.000000],
        [-1.993797, 3.453358, 0.000000],
        [ 0.000000, 0.000000, 6.538394]]

symbols=['Ar'] * 2
scaled_positions = [(0.33333,  0.66666,  0.25000),
                    (0.66667,  0.33333,  0.75000)]

structure = StructureData(cell=cell)
positions = np.dot(scaled_positions, cell)

for i, scaled_position in enumerate(scaled_positions):
    structure.append_atom(position=np.dot(scaled_position, cell).tolist(),
                          symbols=symbols[i])

structure.store()

# Example LJ parameters for Argon. These may not be accurate at all
potential ={'pair_style': 'lennard_jones',
            #                 epsilon,  sigma, cutoff
            'data': {'1  1':  '0.01029   3.4    2.5',
                     #'2  2':   '1.0      1.0    2.5',
                     #'1  2':   '1.0      1.0    2.5'
                     }}

lammps_machine = {
    'num_machines': 1,
    'parallel_env': 'mpi*',
    'tot_num_mpiprocs': 16}

parameters_opt = {'relaxation': 'tri',  # iso/aniso/tri
                  'pressure': 0.0,  # kbars
                  'vmax': 0.000001,  # Angstrom^3
                  'energy_tolerance': 1.0e-25,  # eV
                  'force_tolerance': 1.0e-25,  # eV angstrom
                  'max_evaluations': 1000000,
                  'max_iterations': 500000}

code = Code.get_from_string(codename)

calc = code.new_calc(max_wallclock_seconds=3600,
                     resources=lammps_machine)

calc.label = "test lammps calculation"
calc.description = "A much longer description"
calc.use_code(code)
calc.use_structure(structure)
calc.use_potential(ParameterData(dict=potential))

calc.use_parameters(ParameterData(dict=parameters_opt))

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
