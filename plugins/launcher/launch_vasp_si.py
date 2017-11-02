from aiida import load_dbenv
load_dbenv()
from aiida.orm import Code, DataFactory

from pymatgen.io import vasp as vaspio

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

import numpy as np

codename = 'vasp541mpi@stern'

############################
#  Define input parameters #
############################

a = 5.40400123456789
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

# VASP input parameters
incar_dict = {
    'NELMIN' : 5,
    'NELM'   : 100,
    'ENCUT'  : 400,
    'ALGO'   : 38,
    'ISMEAR' : 0,
    'SIGMA'  : 0.01,
    'GGA'    : 'PS'
}

pseudo_dict = {'functional': 'PBE',
               'symbols': np.unique(symbols).tolist()}  # Can be a list of strings with the potwpaw pseudopotentials folder name. Ex: ['Si']

# supported_style_modes: "Gamma", "Monkhorst", "Automatic", "Line_mode", "Cartesian" & "Reciprocal" (pymatgen)
kpoints_dict = {'style' : 'Monkhorst',
                'points': [2, 2, 2],
                'shift' : [0.0, 0.0, 0.0]}

# Cluster information
machine_dict = {
    'num_machines': 1,
    'parallel_env':'mpi*',
    'tot_num_mpiprocs' : 16}

test_only = True


###################
# Set calculation #
###################

code = Code.get_from_string(codename)
calc = code.new_calc(
    max_wallclock_seconds=3600,
    resources=machine_dict
)
calc.set_withmpi(True)

calc.label = 'VASP'
calc.label = 'Silicon VASP'
calc.description = "This is an example calculation of VASP"

# POSCAR
calc.use_structure(structure)

incar = vaspio.Incar(incar_dict)
calc.use_incar(ParameterData(dict=incar.as_dict()))

# KPOINTS
kpoints = kpoints_dict

kpoints = vaspio.Kpoints(comment='aiida generated',
                         style=kpoints['style'],
                         kpts=(kpoints['points'],), kpts_shift=kpoints['shift'])

calc.use_kpoints(ParameterData(dict=kpoints.as_dict()))

# POTCAR
pseudo = pseudo_dict
potcar = vaspio.Potcar(symbols=pseudo['symbols'],
                       functional=pseudo['functional'])
calc.use_potcar(ParameterData(dict=potcar.as_dict()))

# Define parsers to use
settings = {'PARSER_INSTRUCTIONS': []}
pinstr = settings['PARSER_INSTRUCTIONS']
pinstr += [{
    'instr': 'array_data_parser',
    'type': 'data',
    'params': {}},
    {
    'instr': 'output_parameters',
    'type': 'data',
    'params': {}},
    {
    'instr': 'dummy_error_parser',
    'type': 'error',
    'params': {}},
    {
    'instr': 'default_structure_parser',
    'type': 'structure',
    'params': {}}
]

# additional files to return
settings.setdefault(
    'ADDITIONAL_RETRIEVE_LIST', [
        'OSZICAR',
        'CONTCAR',
        'OUTCAR',
#        'vasprun.xml'
        ]
)
calc.use_settings(ParameterData(dict=settings))

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
