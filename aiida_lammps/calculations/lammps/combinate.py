from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty

from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.calculations.lammps.potentials import LammpsPotential
import numpy as np


def get_FORCE_CONSTANTS_txt(force_constants):

    force_constants = force_constants.get_array('force_constants')

    fc_shape = force_constants.shape
    fc_txt = "%4d\n" % (fc_shape[0])
    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            fc_txt += "%4d%4d\n" % (i+1, j+1)
            for vec in force_constants[i][j]:
                fc_txt +=("%22.15f"*3 + "\n") % tuple(vec)

    return fc_txt

def get_FORCE_SETS_txt(data_sets_object):
    data_sets = data_sets_object.get_force_sets()

    displacements = data_sets['first_atoms']
    forces = [x['forces'] for x in data_sets['first_atoms']]

    # Write FORCE_SETS
    force_sets_txt = "%-5d\n" % data_sets['natom']
    force_sets_txt += "%-5d\n" % len(displacements)
    for count, disp in enumerate(displacements):
        force_sets_txt += "\n%-5d\n" % (disp['number'] + 1)
        force_sets_txt += "%20.16f %20.16f %20.16f\n" % (tuple(disp['displacement']))

        for f in forces[count]:
            force_sets_txt += "%15.10f %15.10f %15.10f\n" % (tuple(f))
    return force_sets_txt

def structure_to_poscar(structure):

    atom_type_unique = np.unique([site.kind_name for site in structure.sites], return_index=True)[1]
    labels = np.diff(np.append(atom_type_unique, [len(structure.sites)]))

    poscar = ' '.join(np.unique([site.kind_name for site in structure.sites]))
    poscar += '\n1.0\n'
    cell = structure.cell
    for row in cell:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)
    poscar += ' '.join(np.unique([site.kind_name for site in structure.sites]))+'\n'
    poscar += ' '.join(np.array(labels, dtype=str))+'\n'
    poscar += 'Cartesian\n'
    for site in structure.sites:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*site.position)

    return poscar


def generate_LAMMPS_structure(structure):
    import numpy as np

    types = [site.kind_name for site in structure.sites]

    type_index_unique = np.unique(types, return_index=True)[1]
    count_index_unique = np.diff(np.append(type_index_unique, [len(types)]))

    atom_index = []
    for i, index in enumerate(count_index_unique):
        atom_index += [i for j in range(index)]

    masses = [site.mass for site in structure.kinds]
    positions = [site.position for site in structure.sites]

    number_of_atoms = len(positions)

    lammps_data_file = 'Generated using dynaphopy\n\n'
    lammps_data_file += '{0} atoms\n\n'.format(number_of_atoms)
    lammps_data_file += '{0} atom types\n\n'.format(len(masses))

    cell = np.array(structure.cell)

    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])

    alpha = np.arccos(np.dot(cell[1], cell[2])/(c*b))
    gamma = np.arccos(np.dot(cell[1], cell[0])/(a*b))
    beta = np.arccos(np.dot(cell[2], cell[0])/(a*c))

    xhi = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    yhi = np.sqrt(pow(b,2)- pow(xy,2))
    yz = (b*c*np.cos(alpha)-xy * xz)/yhi
    zhi = np.sqrt(pow(c,2)-pow(xz,2)-pow(yz,2))

    xhi = xhi + max(0,0, xy, xz, xy+xz)
    yhi = yhi + max(0,0, yz)

    lammps_data_file += '\n{0:20.10f} {1:20.10f} xlo xhi\n'.format(0, xhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} ylo yhi\n'.format(0, yhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} zlo zhi\n'.format(0, zhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f} xy xz yz\n\n'.format(xy, xz, yz)

    lammps_data_file += 'Masses\n\n'

    for i, mass in enumerate(masses):
        lammps_data_file += '{0} {1:20.10f} \n'.format(i+1, mass)


    lammps_data_file += '\nAtoms\n\n'
    for i, row in enumerate(positions):
        lammps_data_file += '{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(i+1, atom_index[i]+1, row[0],row[1],row[2])

    return lammps_data_file


def generate_LAMMPS_input(parameters,
                          potential_obj,
                          structure_file='potential.pot'):

    random_number = np.random.randint(10000000)

    names_str = ' '.join(potential_obj._names)

    lammps_input_file = 'units           metal\n'
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      atomic\n'
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    lammps_input_file += 'velocity        all create {0} {1} dist gaussian mom yes\n'.format(parameters.dict.temperature, random_number)
    lammps_input_file += 'velocity        all scale {}\n'.format(parameters.dict.temperature)

    lammps_input_file += 'fix             int all nvt temp {0} {0} {1}\n'.format(parameters.dict.temperature, parameters.dict.thermostat_variable)

    return lammps_input_file


class CombinateCalculation(BaseLammpsCalculation, JobCalculation):

    _INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'
    _INPUT_FORCE_SETS = 'FORCE_SETS'
    _INPUT_FILE_NAME_DYNA = 'input_dynaphopy'
    _OUTPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS_OUT'
    _OUTPUT_FILE_NAME = 'OUTPUT'
    _OUTPUT_QUASIPARTICLES = 'quasiparticles_data.yaml'

    def _init_internal_params(self):
        super(CombinateCalculation, self)._init_internal_params()

        self._default_parser = 'dynaphopy'

        self._retrieve_list = [self._OUTPUT_QUASIPARTICLES, self._OUTPUT_FILE_NAME, self._OUTPUT_FORCE_CONSTANTS]
        self._generate_input_function = generate_LAMMPS_input

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods
        retdict.update(BaseLammpsCalculation._baseclass_use_methods)

        retdict['parameters_dynaphopy'] = {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters',
               'docstring': ("Node that specifies the dynaphopy input data"),
        }
        retdict['force_constants'] = {
               #'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters',
               'docstring': ("Node that specified the force constants"),
        }
        retdict['force_sets'] = {
               #'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters',
               'docstring': ("Node that specified the force constants"),
        }

        return retdict

    def _create_additional_files(self, tempfolder, inputdict):

        data_force = inputdict.pop(self.get_linkname('force_sets'), None)
        force_constants = inputdict.pop(self.get_linkname('force_constants'), None)

        if force_constants is not None:
            force_constants_txt = get_FORCE_CONSTANTS_txt(force_constants)
            force_constants_filename = tempfolder.get_abs_path(self._INPUT_FORCE_CONSTANTS)
            with open(force_constants_filename, 'w') as infile:
                infile.write(force_constants_txt)

        elif data_force is not None:
            force_sets_txt = get_FORCE_SETS_txt(data_force)
            force_sets_filename = tempfolder.get_abs_path(self._INPUT_FORCE_SETS)
            with open(force_sets_filename, 'w') as infile:
                infile.write(force_sets_txt)
        else:
             raise InputValidationError("no force_sets nor force_constants are specified for this calculation")

        try:
            parameters_data_dynaphopy = inputdict.pop(self.get_linkname('parameters_dynaphopy'))
        except KeyError:
            raise InputValidationError("No dynaphopy parameters specified for this calculation")

        time_step = parameters_data_dynaphopy.dict.timestep
        equilibrium_time = parameters_data_dynaphopy.dict.equilibrium_steps * time_step
        total_time = parameters_data_dynaphopy.dict.total_steps * time_step
        supercell_shape = parameters_data_dynaphopy.dict.supercell_shape

        self._cmdline_params = [self._INPUT_FILE_NAME_DYNA,
                                '--run_lammps', self._INPUT_FILE_NAME,
                                '{}'.format(total_time), '{}'.format(time_step), '{}'.format(equilibrium_time),
                                '--dim', '{}'.format(supercell_shape.dict.shape[0]),
                                '{}'.format(supercell_shape.dict.shape[1]),
                                '{}'.format(supercell_shape.dict.shape[2]),
                                '--silent', '-sfc', self._OUTPUT_FORCE_CONSTANTS, '-thm',  # '--resolution 0.01',
                                '-psm','2', '--normalize_dos', '-sdata', '--velocity_only']

        if 'temperature' in parameters_data_dynaphopy.get_dict():
            self._cmdline_params.append('--temperature')
            self._cmdline_params.append('{}'.format(parameters_data_dynaphopy.dict.temperature))

        if 'md_commensurate' in parameters_data_dynaphopy.get_dict():
            if parameters_data_dynaphopy.dict.md_commensurate:
                self._cmdline_params.append('--MD_commensurate')
