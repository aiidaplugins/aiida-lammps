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


def generate_dynaphopy_input(parameters_object, poscar_name='POSCAR',
                             force_constants_name='FORCE_CONSTANTS',
                             force_sets_filename='FORCE_SETS',
                             use_sets=False):

    parameters = parameters_object.get_dict()
    input_file = 'STRUCTURE FILE POSCAR\n{}\n\n'.format(poscar_name)

    if use_sets:
        input_file += 'FORCE SETS\n{}\n\n'.format(force_sets_filename)
    else:
        input_file += 'FORCE CONSTANTS\n{}\n\n'.format(force_constants_name)

    input_file += 'PRIMITIVE MATRIX\n'
    input_file += '{} {} {} \n'.format(*np.array(parameters['primitive'])[0])
    input_file += '{} {} {} \n'.format(*np.array(parameters['primitive'])[1])
    input_file += '{} {} {} \n'.format(*np.array(parameters['primitive'])[2])
    input_file += '\n'
    input_file += 'SUPERCELL MATRIX PHONOPY\n'
    input_file += '{} {} {} \n'.format(*np.array(parameters['supercell'])[0])
    input_file += '{} {} {} \n'.format(*np.array(parameters['supercell'])[1])
    input_file += '{} {} {} \n'.format(*np.array(parameters['supercell'])[2])
    input_file += '\n'

    return input_file


def generate_LAMMPS_input(parameters,
                          potential_obj,
                          structure_file='potential.pot',
                          trajectory_file=None):

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

    _POSCAR_NAME = 'POSCAR'
    _INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'
    _INPUT_FORCE_SETS = 'FORCE_SETS'
    _INPUT_FILE_NAME_DYNA = 'input_dynaphopy'
    _OUTPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS_OUT'
    _OUTPUT_QUASIPARTICLES = 'quasiparticles_data.yaml'
    _OUTPUT_TRAJECTORY_FILE_NAME = None
    _OUTPUT_FILE_NAME = 'OUTPUT'


    def _init_internal_params(self):
        super(CombinateCalculation, self)._init_internal_params()

        self._default_parser = 'dynaphopy'

        self._retrieve_list = [self._OUTPUT_QUASIPARTICLES, self._OUTPUT_FORCE_CONSTANTS, self._OUTPUT_FILE_NAME]
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
               'linkname': 'parameters_dynaphopy',
               'docstring': ("Node that specifies the dynaphopy input data"),
        }
        retdict['force_constants'] = {
               'valid_types': ArrayData,
               'additional_parameter': None,
               'linkname': 'force_constants',
               'docstring': ("Node that specified the force constants"),
        }
        retdict['force_sets'] = {
               'valid_types': ArrayData,
               'additional_parameter': None,
               'linkname': 'force_sets',
               'docstring': ("Node that specified the force constants"),
        }

        return retdict

    def _create_additional_files(self, tempfolder, inputdict):

        force_constants = inputdict.pop(self.get_linkname('force_constants'), None)
        force_sets = inputdict.pop(self.get_linkname('force_sets'), None)

        cell_txt = structure_to_poscar(self._structure)
        cell_filename = tempfolder.get_abs_path(self._POSCAR_NAME)
        with open(cell_filename, 'w') as infile:
            infile.write(cell_txt)

        if force_constants is not None:
            force_constants_txt = get_FORCE_CONSTANTS_txt(force_constants)
            force_constants_filename = tempfolder.get_abs_path(self._INPUT_FORCE_CONSTANTS)
            with open(force_constants_filename, 'w') as infile:
                infile.write(force_constants_txt)

        elif force_sets is not None:
            force_sets_txt = get_FORCE_SETS_txt(force_sets)
            force_sets_filename = tempfolder.get_abs_path(self._INPUT_FORCE_SETS)
            with open(force_sets_filename, 'w') as infile:
                infile.write(force_sets_txt)
        else:
            raise InputValidationError("no force_sets nor force_constants are specified for this calculation")

        try:
            parameters_data_dynaphopy = inputdict.pop(self.get_linkname('parameters_dynaphopy'))
        except KeyError:
            raise InputValidationError("No dynaphopy parameters specified for this calculation")

        parameters_dynaphopy_txt = generate_dynaphopy_input(parameters_data_dynaphopy,
                                                            poscar_name=self._POSCAR_NAME,
                                                            force_constants_name=self._INPUT_FORCE_CONSTANTS,
                                                            force_sets_filename=self._INPUT_FORCE_SETS,
                                                            use_sets=force_sets is not None)

        dynaphopy_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME_DYNA)
        with open(dynaphopy_filename, 'w') as infile:
            infile.write(parameters_dynaphopy_txt)

        md_supercell = parameters_data_dynaphopy.dict.md_supercell

        time_step = self._parameters_data.dict.timestep
        equilibrium_time = self._parameters_data.dict.equilibrium_steps * time_step
        total_time = self._parameters_data.dict.total_steps * time_step

        self._cmdline_params = [self._INPUT_FILE_NAME_DYNA,
                                '--run_lammps', self._INPUT_FILE_NAME,
                                '{}'.format(total_time), '{}'.format(time_step), '{}'.format(equilibrium_time),
                                '--dim',
                                '{}'.format(md_supercell[0]), '{}'.format(md_supercell[1]), '{}'.format(md_supercell[2]),
                                '--silent', '-sfc', self._OUTPUT_FORCE_CONSTANTS, '-thm',  # '--resolution 0.01',
                                '-psm','2', '--normalize_dos', '-sdata', '--velocity_only',
                                '--temperature', '{}'.format(self._parameters_data.dict.temperature)]

        if 'md_commensurate' in parameters_data_dynaphopy.get_dict():
            if parameters_data_dynaphopy.dict.md_commensurate:
                self._cmdline_params.append('--MD_commensurate')

        self._stdout_name = self._OUTPUT_FILE_NAME
