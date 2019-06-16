# Not working with Aiida 1.0

from aiida.common.exceptions import InputValidationError
from aiida_phonopy.common.raw_parsers import get_FORCE_CONSTANTS_txt, get_poscar_txt, get_FORCE_SETS_txt
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida.orm import Dict, ArrayData
import numpy as np
import six



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

    lammps_input_file = 'units           {0}\n'.format(potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(potential_obj.atom_style)
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    lammps_input_file += 'velocity        all create {0} {1} dist gaussian mom yes\n'.format(parameters.dict.temperature, random_number)
    lammps_input_file += 'velocity        all scale {}\n'.format(parameters.dict.temperature)

    lammps_input_file += 'fix             int all nvt temp {0} {0} {1}\n'.format(parameters.dict.temperature, parameters.dict.thermostat_variable)

    return lammps_input_file


class CombinateCalculation(BaseLammpsCalculation):

    _POSCAR_NAME = 'POSCAR'
    _INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'
    _INPUT_FORCE_SETS = 'FORCE_SETS'
    _INPUT_FILE_NAME_DYNA = 'input_dynaphopy'
    _OUTPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS_OUT'
    _OUTPUT_QUASIPARTICLES = 'quasiparticles_data.yaml'
    _OUTPUT_FILE_NAME = 'OUTPUT'
    _generate_input_function = generate_LAMMPS_input

    #self._retrieve_list = [self._OUTPUT_QUASIPARTICLES, self._OUTPUT_FORCE_CONSTANTS, self._OUTPUT_FILE_NAME]

    @classmethod
    def define(cls, spec):
        super(CombinateCalculation, cls).define(spec)
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='dynaphopy')
        spec.input('parameters_dynaphopy', valid_type=Dict, help='dynaphopy parameters')
        spec.input('force_constants', valid_type=ArrayData, help='harmonic force constants')
        spec.input('force_sets', valid_type=ArrayData, help='phonopy force sets')

        # spec.input('settings', valid_type=six.string_types, default='lammps.optimize')

    def prepare_extra_files(self, tempfolder, potential_object):

        if 'fore_constants' in self.inputs:
            force_constants = self.inputs.force_constants
        else:
            force_constants = None

        if 'fore_constants' in self.inputs:
            force_sets = self.inputs.force_sets
        else:
            force_sets = None

        cell_txt = get_poscar_txt(self.inputs.structure)

        cell_filename = tempfolder (self._POSCAR_NAME)
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
            parameters_data_dynaphopy = Dict.pop(self.get_linkname('parameters_dynaphopy'))
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
