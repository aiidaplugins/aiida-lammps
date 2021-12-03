"""Combined MD and Phonopy calculation"""
# Not working with Aiida 1.0
# pylint: disable=no-name-in-module
import numpy as np
from aiida.common.exceptions import InputValidationError
from aiida import orm
from aiida_phonopy.common.raw_parsers import (  # pylint: disable=import-error
    get_force_constants, get_FORCE_SETS_txt, get_poscar_txt,
)

from aiida_lammps.calculations.lammps import BaseLammpsCalculation


def generate_dynaphopy_input(
    parameters_object,
    poscar_name: str = 'POSCAR',
    force_constants_name: str = 'FORCE_CONSTANTS',
    force_sets_filename: str = 'FORCE_SETS',
    use_sets: bool = False,
) -> str:
    """Generates the input needed for the dynaphopy calculation.

    :param parameters_object: input parameters dictionary
    :type parameters_object: dict
    :param poscar_name: name of the POSCAR file, defaults to 'POSCAR'
    :type poscar_name: str, optional
    :param force_constants_name: name of the file with the force constants,
        defaults to 'FORCE_CONSTANTS'
    :type force_constants_name: str, optional
    :param force_sets_filename: name of the file with the force sets,
        defaults to 'FORCE_SETS'
    :type force_sets_filename: str, optional
    :param use_sets: whether or not to use the force sets, defaults to False
    :type use_sets: bool, optional
    :return: dynaphopy input file
    :rtype: str
    """
    parameters = parameters_object.get_dict()
    input_file = f'STRUCTURE FILE POSCAR\n{poscar_name}\n\n'

    if use_sets:
        input_file += f'FORCE SETS\n{force_sets_filename}\n\n'
    else:
        input_file += f'FORCE CONSTANTS\n{force_constants_name}\n\n'

    input_file += 'PRIMITIVE MATRIX\n'
    input_file += f'{np.array(parameters["primitive"])[0, 0]} '
    input_file += f'{np.array(parameters["primitive"])[1, 1]} '
    input_file += f'{np.array(parameters["primitive"])[2, 2]} \n'
    input_file += f'{np.array(parameters["primitive"])[0, 0]} '
    input_file += f'{np.array(parameters["primitive"])[1, 1]} '
    input_file += f'{np.array(parameters["primitive"])[2, 2]} \n'
    input_file += f'{np.array(parameters["primitive"])[0, 0]} '
    input_file += f'{np.array(parameters["primitive"])[1, 1]} '
    input_file += f'{np.array(parameters["primitive"])[2, 2]} \n'
    input_file += '\n'
    input_file += 'SUPERCELL MATRIX PHONOPY\n'
    input_file += f'{np.array(parameters["supercell"])[0, 0]} '
    input_file += f'{np.array(parameters["supercell"])[0, 1]} '
    input_file += f'{np.array(parameters["supercell"])[0, 2]} \n'
    input_file += f'{np.array(parameters["supercell"])[1, 0]} '
    input_file += f'{np.array(parameters["supercell"])[1, 1]} '
    input_file += f'{np.array(parameters["supercell"])[1, 2]} \n'
    input_file += f'{np.array(parameters["supercell"])[2, 0]} '
    input_file += f'{np.array(parameters["supercell"])[2, 1]} '
    input_file += f'{np.array(parameters["supercell"])[2, 2]} \n'
    input_file += '\n'

    return input_file


class CombinateCalculation(BaseLammpsCalculation):
    """Combined MD and Phonopy calculation"""

    _POSCAR_NAME = 'POSCAR'
    _INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'
    _INPUT_FORCE_SETS = 'FORCE_SETS'
    _INPUT_FILE_NAME_DYNA = 'input_dynaphopy'
    _OUTPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS_OUT'
    _OUTPUT_QUASIPARTICLES = 'quasiparticles_data.yaml'
    _OUTPUT_FILE_NAME = 'OUTPUT'

    # self._retrieve_list = [self._OUTPUT_QUASIPARTICLES,
    # self._OUTPUT_FORCE_CONSTANTS, self._OUTPUT_FILE_NAME]

    @classmethod
    def define(cls, spec):
        super(CombinateCalculation, cls).define(spec)
        spec.input(
            'metadata.options.parser_name',
            valid_type=str,
            default='dynaphopy',
        )
        spec.input(
            'parameters_dynaphopy',
            valid_type=orm.Dict,
            help='dynaphopy parameters',
        )
        spec.input(
            'force_constants',
            valid_type=orm.ArrayData,
            help='harmonic force constants',
        )
        spec.input(
            'force_sets',
            valid_type=orm.ArrayData,
            help='phonopy force sets',
        )

        # spec.input('settings', valid_type=str, default='lammps.optimize')

    @staticmethod
    def create_main_input_content(
        parameter_data,
        potential_data,
        structure_data,
        structure_filename,
        trajectory_filename,
        system_filename,
        restart_filename,
    ):
        # pylint: disable=too-many-arguments, arguments-differ
        random_number = np.random.randint(10000000)

        lammps_input_file = f'units           {potential_data.default_units}\n'
        lammps_input_file += 'boundary        p p p\n'
        lammps_input_file += 'box tilt large\n'
        lammps_input_file += f'atom_style      {potential_data.atom_style}\n'
        lammps_input_file += f'read_data       {structure_filename}\n'

        lammps_input_file += potential_data.get_input_lines(structure_data)

        lammps_input_file += 'neighbor        0.3 bin\n'
        lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

        lammps_input_file += 'velocity        all create '
        lammps_input_file += f'{parameter_data.dict.temperature} {random_number} '
        lammps_input_file += 'dist gaussian mom yes\n'
        lammps_input_file += f'velocity        all scale {parameter_data.dict.temperature}\n'

        lammps_input_file += 'fix             int all nvt temp '
        lammps_input_file += f'{parameter_data.dict.temperature} '
        lammps_input_file += f'{parameter_data.dict.temperature} '
        lammps_input_file += f'{parameter_data.dict.thermostat_variable}\n'

        return lammps_input_file

    def prepare_extra_files(self, tempfolder, potential_object):
        # pylint: disable=too-many-locals
        if 'fore_constants' in self.inputs:
            force_constants = self.inputs.force_constants
        else:
            force_constants = None

        if 'fore_constants' in self.inputs:
            force_sets = self.inputs.force_sets
        else:
            force_sets = None

        cell_txt = get_poscar_txt(self.inputs.structure)

        cell_filename = tempfolder(self._POSCAR_NAME)
        with open(cell_filename, 'w') as infile:
            infile.write(cell_txt)

        if force_constants is not None:
            force_constants_txt = get_force_constants(force_constants)
            force_constants_filename = tempfolder.get_abs_path(
                self._INPUT_FORCE_CONSTANTS)
            with open(force_constants_filename, 'w') as infile:
                infile.write(force_constants_txt)

        elif force_sets is not None:
            force_sets_txt = get_FORCE_SETS_txt(force_sets)
            force_sets_filename = tempfolder.get_abs_path(
                self._INPUT_FORCE_SETS)
            with open(force_sets_filename, 'w') as infile:
                infile.write(force_sets_txt)
        else:
            raise InputValidationError(
                'no force_sets nor force_constants are specified for this calculation'
            )

        try:
            parameters_data_dynaphopy = orm.Dict.pop(  # pylint: disable=no-member
                self.get_linkname('parameters_dynaphopy'))
        except KeyError as key_error:
            raise InputValidationError(
                'No dynaphopy parameters specified for this calculation'
            ) from key_error

        parameters_dynaphopy_txt = generate_dynaphopy_input(
            parameters_data_dynaphopy,
            poscar_name=self._POSCAR_NAME,
            force_constants_name=self._INPUT_FORCE_CONSTANTS,
            force_sets_filename=self._INPUT_FORCE_SETS,
            use_sets=force_sets is not None,
        )

        dynaphopy_filename = tempfolder.get_abs_path(
            self._INPUT_FILE_NAME_DYNA)
        with open(dynaphopy_filename, 'w') as infile:
            infile.write(parameters_dynaphopy_txt)

        md_supercell = parameters_data_dynaphopy.dict.md_supercell

        time_step = self._parameters_data.dict.timestep
        equilibrium_time = self._parameters_data.dict.equilibrium_steps * time_step
        total_time = self._parameters_data.dict.total_steps * time_step

        self._cmdline_params = [
            self._INPUT_FILE_NAME_DYNA,
            '--run_lammps',
            self._INPUT_FILE_NAME,
            f'{total_time}',
            f'{time_step}',
            f'{equilibrium_time}',
            '--dim',
            f'{md_supercell[0]}',
            f'{md_supercell[1]}',
            f'{md_supercell[2]}',
            '--silent',
            '-sfc',
            self._OUTPUT_FORCE_CONSTANTS,
            '-thm',  # '--resolution 0.01',
            '-psm',
            '2',
            '--normalize_dos',
            '-sdata',
            '--velocity_only',
            '--temperature',
            '{}'.format(self._parameters_data.dict.temperature),
        ]

        if 'md_commensurate' in parameters_data_dynaphopy.get_dict():
            if parameters_data_dynaphopy.dict.md_commensurate:
                self._cmdline_params.append('--MD_commensurate')
