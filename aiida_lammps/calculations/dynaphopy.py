"""Base dynaphopy calculation"""
# pylint: disable=too-many-instance-attributes, abstract-method
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError
from aiida.common.utils import classproperty
from aiida.engine import CalcJob
from aiida.orm import ArrayData, StructureData, TrajectoryData
from aiida_phonopy.common.raw_parsers import get_force_constants, get_poscar_txt  # pylint: disable=no-name-in-module, import-error

from aiida_lammps.common.generate_input_files import (
    get_trajectory_txt,
    parameters_to_input_file,
)


class DynaphopyCalculation(CalcJob):
    """
    A basic plugin for calculating force constants using Phonopy.

    Requirement: the node should be able to import phonopy
    """
    def _init_internal_params(self):
        super(DynaphopyCalculation, self)._init_internal_params()

        self._INPUT_FILE_NAME = 'input_dynaphopy'  # pylint: disable=invalid-name, attribute-defined-outside-init
        self._INPUT_TRAJECTORY = 'trajectory'  # pylint: disable=invalid-name, attribute-defined-outside-init
        self._INPUT_CELL = 'POSCAR'  # pylint: disable=invalid-name, attribute-defined-outside-init
        self._INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'  # pylint: disable=invalid-name, attribute-defined-outside-init

        self._OUTPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS_OUT'  # pylint: disable=invalid-name, attribute-defined-outside-init
        self._OUTPUT_FILE_NAME = 'OUTPUT'  # pylint: disable=invalid-name, attribute-defined-outside-init
        self._OUTPUT_QUASIPARTICLES = 'quasiparticles_data.yaml'  # pylint: disable=invalid-name, attribute-defined-outside-init

        self._default_parser = 'dynaphopy'  # pylint: disable=attribute-defined-outside-init

    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the namelists class.
        """
        # pylint: disable=no-self-argument, no-self-use
        retdict = JobCalculation._use_methods  # pylint: disable=protected-access, undefined-variable
        retdict.update({
            'parameters': {
                'valid_types':
                ParameterData,  # pylint: disable=undefined-variable
                'additional_parameter':
                None,
                'linkname':
                'parameters',
                'docstring': ('Use a node that specifies the dynaphopy input '
                              'for the namelists'),
            },
            'trajectory': {
                'valid_types':
                TrajectoryData,
                'additional_parameter':
                None,
                'linkname':
                'trajectory',
                'docstring': ('Use a node that specifies the trajectory data '
                              'for the namelists'),
            },
            'force_constants': {
                'valid_types':
                ArrayData,
                'additional_parameter':
                None,
                'linkname':
                'force_constants',
                'docstring': ('Use a node that specifies the force_constants '
                              'for the namelists'),
            },
            'structure': {
                'valid_types': StructureData,
                'additional_parameter': None,
                'linkname': 'structure',
                'docstring': 'Use a node for the structure',
            },
        })
        return retdict

    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """
        # pylint: disable=too-many-locals, too-many-statements
        try:
            parameters_data = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            pass
            # raise InputValidationError("No parameters specified for this "
            #                           "calculation")
        if not isinstance(parameters_data, ParameterData):  # pylint: disable=undefined-variable
            raise InputValidationError('parameters is not of type '
                                       'ParameterData')

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError as key_error:
            raise InputValidationError(
                'no structure is specified for this calculation'
            ) from key_error

        try:
            trajectory = inputdict.pop(self.get_linkname('trajectory'))
        except KeyError as key_error:
            raise InputValidationError(
                'trajectory is specified for this calculation') from key_error

        try:
            force_constants = inputdict.pop(
                self.get_linkname('force_constants'))
        except KeyError as key_error:
            raise InputValidationError(
                'no force_constants is specified for this calculation'
            ) from key_error

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError as key_error:
            raise InputValidationError(
                'no code is specified for this calculation') from key_error

        time_step = trajectory.get_times()[1] - trajectory.get_times()[0]

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        # =================== prepare the python input files =====================

        cell_txt = get_poscar_txt(structure)
        input_txt = parameters_to_input_file(parameters_data)
        force_constants_txt = get_force_constants(force_constants)
        trajectory_txt = get_trajectory_txt(trajectory)

        # =========================== dump to file =============================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        cell_filename = tempfolder.get_abs_path(self._INPUT_CELL)
        with open(cell_filename, 'w') as infile:
            infile.write(cell_txt)

        force_constants_filename = tempfolder.get_abs_path(
            self._INPUT_FORCE_CONSTANTS)
        with open(force_constants_filename, 'w') as infile:
            infile.write(force_constants_txt)

        trajectory_filename = tempfolder.get_abs_path(self._INPUT_TRAJECTORY)
        with open(trajectory_filename, 'w') as infile:
            infile.write(trajectory_txt)

        # ============================ calcinfo ================================

        local_copy_list = []
        remote_copy_list = []
        #    additional_retrieve_list = settings_dict.pop("ADDITIONAL_RETRIEVE_LIST",[])

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        # Empty command line by default
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        # Retrieve files
        calcinfo.retrieve_list = [
            self._OUTPUT_FILE_NAME,
            self._OUTPUT_FORCE_CONSTANTS,
            self._OUTPUT_QUASIPARTICLES,
        ]

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = [
            self._INPUT_FILE_NAME,
            self._INPUT_TRAJECTORY,
            '-ts',
            f'{time_step}',
            '--silent',
            '-sfc',
            self._OUTPUT_FORCE_CONSTANTS,
            '-thm',  # '--resolution 0.01',
            '-psm',
            '2',
            '--normalize_dos',
            '-sdata',
        ]

        if 'temperature' in parameters_data.get_dict():
            codeinfo.cmdline_params.append('--temperature')
            codeinfo.cmdline_params.append(
                f'{parameters_data.dict.temperature}')

        if 'md_commensurate' in parameters_data.get_dict():
            if parameters_data.dict.md_commensurate:
                codeinfo.cmdline_params.append('--MD_commensurate')

        codeinfo.stdout_name = self._OUTPUT_FILE_NAME
        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = False
        calcinfo.codes_info = [codeinfo]
        return calcinfo
