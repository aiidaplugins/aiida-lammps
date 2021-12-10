"""
Base ``LAMMPS`` calculation for AiiDA.
"""
from aiida import orm
from aiida.engine import CalcJob
from aiida.common import datastructures
from aiida_lammps.data.lammps_potential import LammpsPotentialData
from aiida_lammps.common.generate_structure import generate_lammps_structure
from aiida_lammps.common.input_generator import generate_input_file


class BaseLammpsCalculation(CalcJob):
    """
    A basic plugin for performing calculations in ``LAMMPS`` using aiida.
    """

    _INPUT_FILENAME = 'input.in'
    _STRUCTURE_FILENAME = 'structure.dat'

    _DEFAULT_OUTPUT_FILENAME = 'log.lammps'
    _DEFAULT_TRAJECTORY_FILENAME = 'aiida_lampps.trajectory.dump'
    _DEFAULT_VARIABLES_FILENAME = 'aiida_lammps.yaml'
    _DEFAULT_RESTART_FILENAME = 'lammps.restart'
    _DEFAULT_POTENTIAL_FILENAME = 'potential.dat'
    _DEFAULT_READ_RESTART_FILENAME = 'aiida_lammps.restart'

    _cmdline_params = ('-in', _INPUT_FILENAME)
    _stdout_name = None

    @classmethod
    def define(cls, spec):
        super(BaseLammpsCalculation, cls).define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=True,
            help='Structure used in the ``LAMMPS`` calculation',
        )
        spec.input(
            'potential',
            valid_type=LammpsPotentialData,
            required=True,
            help='Potential used in the ``LAMMPS`` calculation',
        )
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            required=True,
            help='Parameters that control the ``LAMMPS`` calculation',
        )
        spec.input(
            'input_restartfile',
            valid_type=orm.SinglefileData,
            required=False,
            help=
            'Input restartfile to continue from a previous ``LAMMPS`` calculation'
        )
        spec.input(
            'metadata.options.input_filename',
            valid_type=str,
            default=cls._INPUT_FILENAME,
        )
        spec.input(
            'metadata.options.structure_filename',
            valid_type=str,
            default=cls._STRUCTURE_FILENAME,
        )
        spec.input(
            'metadata.options.output_filename',
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILENAME,
        )
        spec.input(
            'metadata.options.variables_filename',
            valid_type=str,
            default=cls._DEFAULT_VARIABLES_FILENAME,
        )
        spec.input(
            'metadata.options.trajectory_filename',
            valid_type=str,
            default=cls._DEFAULT_TRAJECTORY_FILENAME,
        )
        spec.input(
            'metadata.options.restart_filename',
            valid_type=str,
            default=cls._DEFAULT_RESTART_FILENAME,
        )
        spec.output(
            'results',
            valid_type=orm.Dict,
            required=True,
            help='The data extracted from the lammps log file',
        )
        spec.output(
            'trajectories',
            valid_type=LammpsPotentialData,
            required=True,
            help='The data extracted from the lammps trajectory file',
        )
        spec.output(
            'time_dependent_computes',
            valid_types=orm.Dict,
            required=True,
            help=
            'The data with the time dependent computes parsed from the lammps.log',
        )
        spec.output(
            'restartfile',
            valid_type=orm.SinglefileData,
            required=True,
            help='The restartfile of a ``LAMMPS`` calculation',
        )
        spec.output(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help='The output structure.',
        )
        spec.exit_code(
            350,
            'ERROR_NO_RETRIEVED_FOLDER',
            message='the retrieved folder data node could not be accessed.',
            invalidates_cache=True,
        )
        spec.exit_code(
            351,
            'ERROR_LOG_FILE_MISSING',
            mesage='the file with the lammps log was not found',
            invalidates_cache=True,
        )
        spec.exit_code(
            352,
            'ERROR_FINAL_VARIABLE_FILE_MISSING',
            mesage='the file with the final variables was not found',
            invalidates_cache=True,
        )
        spec.exit_code(
            353,
            'ERROR_TRAJECTORY_FILE_MISSING',
            mesage='the file with the trajectories was not found',
            invalidates_cache=True,
        )
        spec.exit_code(
            354,
            'ERROR_STDOUT_FILE_MISSING',
            message='the stdout output file was not found',
        )
        spec.exit_code(
            355,
            'ERROR_STDERR_FILE_MISSING',
            message='the stderr output file was not found',
        )
        spec.exit_code(
            1001,
            'ERROR_PARSING_LOGFILE',
            message='error parsing the log file has failed.',
        )
        spec.exit_code(
            1002,
            'ERROR_PARSING_FINAL_VARIABLES',
            message='error parsing the final variable file has failed.',
        )

    def prepare_for_submission(self, folder):
        # pylint: disable=too-many-locals
        # Setup structure
        structure_filecontent, _ = generate_lammps_structure(
            self.inputs.structure,
            self.inputs.potential.atom_style,
        )

        _structure_filename = self.inputs.metadata.options.structure_filename

        with folder.open(_structure_filename, 'w') as handle:
            handle.write(structure_filecontent)

        _parameters = self.inputs.parameters.get_dict()

        _trajectory_filename = self.inputs.metadata.options.restart_filename

        _variables_filename = self.inputs.metadata.options.variables_filename

        _restart_filename = self.inputs.metadata.options.restart_filename

        _output_filename = self.inputs.metadata.options.output_filename

        if 'input_restartfile' in self.inputs:
            _read_restart_filename = self.inputs.input_restartfile
        else:
            _read_restart_filename = None

        input_filecontent = generate_input_file(
            potential=self.inputs.potential,
            structure=self.inputs.strutcure,
            parameters=_parameters,
            restart_filename=_restart_filename,
            trajectory_filename=_trajectory_filename,
            variables_filename=_variables_filename,
            read_restart_filename=_read_restart_filename,
        )

        _input_filename = self.inputs.metadata.options.input_filename

        with folder.open(_input_filename, 'w') as handle:
            handle.write(input_filecontent)

        with folder.open(self._DEFAULT_POTENTIAL_FILENAME, 'w') as handle:
            handle.write(self.inputs.potential.get_content())

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = []
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = _output_filename
        codeinfo.withmpi = self.inputs.metadata.options.withmpi

        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = str(self.uuid)
        # Retrieve by default the output file and the xml file
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(_output_filename)
        calcinfo.retrieve_list.append(_restart_filename)
        calcinfo.retrieve_list.append(_variables_filename)
        calcinfo.retrieve_list.append(_trajectory_filename)
        calcinfo.codes_info = [codeinfo]

        return calcinfo
