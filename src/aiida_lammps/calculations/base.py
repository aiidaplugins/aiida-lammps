"""
A basic plugin for performing calculations in ``LAMMPS`` using aiida.

The plugin will take the input parameters validate them against a schema
and then use them to generate the ``LAMMPS`` input file. The input file
is generated depending on the parameters provided, the type of potential,
the input structure and whether or not a restart file is provided.
"""
import os
from typing import ClassVar, Union

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob

from aiida_lammps.data.potential import LammpsPotentialData
from aiida_lammps.data.trajectory import LammpsTrajectory
from aiida_lammps.parsers.inputfile import generate_input_file
from aiida_lammps.parsers.utils import generate_lammps_structure
from aiida_lammps.validation.utils import validate_against_schema


class LammpsBaseCalculation(CalcJob):
    """
    A basic plugin for performing calculations in ``LAMMPS`` using aiida.

    The plugin will take the input parameters validate them against a schema
    and then use them to generate the ``LAMMPS`` input file. The input file
    is generated depending on the parameters provided, the type of potential,
    the input structure and whether or not a restart file is provided.
    """

    _DEFAULT_VARIABLES: ClassVar[dict[str, str]] = {
        "input_filename": "input.in",
        "structure_filename": "structure.dat",
        "output_filename": "lammps.out",
        "logfile_filename": "log.lammps",
        "variables_filename": "aiida_lammps.yaml",
        "trajectory_filename": "aiida_lammps.trajectory.dump",
        "restart_filename": "lammps.restart",
        "parser_name": "lammps.base",
    }

    _POTENTIAL_FILENAME = "potential.dat"

    # In restarts, will not copy but use symlinks
    _default_symlink_usage = True

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="Structure used in the ``LAMMPS`` calculation",
        )
        spec.input(
            "potential",
            valid_type=LammpsPotentialData,
            help="Potential used in the ``LAMMPS`` calculation",
        )
        spec.input(
            "parameters",
            valid_type=orm.Dict,
            validator=cls._validate_parameters,
            help="Parameters that control the input script generated for the ``LAMMPS`` calculation",
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            required=False,
            validator=cls._validate_settings,
            help="Additional settings that control the ``LAMMPS`` calculation",
        )
        spec.input(
            "input_restartfile",
            valid_type=orm.SinglefileData,
            required=False,
            help="Input restartfile to continue from a previous ``LAMMPS`` calculation",
        )
        spec.input(
            "parent_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="An optional working directory of a previously completed calculation to restart from.",
        )
        spec.input(
            "metadata.options.input_filename",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["input_filename"],
        )
        spec.input(
            "metadata.options.structure_filename",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["structure_filename"],
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["output_filename"],
        )
        spec.input(
            "metadata.options.variables_filename",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["variables_filename"],
        )
        spec.input(
            "metadata.options.trajectory_filename",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["trajectory_filename"],
        )
        spec.input(
            "metadata.options.restart_filename",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["restart_filename"],
        )
        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._DEFAULT_VARIABLES["parser_name"],
        )
        spec.inputs.validator = cls._validate_inputs

        spec.output(
            "results",
            valid_type=orm.Dict,
            required=True,
            help="The data extracted from the lammps output file",
        )
        spec.output(
            "trajectories",
            valid_type=LammpsTrajectory,
            required=True,
            help="The data extracted from the lammps trajectory file",
        )
        spec.output(
            "time_dependent_computes",
            valid_type=orm.ArrayData,
            required=True,
            help="The data with the time dependent computes parsed from the lammps.out",
        )
        spec.output(
            "restartfile",
            valid_type=orm.SinglefileData,
            required=False,
            help="The restartfile of a ``LAMMPS`` calculation",
        )
        spec.output(
            "structure",
            valid_type=orm.StructureData,
            required=False,
            help="The output structure.",
        )
        spec.exit_code(
            301,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="the retrieved folder data node could not be accessed.",
            invalidates_cache=True,
        )
        spec.exit_code(
            302,
            "ERROR_STDOUT_FILE_MISSING",
            message="the stdout output file was not found",
        )
        spec.exit_code(
            303,
            "ERROR_STDERR_FILE_MISSING",
            message="the stderr output file was not found",
        )
        spec.exit_code(
            304,
            "ERROR_OUTPUT_FILE_MISSING",
            message="the output file is missing, it is possible that LAMMPS never ran",
        )
        spec.exit_code(
            305,
            "ERROR_LOG_FILE_MISSING",
            message="the file with the lammps log was not found",
            invalidates_cache=True,
        )
        spec.exit_code(
            306,
            "ERROR_FINAL_VARIABLE_FILE_MISSING",
            message="the file with the final variables was not found",
            invalidates_cache=True,
        )
        spec.exit_code(
            307,
            "ERROR_TRAJECTORY_FILE_MISSING",
            message="the file with the trajectories was not found",
            invalidates_cache=True,
        )
        spec.exit_code(
            308,
            "ERROR_RESTART_FILE_MISSING",
            message="the file with the restart information was not found",
        )
        spec.exit_code(
            309,
            "ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR",
            message="The parser detected the lammps error :{error}",
        )
        spec.exit_code(
            400,
            "ERROR_OUT_OF_WALLTIME",
            message="The calculation stopped prematurely because it ran out of walltime.",
        )
        spec.exit_code(
            401,
            "ERROR_ENERGY_NOT_CONVERGED",
            message="The energy tolerance was not reached at minimization.",
        )
        spec.exit_code(
            402,
            "ERROR_FORCE_NOT_CONVERGED",
            message="The force tolerance was not reached at minimization.",
        )
        spec.exit_code(
            1001,
            "ERROR_PARSING_OUTFILE",
            message="error parsing the output file has failed.",
        )
        spec.exit_code(
            1002,
            "ERROR_PARSING_FINAL_VARIABLES",
            message="error parsing the final variable file has failed.",
        )

    @classmethod
    def _validate_inputs(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument, inconsistent-return-statements
        """Validate the top-level inputs namespace."""
        if "parameters" in value:
            _restart = any(
                value["parameters"].get_dict().get("restart", {}).get(key, False)
                for key in ["print_final", "print_intermediate"]
            )
            if "settings" in value:
                _store_restart = (
                    value["settings"].get_dict().get("store_restart", False)
                )
            else:
                _store_restart = False

            if _store_restart and not _restart:
                return (
                    "To store the restartfile one needs to indicate "
                    "that either the final or intermediate restartfiles must be printed"
                )

    @classmethod
    def _validate_settings(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument, inconsistent-return-statements
        """Validate the ``settings`` input."""
        if not value:
            return

        settings = value.get_dict()
        additional_cmdline_params = settings.get("additional_cmdline_params", [])

        additional_retrieve_list = settings.get("additional_retrieve_list", [])

        if not isinstance(additional_cmdline_params, list) or any(
            not isinstance(e, str) for e in additional_cmdline_params
        ):
            return (
                "Invalid value for `additional_cmdline_params`, should be "
                f"list of strings but got: {additional_cmdline_params}"
            )
        if not isinstance(additional_retrieve_list, list) or any(
            not isinstance(e, (str, tuple)) for e in additional_retrieve_list
        ):
            return (
                "Invalid value for `additional_retrieve_list`, should be "
                f"list of strings or of tuples but got: {additional_retrieve_list}"
            )

    @classmethod
    def _validate_parameters(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument,inconsistent-return-statements
        """
        Validate the input parameters and compares them against a schema.

        Takes the input parameters dictionaries that will be used to generate the
        LAMMPS input parameter and will be checked against a schema for validation.
        """

        parameters = value.get_dict()
        if not any(key in parameters for key in ["md", "minimize"]):
            return (
                "If not using a script the type of calculation, either "
                "'md' or 'minimize' must be given"
            )

        _file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "validation/schemas/lammps_schema.json",
        )

        validate_against_schema(data=parameters, filename=_file)

    def prepare_for_submission(self, folder):
        """
        Create the input files from the input nodes passed to this instance of the `CalcJob`.
        """
        # pylint: disable=too-many-locals

        # Get the name of the trajectory file
        _trajectory_filename = self.inputs.metadata.options.trajectory_filename

        # Get the name of the variables file
        _variables_filename = self.inputs.metadata.options.variables_filename

        # Get the name of the restart file
        _restart_filename = self.inputs.metadata.options.restart_filename

        # Get the name of the output file
        _output_filename = self.inputs.metadata.options.output_filename

        # Get the parameters dictionary so that they can be used for creating
        # the input file
        if "parameters" in self.inputs:
            _parameters = self.inputs.parameters.get_dict()
        else:
            _parameters = {}

        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}

        # Set the remote copy list and the symlink so that if one needs to use restartfiles from
        # a previous calculations one can do so without problems
        remote_copy_list = []
        remote_symlink_list = []
        local_copy_list = []
        retrieve_temporary_list = []
        retrieve_list = [
            _output_filename,
            _variables_filename,
            _trajectory_filename,
        ]

        # Handle the restart file for simulations coming from previous runs
        restart_data = self.handle_restartfiles(
            settings=settings,
            parameters=_parameters,
        )
        _read_restart_filename = restart_data.get("restart_file", None)
        remote_copy_list += restart_data.get("remote_copy_list", [])
        remote_symlink_list += restart_data.get("remote_symlink_list", [])
        local_copy_list += restart_data.get("local_copy_list", [])
        retrieve_list += restart_data.get("retrieve_list", [])
        retrieve_temporary_list += restart_data.get("retrieve_temporary_list", [])

        # Generate the content of the structure file based on the input
        # structure
        structure_filecontent, _ = generate_lammps_structure(
            self.inputs.structure,
            self.inputs.potential.atom_style,
        )

        # Get the name of the structure file and write it to the remote folder
        _structure_filename = self.inputs.metadata.options.structure_filename

        with folder.open(_structure_filename, "w") as handle:
            handle.write(structure_filecontent)

        # Write the potential to the remote folder
        with folder.open(self._POTENTIAL_FILENAME, "w") as handle:
            handle.write(self.inputs.potential.get_content())

        # Write the input file content. This function will also check the
        # sanity of the passed parameters when comparing it to a schema
        input_filecontent = generate_input_file(
            potential=self.inputs.potential,
            structure=self.inputs.structure,
            parameters=_parameters,
            restart_filename=_restart_filename,
            trajectory_filename=_trajectory_filename,
            variables_filename=_variables_filename,
            read_restart_filename=_read_restart_filename,
        )

        # Get the name of the input file, and write it to the remote folder
        _input_filename = self.inputs.metadata.options.input_filename

        with folder.open(_input_filename, "w") as handle:
            handle.write(input_filecontent)

        cmdline_params = ["-in", _input_filename]

        if "settings" in self.inputs:
            settings = self.inputs.settings.get_dict()
            cmdline_params += settings.get("additional_cmdline_params", [])

        codeinfo = datastructures.CodeInfo()
        # Command line variables to ensure that the input file from LAMMPS can
        # be read
        codeinfo.cmdline_params = cmdline_params
        # Set the code uuid
        codeinfo.code_uuid = self.inputs.code.uuid
        # Set the name of the stdout
        codeinfo.stdout_name = _output_filename

        # Generate the datastructure for the calculation information
        calcinfo = datastructures.CalcInfo()
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list
        # Define the list of temporary files that will be retrieved
        calcinfo.retrieve_temporary_list = retrieve_temporary_list
        # Set the files that must be retrieved
        calcinfo.retrieve_list = retrieve_list
        calcinfo.retrieve_list += settings.get("additional_retrieve_list", [])
        # Set the information of the code into the calculation datastructure
        calcinfo.codes_info = [codeinfo]

        return calcinfo

    def handle_restartfiles(
        self,
        settings: dict,
        parameters: dict,
    ) -> dict:
        """Get the information needed to handle the restartfiles

        :param settings: Additional settings that control the ``LAMMPS`` calculation
        :type settings: dict
        :param parameters: Parameters that control the input script generated for the ``LAMMPS`` calculation
        :type parameters: dict
        :raises aiida.common.exceptions.InputValidationError: if the name of the given restart file is not in the \
            remote folder
        :return: dictionary with the information about how to handle the restartfile either for parsing, \
            storage or input
        :rtype: dict
        """
        local_copy_list = []
        remote_symlink_list = []
        remote_copy_list = []
        retrieve_list = []
        retrieve_temporary_list = []
        # If there is a restartfile set its name to the input variables and
        # write it in the remote folder
        if "input_restartfile" in self.inputs:
            _read_restart_filename = self.inputs.metadata.options.restart_filename
            local_copy_list.append(
                (
                    self.inputs.input_restartfile.uuid,
                    self.inputs.input_restartfile.filename,
                    self.inputs.metadata.options.restart_filename,
                )
            )
        else:
            _read_restart_filename = None

        # check if there is a parent folder to restart the simulation from a previous run
        if "parent_folder" in self.inputs:
            # Check if one should do symlinks or if one should copy the files
            # By default symlinks are used as the file can be quite large
            symlink = settings.pop("parent_folder_symlink", self._default_symlink_usage)
            # Find the name of the previous restartfile, if none is given the default one is assumed
            # Setting the name here will mean that if the input file is generated from the parameters
            # that this name will be used
            _read_restart_filename = settings.pop(
                "previous_restartfile", self.inputs.metadata.options.restart_filename
            )

            if symlink:
                # Symlink the old restart file to the new one in the current directory
                remote_symlink_list.append(
                    (
                        self.inputs.parent_folder.computer.uuid,
                        os.path.join(
                            self.inputs.parent_folder.get_remote_path(),
                            _read_restart_filename,
                        ),
                        "input_lammps.restart",
                    )
                )
                _read_restart_filename = "input_lammps.restart"
            else:
                # Copy the old restart file to the current directory
                remote_copy_list.append(
                    (
                        self.inputs.parent_folder.computer.uuid,
                        os.path.join(
                            self.inputs.parent_folder.get_remote_path(),
                            _read_restart_filename,
                        ),
                        "input_lammps.restart",
                    )
                )
                _read_restart_filename = "input_lammps.restart"

        # Add the restart file to the list of files to be retrieved if we want to store it in the
        # database
        if "restart" in parameters and settings.get("store_restart", False):
            if parameters.get("restart", {}).get("print_final", False):
                retrieve_list.append(self.inputs.metadata.options.restart_filename)
            if parameters.get("restart", {}).get("print_intermediate", False):
                retrieve_temporary_list.append(
                    f"{self.inputs.metadata.options.restart_filename}*"
                )
        data = {
            "remote_copy_list": remote_copy_list,
            "remote_symlink_list": remote_symlink_list,
            "local_copy_list": local_copy_list,
            "restart_file": _read_restart_filename,
            "retrieve_list": retrieve_list,
            "retrieve_temporary_list": retrieve_temporary_list,
        }
        return data
