from collections import namedtuple
from fnmatch import fnmatch
import os
import traceback

from aiida.common import exceptions
from aiida.parsers.parser import Parser

from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_log_file

ParsingResources = namedtuple(
    "ParsingResources", ["exit_code", "sys_paths", "traj_paths", "restart_paths"]
)


class LAMMPSBaseParser(Parser):
    """Abstract Base Parser for LAMMPS, supplying common methods."""

    def __init__(self, node):
        """Initialize the parser."""
        super(LAMMPSBaseParser, self).__init__(node)

    def get_parsing_resources(self, kwargs, traj_in_temp=False, sys_in_temp=True):
        """Check that all resources, required for parsing, are present."""
        # Check for retrieved folder
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return ParsingResources(
                self.exit_codes.ERROR_NO_RETRIEVED_FOLDER, None, None, None
            )

        # Check for temporary folder
        if traj_in_temp or sys_in_temp:
            if "retrieved_temporary_folder" not in kwargs:
                return ParsingResources(
                    self.exit_codes.ERROR_NO_RETRIEVED_TEMP_FOLDER, None, None, None
                )
            temporary_folder = kwargs["retrieved_temporary_folder"]
            list_of_temp_files = os.listdir(temporary_folder)

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()

        # check log file
        if self.node.get_option("output_filename") not in list_of_files:
            return ParsingResources(
                self.exit_codes.ERROR_LOG_FILE_MISSING, None, None, None
            )

        # check stdin and stdout
        if self.node.get_option("scheduler_stdout") not in list_of_files:
            return ParsingResources(
                self.exit_codes.ERROR_STDOUT_FILE_MISSING, None, None, None
            )
        if self.node.get_option("scheduler_stderr") not in list_of_files:
            return ParsingResources(
                self.exit_codes.ERROR_STDERR_FILE_MISSING, None, None, None
            )

        # check for system info file(s)
        system_suffix = self.node.get_option("system_suffix")
        system_filepaths = []
        if sys_in_temp:
            for filename in list_of_temp_files:
                if fnmatch(filename, "*" + system_suffix):
                    system_filepaths.append(os.path.join(temporary_folder, filename))
        else:
            for filename in list_of_files:
                if fnmatch(filename, "*" + system_suffix):
                    system_filepaths.append(filename)

        # check for trajectory file(s)
        trajectory_suffix = self.node.get_option("trajectory_suffix")
        trajectory_filepaths = []
        if traj_in_temp:
            for filename in list_of_temp_files:
                if fnmatch(filename, "*" + trajectory_suffix):
                    trajectory_filepaths.append(
                        os.path.join(temporary_folder, filename)
                    )
        else:
            for filename in list_of_files:
                if fnmatch(filename, "*" + trajectory_suffix):
                    trajectory_filepaths.append(filename)

        # check for restart file(s)
        restart_file = self.node.get_option("restart_filename")
        restart_filepaths = []
        for filename in list_of_temp_files:
            if fnmatch(filename, "*" + restart_file + "*"):
                restart_filepaths.append(os.path.join(temporary_folder, filename))

        return ParsingResources(
            None, system_filepaths, trajectory_filepaths, restart_filepaths
        )

    def parse_log_file(self, compute_stress=False):
        """Parse the log file."""
        output_filename = self.node.get_option("output_filename")
        output_txt = self.retrieved.get_object_content(output_filename)
        try:
            output_data = read_log_file(output_txt, compute_stress=compute_stress)
        except Exception:
            traceback.print_exc()
            return None, self.exit_codes.ERROR_LOG_PARSING
        return output_data, None

    def add_warnings_and_errors(self, output_data):
        """Add warning and errors to the output data."""
        # add the dictionary with warnings and errors
        warnings = (
            self.retrieved.get_object_content(self.node.get_option("scheduler_stderr"))
            .strip()
            .splitlines()
        )
        # for some reason, errors may be in the stdout, but not the log.lammps
        stdout = self.retrieved.get_object_content(
            self.node.get_option("scheduler_stdout")
        )
        errors = [line for line in stdout.splitlines() if line.startswith("ERROR")]
        warnings.extend(
            [line for line in stdout.splitlines() if line.startswith("WARNING")]
        )

        for error in errors:
            self.logger.error(error)

        output_data.setdefault("warnings", []).extend(warnings)
        output_data.setdefault("errors", []).extend(errors)

    def add_standard_info(self, output_data):
        """Add standard information to output data."""
        output_data["parser_class"] = self.__class__.__name__
        output_data["parser_version"] = aiida_lammps_version
