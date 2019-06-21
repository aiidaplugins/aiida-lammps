import os
import traceback

from aiida.parsers.parser import Parser
from aiida.common import exceptions

from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_log_file


class LAMMPSBaseParser(Parser):
    """
    Abstract Base Parser for LAMMPS, supplying common methods
    """

    def __init__(self, node):
        """
        Initialize the instance of Force LammpsParser
        """
        super(LAMMPSBaseParser, self).__init__(node)

    def get_parsing_resources(self, kwargs, traj_in_temp=False, sys_info=False):
        """ check that all resources, required for parsing, are present """
        # Check that the retrieved folder is there
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return None, self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        if traj_in_temp or sys_info:
            if 'retrieved_temporary_folder' not in kwargs:
                return None, self.exit_codes.ERROR_NO_RETRIEVED_TEMP_FOLDER
            temporary_folder = kwargs['retrieved_temporary_folder']
            list_of_temp_files = os.listdir(temporary_folder)

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()

        # check log file
        if self.node.get_option('output_filename') not in list_of_files:
            return None, self.exit_codes.ERROR_LOG_FILE_MISSING

        trajectory_filename = self.node.get_option('trajectory_name')

        if traj_in_temp:
            # check trajectory in temporal folder
            if trajectory_filename not in list_of_temp_files:
                return None, self.exit_codes.ERROR_TRAJ_FILE_MISSING
            trajectory_filepath = os.path.join(temporary_folder, trajectory_filename)
        else:
            # check trajectory in retrived folder
            if trajectory_filename not in list_of_files:
                return None, self.exit_codes.ERROR_TRAJ_FILE_MISSING
            trajectory_filepath = None

        # check stdin and stdout
        if self.node.get_option('scheduler_stdout') not in list_of_files:
            return None, self.exit_codes.ERROR_STDOUT_FILE_MISSING
        if self.node.get_option('scheduler_stderr') not in list_of_files:
            return None, self.exit_codes.ERROR_STDERR_FILE_MISSING

        info_filepath = None
        if sys_info:
            info_filename = self.node.get_option('info_filename')
            if info_filename in list_of_temp_files:
                info_filepath = os.path.join(temporary_folder, self.node.get_option('info_filename'))

        return (trajectory_filename, trajectory_filepath, info_filepath), None

    def parse_log_file(self, compute_stress=False):
        """ parse the log file """
        output_filename = self.node.get_option('output_filename')
        output_txt = self.retrieved.get_object_content(output_filename)
        try:
            output_data = read_log_file(output_txt, compute_stress=compute_stress)
        except Exception:
            traceback.print_exc()
            return None, self.exit_codes.ERROR_LOG_PARSING
        return output_data, None

    def add_warnings_and_errors(self, output_data):
        """ add warning and errors to the output data """
        # add the dictionary with warnings and errors
        warnings = self.retrieved.get_object_content(self.node.get_option("scheduler_stderr"))
        # for some reason, errors may be in the stdout, but not the log.lammps
        stdout = self.retrieved.get_object_content(self.node.get_option("scheduler_stdout"))
        errors = [line for line in stdout.splitlines() if line.startswith("ERROR")]

        for error in errors:
            self.logger.error(error)

        output_data.update({'warnings': warnings})
        output_data.update({'errors': errors})

    def add_standard_info(self, output_data):
        """ add standard information to output data """
        output_data["parser_class"] = self.__class__.__name__
        output_data["parser_version"] = aiida_lammps_version
