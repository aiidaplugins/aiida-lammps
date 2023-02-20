"""
Base parser for LAMMPS calculations.

It takes care of parsing the log.lammps file, the trajectory file and the
yaml file with the final value of the variables printed in the ``thermo_style``.
"""
from aiida import orm
from aiida.common import exceptions
from aiida.parsers.parser import Parser
import numpy as np

from aiida_lammps.common.raw_parsers import parse_final_data, parse_logfile
from aiida_lammps.data.trajectory import LammpsTrajectory


class LAMMPSBaseParser(Parser):
    """
    Base parser for LAMMPS calculations.

    It takes care of parsing the log.lammps file, the trajectory file and the
    yaml file with the final value of the variables printed in the
    ``thermo_style``.
    """

    def __init__(self, node):
        """Initialize the parser"""
        # pylint: disable=useless-super-delegation
        super().__init__(node)

    def parse(self, **kwargs):
        """
        Parse the files produced by lammps.

        It takes care of parsing the log.lammps file, the trajectory file and the
        yaml file with the final value of the variables printed in the
        ``thermo_style``.
        """
        # pylint: disable=too-many-return-statements, too-many-locals

        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        list_of_files = out_folder.base.repository.list_object_names()

        # check log file
        if self.node.get_option("logfile_filename") not in list_of_files:
            return self.exit_codes.ERROR_LOG_FILE_MISSING
        filename = self.node.get_option("logfile_filename")
        parsed_data = parse_logfile(
            file_contents=self.node.outputs.retrieved.base.repository.get_object_content(
                filename
            )
        )
        if parsed_data is None:
            return self.exit_codes.ERROR_PARSING_LOGFILE
        global_data = parsed_data["global"]
        arrays = parsed_data["time_dependent"]

        # check final variable file
        if self.node.get_option("variables_filename") not in list_of_files:
            return self.exit_codes.ERROR_FINAL_VARIABLE_FILE_MISSING

        filename = self.node.get_option("variables_filename")
        final_variables = parse_final_data(
            file_contents=self.node.outputs.retrieved.base.repository.get_object_content(
                filename
            )
        )
        if final_variables is None:
            return self.exit_codes.ERROR_PARSING_FINAL_VARIABLES

        results = orm.Dict(dict={**final_variables, "compute_variables": global_data})

        # Expose the results from the log.lammps outputs
        self.out("results", results)

        # Get the time-dependent outputs exposed as an ArrayData

        time_dependent_computes = orm.ArrayData()

        for key, value in arrays.items():
            _data = [val if val is not None else np.nan for val in value]
            time_dependent_computes.set_array(key, np.array(_data))

        self.out("time_dependent_computes", time_dependent_computes)

        # check trajectory file
        if self.node.get_option("trajectory_filename") not in list_of_files:
            return self.exit_codes.ERROR_TRAJECTORY_FILE_MISSING
        # Gather the lammps trajectory data
        filename = self.node.get_option("trajectory_filename")
        with self.node.outputs.retrieved.open(filename) as handle:
            lammps_trajectory = LammpsTrajectory(handle)
        self.out("trajectories", lammps_trajectory)

        self.out("structure", lammps_trajectory.get_step_structure(-1))

        # check stdout
        if self.node.get_option("scheduler_stdout") not in list_of_files:
            return self.exit_codes.ERROR_STDOUT_FILE_MISSING

        # check stderr
        if self.node.get_option("scheduler_stderr") not in list_of_files:
            return self.exit_codes.ERROR_STDERR_FILE_MISSING

        return None
