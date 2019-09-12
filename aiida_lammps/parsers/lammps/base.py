from collections import namedtuple
from fnmatch import fnmatch
import io
import os
import traceback

import numpy as np

from aiida.parsers.parser import Parser
from aiida.orm import TrajectoryData
from aiida.common import exceptions

from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_log_file, iter_lammps_trajectories

ParsingResources = namedtuple(
    "ParsingResources", ["exit_code", "sys_data_path", "traj_paths"]
)


class LAMMPSBaseParser(Parser):
    """Abstract Base Parser for LAMMPS, supplying common methods."""

    def __init__(self, node):
        """Initialize the parser."""
        super(LAMMPSBaseParser, self).__init__(node)

    def get_parsing_resources(self, kwargs, traj_in_temp=False, sys_info=False):
        """Check that all resources, required for parsing, are present."""
        # Check for retrieved folder
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return ParsingResources(
                self.exit_codes.ERROR_NO_RETRIEVED_FOLDER, None, None
            )

        # Check for temporary folder
        if traj_in_temp or sys_info:
            if "retrieved_temporary_folder" not in kwargs:
                return ParsingResources(
                    self.exit_codes.ERROR_NO_RETRIEVED_TEMP_FOLDER, None, None
                )
            temporary_folder = kwargs["retrieved_temporary_folder"]
            list_of_temp_files = os.listdir(temporary_folder)

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()

        # check log file
        if self.node.get_option("output_filename") not in list_of_files:
            return ParsingResources(self.exit_codes.ERROR_LOG_FILE_MISSING, None, None)

        # check stdin and stdout
        if self.node.get_option("scheduler_stdout") not in list_of_files:
            return ParsingResources(
                self.exit_codes.ERROR_STDOUT_FILE_MISSING, None, None
            )
        if self.node.get_option("scheduler_stderr") not in list_of_files:
            return ParsingResources(
                self.exit_codes.ERROR_STDERR_FILE_MISSING, None, None
            )

        # check for system info file
        info_filepath = None
        if sys_info:
            info_filename = self.node.get_option("info_filename")
            if info_filename in list_of_temp_files:
                info_filepath = os.path.join(
                    temporary_folder, self.node.get_option("info_filename")
                )

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

        return ParsingResources(None, info_filepath, trajectory_filepaths)

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
        warnings = self.retrieved.get_object_content(
            self.node.get_option("scheduler_stderr")
        )
        # for some reason, errors may be in the stdout, but not the log.lammps
        stdout = self.retrieved.get_object_content(
            self.node.get_option("scheduler_stdout")
        )
        errors = [line for line in stdout.splitlines() if line.startswith("ERROR")]

        for error in errors:
            self.logger.error(error)

        output_data.update({"warnings": warnings})
        output_data.update({"errors": errors})

    def add_standard_info(self, output_data):
        """Add standard information to output data."""
        output_data["parser_class"] = self.__class__.__name__
        output_data["parser_version"] = aiida_lammps_version

    @staticmethod
    def parse_trajectory_file(trajectory_filepath):
        """Parse a trajectory file."""
        with io.open(trajectory_filepath, "r") as handle:
            traj_steps = list(iter_lammps_trajectories(handle))
        if not traj_steps:
            raise IOError("trajectory file empty")

        timesteps = []
        cells = []
        field_names = None
        elements = None
        positions = []
        other = {}
        for traj_step in traj_steps:

            if not set(traj_step.field_names).issuperset(["element", "x", "y", "z"]):
                raise IOError(
                    "trajectory step {} does not contain required fields".format(
                        traj_step.timestep
                    )
                )
            if field_names is None:
                field_names = traj_step.field_names
            elif traj_step.field_names != field_names:
                raise IOError(
                    "trajectory step {} contains different field names".format(
                        traj_step.timestep
                    )
                )
            fmap = {n: i for i, n in enumerate(traj_step.field_names)}
            if elements is None:
                elements = [f[fmap["element"]] for f in traj_step.fields]
            elif elements != [f[fmap["element"]] for f in traj_step.fields]:
                raise IOError(
                    "trajectory step {} contains different elements".format(
                        traj_step.timestep
                    )
                )
            positions.append(
                [[f[fmap["x"]], f[fmap["y"]], f[fmap["z"]]] for f in traj_step.fields]
            )
            for key in fmap:
                if key not in ["element", "x", "y", "z"]:
                    other.setdefault(key, []).append(
                        [f[fmap[key]] for f in traj_step.fields]
                    )

            cells.append(traj_step.cell)
            timesteps.append(traj_step.timestep)

        # save trajectories into node
        trajectory_data = TrajectoryData()
        trajectory_data.set_trajectory(
            elements,
            np.array(positions, dtype=float),
            stepids=np.array(timesteps, dtype=int),
            cells=np.array(cells),
        )
        for key, val in other.items():
            trajectory_data.set_array(key, np.array(val, dtype=float))

        return trajectory_data
