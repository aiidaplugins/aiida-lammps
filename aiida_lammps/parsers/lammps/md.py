import traceback

from aiida.orm import ArrayData, Dict
import numpy as np

from aiida_lammps.common.raw_parsers import convert_units, get_units_dict
from aiida_lammps.data.trajectory import LammpsTrajectory
from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser


class MdParser(LAMMPSBaseParser):
    """Parser for LAMMPS MD calculations."""

    def __init__(self, node):
        """Initialize the instance of Lammps MD Parser."""
        super(MdParser, self).__init__(node)

    def parse(self, **kwargs):
        """Parse the retrieved folder and store results."""
        # retrieve resources
        resources = self.get_parsing_resources(kwargs, traj_in_temp=True)
        if resources.exit_code is not None:
            return resources.exit_code

        # parse log file
        log_data, exit_code = self.parse_log_file()
        if exit_code is not None:
            return exit_code

        traj_error = None
        if not resources.traj_paths:
            traj_error = self.exit_codes.ERROR_TRAJ_FILE_MISSING
        else:
            try:
                trajectory_data = LammpsTrajectory(resources.traj_paths[0])
                self.out("trajectory_data", trajectory_data)
            except Exception as err:
                traceback.print_exc()
                self.logger.error(str(err))
                traj_error = self.exit_codes.ERROR_TRAJ_PARSING

        # save results into node
        output_data = log_data["data"]
        if "units_style" in output_data:
            output_data.update(
                get_units_dict(
                    output_data["units_style"], ["distance", "time", "energy"]
                )
            )
        else:
            self.logger.warning("units missing in log")
        self.add_warnings_and_errors(output_data)
        self.add_standard_info(output_data)
        if "parameters" in self.node.get_incoming().all_link_labels():
            output_data["timestep_picoseconds"] = convert_units(
                self.node.inputs.parameters.dict.timestep,
                output_data["units_style"],
                "time",
                "picoseconds",
            )
        parameters_data = Dict(dict=output_data)
        self.out("results", parameters_data)

        # parse the system data file
        sys_data_error = None
        if resources.sys_paths:
            sys_data = ArrayData()
            try:
                with open(resources.sys_paths[0]) as handle:
                    names = handle.readline().strip().split()
                for i, col in enumerate(
                    np.loadtxt(resources.sys_paths[0], skiprows=1, unpack=True, ndmin=2)
                ):
                    sys_data.set_array(names[i], col)
            except Exception:
                traceback.print_exc()
                sys_data_error = self.exit_codes.ERROR_INFO_PARSING
            sys_data.set_attribute("units_style", output_data.get("units_style", None))
            self.out("system_data", sys_data)

        if output_data["errors"]:
            return self.exit_codes.ERROR_LAMMPS_RUN

        if traj_error:
            return traj_error

        if sys_data_error:
            return sys_data_error

        if not log_data.get("found_end", False):
            return self.exit_codes.ERROR_RUN_INCOMPLETE
