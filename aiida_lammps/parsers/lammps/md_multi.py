import io
import os
import re
import traceback

from aiida.orm import ArrayData, Dict
import numpy as np

from aiida_lammps.common.raw_parsers import convert_units, get_units_dict
from aiida_lammps.data.trajectory import LammpsTrajectory
from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser


class MdMultiParser(LAMMPSBaseParser):
    """Parser for LAMMPS MDMulti calculations."""

    def __init__(self, node):
        """Initialize the instance of Lammps MD Parser."""
        super(MdMultiParser, self).__init__(node)

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
                trajectories = {
                    os.path.basename(traj_path).split("-")[0]: LammpsTrajectory(
                        traj_path
                    )
                    for traj_path in resources.traj_paths
                }
                self.out("trajectory", trajectories)
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
            output_data["stage_names"] = [
                s["name"] for s in self.node.inputs.parameters.dict.stages
            ]
        parameters_data = Dict(dict=output_data)
        self.out("results", parameters_data)

        # parse the system data file
        sys_data_error = None
        arrays = {}
        for sys_path in resources.sys_paths:
            stage_name = os.path.basename(sys_path).split("-")[0]
            sys_data = ArrayData()
            sys_data.set_attribute("units_style", output_data.get("units_style", None))
            try:
                with open(sys_path) as handle:
                    names = handle.readline().strip().split()
                for i, col in enumerate(
                    np.loadtxt(sys_path, skiprows=1, unpack=True, ndmin=2)
                ):
                    sys_data.set_array(names[i], col)
                arrays[stage_name] = sys_data
            except Exception:
                traceback.print_exc()
                sys_data_error = self.exit_codes.ERROR_INFO_PARSING
        if arrays:
            self.out("system", arrays)

        # retrieve the last restart file, per stage
        restart_map = {}
        for rpath in resources.restart_paths:
            rpath_base = os.path.basename(rpath)
            match = re.match(r"([^\-]*)\-.*\.([\d]+)", rpath_base)
            if match:
                stage, step = match.groups()
                if int(step) > restart_map.get(stage, (-1, None))[0]:
                    restart_map[stage] = (int(step), rpath)

        for stage, (step, rpath) in restart_map.items():
            with io.open(rpath, "rb") as handle:
                self.retrieved.put_object_from_filelike(
                    handle, os.path.basename(rpath), "wb", force=True
                )

        if output_data["errors"]:
            return self.exit_codes.ERROR_LAMMPS_RUN

        if traj_error:
            return traj_error

        if sys_data_error:
            return sys_data_error

        if not log_data.get("found_end", False):
            return self.exit_codes.ERROR_RUN_INCOMPLETE
