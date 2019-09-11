import io
import traceback

import numpy as np

from aiida.orm import Dict, TrajectoryData, ArrayData

from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser
from aiida_lammps.common.raw_parsers import (
    convert_units,
    iter_lammps_trajectories,
    get_units_dict,
    TRAJ_BLOCK,  # noqa: F401
)


class MdParser(LAMMPSBaseParser):
    """Parser for LAMMPS MD calculations."""

    def __init__(self, node):
        """Initialize the instance of Lammps MD Parser."""
        super(MdParser, self).__init__(node)

    def parse(self, **kwargs):
        """Parse the retrieved folder and store results."""
        # retrieve resources
        resources, exit_code = self.get_parsing_resources(
            kwargs, traj_in_temp=True, sys_info=True
        )
        if exit_code is not None:
            return exit_code
        trajectory_filename, trajectory_filepath, info_filepath = resources

        # parse log file
        log_data, exit_code = self.parse_log_file()
        if exit_code is not None:
            return exit_code

        traj_error = None
        try:
            trajectory_data = self.parse_traj_file(trajectory_filepath)
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
        if info_filepath:
            sys_data = ArrayData()
            try:
                with open(info_filepath) as handle:
                    names = handle.readline().strip().split()
                for i, col in enumerate(
                    np.loadtxt(info_filepath, skiprows=1, unpack=True)
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

    def parse_traj_file(self, trajectory_filepath):
        with io.open(trajectory_filepath, "r") as handle:
            traj_steps = list(iter_lammps_trajectories(handle))
        if not traj_steps:
            raise IOError("trajectory file empty")

        positions = []
        elements = None
        charges = []
        cells = []
        timesteps = []
        for traj_step in traj_steps:  # type: TRAJ_BLOCK
            if traj_step.timestep == 0:
                continue
            if timesteps and traj_step.timestep == timesteps[-1]:
                # This can occur if the dump is reset (using `undump`, `dump`)
                continue
            if not set(traj_step.field_names).issuperset(["element", "x", "y", "z"]):
                raise IOError(
                    "trajectory step {} does not contain required fields".format(
                        traj_step.timestep
                    )
                )
            fmap = {n: i for i, n in enumerate(traj_step.field_names)}
            elements = [f[fmap["element"]] for f in traj_step.fields]
            positions.append(
                [[f[fmap["x"]], f[fmap["y"]], f[fmap["z"]]] for f in traj_step.fields]
            )
            if "q" in fmap:
                charges.append([f[fmap["q"]] for f in traj_step.fields])
            cells.append(traj_step.cell)
            timesteps.append(traj_step.timestep)

        # save trajectories into node
        trajectory_data = TrajectoryData()
        trajectory_data.set_trajectory(
            elements,
            np.array(positions, dtype=float),
            stepids=np.array(timesteps, dtype=int),
            cells=np.array(cells),
            # times=time,
        )
        if charges:
            trajectory_data.set_array("charges", np.array(charges, dtype=float))

        return trajectory_data
