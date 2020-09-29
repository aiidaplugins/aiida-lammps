from aiida.orm import ArrayData, Dict
import numpy as np

from aiida_lammps.common.parse_trajectory import TRAJ_BLOCK  # noqa: F401
from aiida_lammps.common.parse_trajectory import iter_trajectories
from aiida_lammps.common.raw_parsers import get_units_dict
from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser


class ForceParser(LAMMPSBaseParser):
    """Parser for LAMMPS single point energy calculation."""

    def __init__(self, node):
        """Initialize the instance of Force Lammps Parser."""
        super(ForceParser, self).__init__(node)

    def parse(self, **kwargs):
        """Parse the retrieved files and store results."""
        # retrieve resources
        resources = self.get_parsing_resources(kwargs)
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
                array_data = self.parse_traj_file(resources.traj_paths[0])
                self.out("arrays", array_data)
            except IOError as err:
                self.logger.error(str(err))
                traj_error = self.exit_codes.ERROR_TRAJ_PARSING

        # save results into node
        output_data = log_data["data"]
        if "units_style" in output_data:
            output_data.update(
                get_units_dict(
                    output_data["units_style"], ["energy", "force", "distance"]
                )
            )
        else:
            self.logger.warning("units missing in log")
        self.add_warnings_and_errors(output_data)
        self.add_standard_info(output_data)
        parameters_data = Dict(dict=output_data)
        self.out("results", parameters_data)

        if output_data["errors"]:
            return self.exit_codes.ERROR_LAMMPS_RUN

        if traj_error:
            return traj_error

        if not log_data.get("found_end", False):
            return self.exit_codes.ERROR_RUN_INCOMPLETE

    def parse_traj_file(self, trajectory_filename):
        with self.retrieved.open(trajectory_filename, "r") as handle:
            traj_steps = list(iter_trajectories(handle))
        if not traj_steps:
            raise IOError("trajectory file empty")
        if len(traj_steps) > 1:
            raise IOError("trajectory file has multiple steps (expecting only one)")

        traj_step = traj_steps[0]  # type: TRAJ_BLOCK

        for field in ["fx", "fy", "fz"]:
            if field not in traj_step.atom_fields:
                raise IOError(
                    "trajectory file does not contain fields {}".format(field)
                )

        array_data = ArrayData()

        array_data.set_array(
            "forces",
            np.array(
                [
                    traj_step.atom_fields["fx"],
                    traj_step.atom_fields["fy"],
                    traj_step.atom_fields["fz"],
                ],
                dtype=float,
            ).T,
        )

        if "q" in traj_step.atom_fields:
            array_data.set_array(
                "charges", np.array(traj_step.atom_fields["q"], dtype=float)
            )

        return array_data
