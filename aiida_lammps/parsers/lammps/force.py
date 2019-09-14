import numpy as np

from aiida.orm import Dict, ArrayData

from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser
from aiida_lammps.common.raw_parsers import (
    iter_lammps_trajectories,
    get_units_dict,
    TRAJ_BLOCK,  # noqa: F401
)


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
            traj_steps = list(iter_lammps_trajectories(handle))
        if not traj_steps:
            raise IOError("trajectory file empty")
        if len(traj_steps) > 1:
            raise IOError("trajectory file has multiple steps (expecting only one)")

        traj_step = traj_steps[0]  # type: TRAJ_BLOCK

        array_data = ArrayData()

        try:
            fx_idx = traj_step.field_names.index("fx")
            fy_idx = traj_step.field_names.index("fy")
            fz_idx = traj_step.field_names.index("fz")
        except ValueError:
            raise IOError("trajectory file does not contain fields fx fy fz")

        forces = [[f[fx_idx], f[fy_idx], f[fz_idx]] for f in traj_step.fields]

        array_data.set_array("forces", np.array(forces, dtype=float))

        if "q" in traj_step.field_names:
            q_idx = traj_step.field_names.index("q")
            charges = [f[q_idx] for f in traj_step.fields]
            array_data.set_array("charges", np.array(charges, dtype=float))

        return array_data
