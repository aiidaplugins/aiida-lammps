import traceback

from aiida.orm import Dict

from aiida_lammps.common.raw_parsers import get_units_dict
from aiida_lammps.data.trajectory import LammpsTrajectory
from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser


class OptimizeParser(LAMMPSBaseParser):
    """Parser for LAMMPS optimization calculation."""

    def __init__(self, node):
        """Initialize the instance of Optimize Lammps Parser."""
        super(OptimizeParser, self).__init__(node)

    def parse(self, **kwargs):
        """Parses the datafolder, stores results."""
        resources = self.get_parsing_resources(kwargs, traj_in_temp=True)
        if resources.exit_code is not None:
            return resources.exit_code

        log_data, exit_code = self.parse_log_file()
        if exit_code is not None:
            return exit_code

        traj_error = None
        if not resources.traj_paths:
            traj_error = self.exit_codes.ERROR_TRAJ_FILE_MISSING
        else:
            try:
                trajectory_data = LammpsTrajectory(
                    resources.traj_paths[0],
                    aliases={
                        "stresses": ["c_stpa[{}]".format(i + 1) for i in range(6)],
                        "forces": ["fx", "fy", "fz"],
                    },
                )
                self.out("trajectory_data", trajectory_data)
                self.out(
                    "structure",
                    trajectory_data.get_step_structure(
                        -1, original_structure=self.node.inputs.structure
                    ),
                )
            except Exception as err:
                traceback.print_exc()
                self.logger.error(str(err))
                traj_error = self.exit_codes.ERROR_TRAJ_PARSING

        # save results into node
        output_data = log_data["data"]
        if "units_style" in output_data:
            output_data.update(
                get_units_dict(
                    output_data["units_style"],
                    ["energy", "force", "distance", "pressure"],
                )
            )
            output_data["stress_units"] = output_data.pop("pressure_units")
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
