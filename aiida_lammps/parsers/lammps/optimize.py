import numpy as np

from aiida.orm import ArrayData, Dict, StructureData

from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser
from aiida_lammps.common.raw_parsers import (
    iter_lammps_trajectories,
    get_units_dict,
    TRAJ_BLOCK,  # noqa: F401
)


class OptimizeParser(LAMMPSBaseParser):
    """Parser for LAMMPS optimization calculation."""

    def __init__(self, node):
        """Initialize the instance of Optimize Lammps Parser."""
        super(OptimizeParser, self).__init__(node)

    def parse(self, **kwargs):
        """Parses the datafolder, stores results."""
        resources = self.get_parsing_resources(kwargs)
        if resources.exit_code is not None:
            return resources.exit_code

        log_data, exit_code = self.parse_log_file(compute_stress=True)
        if exit_code is not None:
            return exit_code

        traj_error = None
        if not resources.traj_paths:
            traj_error = self.exit_codes.ERROR_TRAJ_FILE_MISSING
        else:
            try:
                array_data, structure = self.parse_traj_file(
                    resources.traj_paths[0], log_data["cell"], log_data["stress"]
                )
                self.out("structure", structure)
                self.out("arrays", array_data)
            except Exception as err:
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

    def parse_traj_file(self, trajectory_filename, cell, stress):
        with self.retrieved.open(trajectory_filename, "r") as handle:
            traj_steps = list(iter_lammps_trajectories(handle))
        if not traj_steps:
            raise IOError("trajectory file empty")

        forces = []
        positions = []
        elements = []
        charges = []
        for traj_step in traj_steps:  # type: TRAJ_BLOCK
            if not set(traj_step.field_names).issuperset(
                ["element", "x", "y", "z", "fx", "fy", "fz"]
            ):
                raise IOError(
                    "trajectory step {} does not contain required fields".format(
                        traj_step.timestep
                    )
                )

            fmap = {n: i for i, n in enumerate(traj_step.field_names)}

            positions.append(
                [[f[fmap["x"]], f[fmap["y"]], f[fmap["z"]]] for f in traj_step.fields]
            )
            forces.append(
                [
                    [f[fmap["fx"]], f[fmap["fy"]], f[fmap["fz"]]]
                    for f in traj_step.fields
                ]
            )
            elements.append([f[fmap["element"]] for f in traj_step.fields])
            if "q" in fmap:
                charges.append([f[fmap["q"]] for f in traj_step.fields])

        forces = np.array(forces, dtype=float)
        positions = np.array(positions, dtype=float)

        # save forces and stresses into node
        array_data = ArrayData()
        array_data.set_array("forces", forces)
        array_data.set_array("stress", stress)
        array_data.set_array("positions", positions)
        if charges:
            array_data.set_array("charges", np.array(charges, dtype=float))

        # save optimized structure into node
        # TODO clone input structure, then change cell and positions
        structure = StructureData(cell=cell)
        for element, position in zip(elements[-1], positions[-1]):
            structure.append_atom(position=position.tolist(), symbols=element)

        return array_data, structure
