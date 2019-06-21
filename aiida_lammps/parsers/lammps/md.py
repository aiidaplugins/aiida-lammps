import traceback
import numpy as np
from aiida.orm import Dict, TrajectoryData, ArrayData

from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser
from aiida_lammps.common.raw_parsers import read_lammps_trajectory, get_units_dict


class MdParser(LAMMPSBaseParser):
    """
    Simple Parser for LAMMPS.
    """

    def __init__(self, node):
        """
        Initialize the instance of MDLammpsParser
        """
        super(MdParser, self).__init__(node)

    def parse(self, **kwargs):
        """
        Parses the datafolder, stores results.
        """
        # retrieve resources
        resources, exit_code = self.get_parsing_resources(
            kwargs, traj_in_temp=True, sys_info=True)
        if exit_code is not None:
            return exit_code
        trajectory_filename, trajectory_filepath, info_filepath = resources

        # parse log file
        log_data, exit_code = self.parse_log_file()
        if exit_code is not None:
            return exit_code

        # parse trajectory file
        try:
            timestep = self.node.inputs.parameters.dict.timestep
            positions, charges, step_ids, cells, symbols, time = read_lammps_trajectory(
                trajectory_filepath, timestep=timestep,
                log_warning_func=self.logger.warning)
        except Exception:
            traceback.print_exc()
            return self.exit_codes.ERROR_TRAJ_PARSING

        # save results into node
        output_data = log_data["data"]
        if 'units_style' in output_data:
            output_data.update(get_units_dict(output_data['units_style'],
                                              ["distance", "time", "energy"]))
        else:
            self.logger.warning("units missing in log")
        self.add_warnings_and_errors(output_data)
        self.add_standard_info(output_data)
        parameters_data = Dict(dict=output_data)
        self.out('results', parameters_data)

        # save trajectories into node
        trajectory_data = TrajectoryData()
        trajectory_data.set_trajectory(
            symbols, positions, stepids=step_ids, cells=cells, times=time)
        if charges is not None:
            trajectory_data.set_array('charges', charges)       
        self.out('trajectory_data', trajectory_data)

        # parse the system data file
        if info_filepath:
            sys_data = ArrayData()
            try:
                with open(info_filepath) as handle:
                    names = handle.readline().strip().split()
                for i, col in enumerate(np.loadtxt(info_filepath, skiprows=1, unpack=True)):
                    sys_data.set_array(names[i], col)
            except Exception:
                traceback.print_exc()
                return self.exit_codes.ERROR_INFO_PARSING
            sys_data.set_attribute('units_style', output_data.get('units_style', None))
            self.out('system_data', sys_data)

        if output_data["errors"]:
            return self.exit_codes.ERROR_LAMMPS_RUN
