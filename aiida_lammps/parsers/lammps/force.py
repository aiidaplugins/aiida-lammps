from aiida.orm import Dict, ArrayData

from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser
from aiida_lammps.common.raw_parsers import read_lammps_positions_and_forces_txt, get_units_dict


class ForceParser(LAMMPSBaseParser):
    """
    Simple Parser for LAMMPS.
    """

    def __init__(self, node):
        """
        Initialize the instance of Force LammpsParser
        """
        super(ForceParser, self).__init__(node)

    def parse(self, **kwargs):
        """
        Parses the datafolder, stores results.
        """
        # retrieve resources
        resources, exit_code = self.get_parsing_resources(kwargs)
        if exit_code is not None:
            return exit_code
        trajectory_filename, trajectory_filepath, info_filepath = resources

        # parse log file
        log_data, exit_code = self.parse_log_file()
        if exit_code is not None:
            return exit_code

        # parse trajectory file
        trajectory_txt = self.retrieved.get_object_content(trajectory_filename)
        if not trajectory_txt:
            self.logger.error("trajectory file empty")
            return self.exit_codes.ERROR_TRAJ_PARSING
        positions, forces, charges, symbols, cell2 = read_lammps_positions_and_forces_txt(
            trajectory_txt)

        # save forces and stresses into node
        array_data = ArrayData()
        array_data.set_array('forces', forces)
        if charges is not None:
            array_data.set_array('charges', charges)
        self.out('arrays', array_data)

        # save results into node
        output_data = log_data["data"]
        if 'units_style' in output_data:
            output_data.update(get_units_dict(output_data['units_style'],
                                              ["energy", "force", "distance"]))
        else:
            self.logger.warning("units missing in log")
        self.add_warnings_and_errors(output_data)
        self.add_standard_info(output_data)
        parameters_data = Dict(dict=output_data)
        self.out('results', parameters_data)

        if output_data["errors"]:
            return self.exit_codes.ERROR_LAMMPS_RUN
