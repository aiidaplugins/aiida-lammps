import os
from aiida.parsers.parser import Parser
from aiida.common import OutputParsingError, exceptions
from aiida.orm import Dict, TrajectoryData
from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_lammps_trajectory, get_units_dict, read_log_file


class MdParser(Parser):
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

        # Check that the retrieved folder is there
        try:
            out_folder = self.retrieved
            temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()
        list_of_temp_files = os.listdir(temporary_folder)

        # check outout folder
        output_filename = self.node.get_option('output_filename')
        if output_filename not in list_of_files:
            raise OutputParsingError('Output file not found')

        # check temporal folder
        trajectory_filename = self.node.get_option('trajectory_name')
        if trajectory_filename not in list_of_temp_files:
            raise OutputParsingError('Trajectory file not found')

        # Read trajectory from temporal folder
        trajectory_filepath = temporary_folder + '/' + self.node.get_option('trajectory_name')
        timestep = self.node.inputs.parameters.dict.timestep
        positions, step_ids, cells, symbols, time = read_lammps_trajectory(trajectory_filepath, timestep=timestep)

        # save trajectory into node
        try:
            trajectory_data = TrajectoryData()
            trajectory_data.set_trajectory(symbols, positions, stepids=step_ids, cells=cells, times=time)
            self.out('trajectory_data', trajectory_data)

        except KeyError:  # keys not found in json
            raise OutputParsingError('Trajectory not parsed correctly')

        # Read other data from output folder
        warnings = out_folder.get_object_content('_scheduler-stderr.txt')

        output_txt = out_folder.get_object_content(output_filename)

        output_data, units = read_log_file(output_txt)
        output_data.update(get_units_dict(units, ["distance", "time"]))

        # add the dictionary with warnings
        output_data.update({'warnings': warnings})
        output_data["parser_class"] = self.__class__.__name__
        output_data["parser_version"] = aiida_lammps_version

        parameters_data = Dict(dict=output_data)

        self.out('results', parameters_data)
