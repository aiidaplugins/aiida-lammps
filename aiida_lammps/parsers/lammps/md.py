import os
from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm import DataFactory

from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_lammps_trajectory, get_units_dict
from aiida_lammps.utils import aiida_version, cmp_version

ArrayData = DataFactory('array')
ParameterData = DataFactory('parameter')
TrajectoryData = DataFactory('array.trajectory')


class MdParser(Parser):
    """
    Simple Parser for LAMMPS.
    """

    def __init__(self, calc):
        """
        Initialize the instance of LammpsParser
        """
        super(MdParser, self).__init__(calc)

    def parse_with_retrieved(self, retrieved):
        """
        Parses the datafolder, stores results.
        """

        # suppose at the start that the job is successful
        successful = True

        # select the folder object
        # Check that the retrieved folder is there
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
            temporary_folder = retrieved[self.retrieved_temporary_folder_key]
        except KeyError:
            self.logger.error("No retrieved folder found")
            return False, ()

        if aiida_version() < cmp_version('1.0.0a1'):
            get_temp_path = temporary_folder.get_abs_path
        else:
            get_temp_path = lambda x: os.path.join(temporary_folder, x)

        # check what is inside the folder
        list_of_files = out_folder.get_folder_list()

        # OUTPUT file should exist
        if not self._calc._OUTPUT_FILE_NAME in list_of_files:
            successful = False
            self.logger.error("Output file not found")
            return successful, ()

        # Get file and do the parsing
        outfile = out_folder.get_abs_path(self._calc._OUTPUT_FILE_NAME)
        output_trajectory = get_temp_path(self._calc._OUTPUT_TRAJECTORY_FILE_NAME)

        timestep = self._calc.inp.parameters.dict.timestep
        positions, step_ids, cells, symbols, time = read_lammps_trajectory(output_trajectory, timestep=timestep)

        # look at warnings
        warnings = []
        with open(out_folder.get_abs_path( self._calc._SCHED_ERROR_FILE )) as f:
            errors = f.read()
        if errors:
            warnings = [errors]

        # ====================== prepare the output node ======================

        # save the outputs
        new_nodes_list = []

        # save trajectory into node
        try:
            trajectory_data = TrajectoryData()
            trajectory_data.set_trajectory(step_ids, cells, symbols, positions, times=time)
            new_nodes_list.append(('trajectory_data', trajectory_data))
        except KeyError: # keys not found in json
            pass

        # add the dictionary with warnings
        param_dict = {'warnings': warnings,
                      "parser_class": self.__class__.__name__,
                      "parser_version": aiida_lammps_version}
        # add units used
        with open(get_temp_path(self._calc._INPUT_UNITS)) as f:
            units = f.readlines()[0].strip()
        param_dict.update(get_units_dict(units, ["distance"]))

        # TODO return energies per step

        new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict=param_dict)))

        return successful, new_nodes_list

