from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm import DataFactory

from aiida_lammps.common.raw_parsers import read_log_file2 as read_log_file, read_lammps_positions_and_forces, \
    get_units_dict

ArrayData = DataFactory('array')
ParameterData = DataFactory('parameter')
StructureData = DataFactory('structure')

import numpy as np




class OptimizeParser(Parser):
    """
    Simple Parser for LAMMPS.
    """

    def __init__(self, calc):
        """
        Initialize the instance of LammpsParser
        """
        super(OptimizeParser, self).__init__(calc)

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
        except KeyError:
            self.logger.error("No retrieved folder found")
            return False, ()

        # check what is inside the folder
        list_of_files = out_folder.get_folder_list()

        # OUTPUT file should exist
        if not self._calc._OUTPUT_FILE_NAME in list_of_files:
            successful = False
            self.logger.error("Output file not found")
            return successful, ()

        # Get file and do the parsing
        outfile = out_folder.get_abs_path( self._calc._OUTPUT_FILE_NAME)
        ouput_trajectory = out_folder.get_abs_path( self._calc._OUTPUT_TRAJECTORY_FILE_NAME)

        output_data, cell, stress_tensor = read_log_file(outfile)
        output_data.update(get_units_dict('metal', ["energy", "force"]))

        positions, forces, symbols, cell2 = read_lammps_positions_and_forces(ouput_trajectory)

        # look at warnings
        warnings = []
        with open(out_folder.get_abs_path( self._calc._SCHED_ERROR_FILE )) as f:
            errors = f.read()
        if errors:
            warnings = [errors]

        # ====================== prepare the output node ======================

        # save the outputs
        new_nodes_list = []

        # save optimized structure into node
        structure = StructureData(cell=cell)

        for i, position in enumerate(positions[-1]):
            structure.append_atom(position=position.tolist(),
                                  symbols=symbols[i])

        new_nodes_list.append(('output_structure', structure))

        # save forces into node
        array_data = ArrayData()
        array_data.set_array('forces', forces)
        array_data.set_array('stress', stress_tensor)

        new_nodes_list.append(('output_array', array_data))

        # add the dictionary with warnings
        output_data.update({'warnings': warnings})

        parameters_data = ParameterData(dict=output_data)
        new_nodes_list.append((self.get_linkname_outparams(), parameters_data))

        # add the dictionary with warnings
        # new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list
