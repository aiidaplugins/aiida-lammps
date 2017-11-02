from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm.data.array import ArrayData
from aiida.orm.data.force_constants import ForceConstants

from aiida.orm.data.parameter import ParameterData

import numpy as np


def parse_FORCE_CONSTANTS(filename):
    fcfile = open(filename)
    num = int((fcfile.readline().strip().split())[0])
    force_constants = np.zeros((num, num, 3, 3), dtype=float)
    for i in range(num):
        for j in range(num):
            fcfile.readline()
            tensor = []
            for k in range(3):
                tensor.append([float(x) for x in fcfile.readline().strip().split()])
            force_constants[i, j] = np.array(tensor)
    return force_constants


class PhonopyParser(Parser):
    """
    Parser the FORCE_CONSTANTS file of phonopy.
    """

    def __init__(self, calc):
        """
        Initialize the instance of PhonopyParser
        """
        super(PhonopyParser, self).__init__(calc)

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
        outfile = out_folder.get_abs_path(self._calc._OUTPUT_FILE_NAME)

        force_constants = parse_FORCE_CONSTANTS(outfile)

        # look at warnings
        warnings = []
        with open(out_folder.get_abs_path(self._calc._SCHED_ERROR_FILE)) as f:
            errors = f.read()
        if errors:
            warnings = [errors]

        # ====================== prepare the output node ======================

        # save the outputs
        new_nodes_list = []

        # save force constants into node
        try:
            #array_data = ArrayData()
            #array_data.set_array('force_constants', force_constants)

            new_nodes_list.append(('force_constants', ForceConstants(array=force_constants)))
        except KeyError:  # keys not found in json
            pass

        # add the dictionary with warnings
        new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list
