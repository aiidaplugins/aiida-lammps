# Not working with Aiida 1.0

from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm.data.array import ArrayData
from aiida.orm.data.parameter import ParameterData

from aiida_lammps.common.raw_parsers import parse_dynaphopy_output, parse_quasiparticle_data
from aiida_phonopy.common.raw_parsers import parse_FORCE_CONSTANTS

import numpy as np




class DynaphopyParser(Parser):
    """
    Simple Parser for LAMMPS.
    """

    def __init__(self, calc):
        """
        Initialize the instance of LammpsParser
        """
        super(DynaphopyParser, self).__init__(calc)

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
        #if not self._calc._OUTPUT_FILE_NAME in list_of_files:
        #    successful = False
        #    self.logger.error("Output file not found")
        #    return successful, ()

        # Get file and do the parsing
        outfile = out_folder.get_abs_path( self._calc._OUTPUT_FILE_NAME)
        force_constants_file = out_folder.get_abs_path(self._calc._OUTPUT_FORCE_CONSTANTS)
        qp_file = out_folder.get_abs_path(self._calc._OUTPUT_QUASIPARTICLES)

        try:
            thermal_properties = parse_dynaphopy_output(outfile)
            quasiparticle_data = parse_quasiparticle_data(qp_file)
        except ValueError:
            pass

        try:
            force_constants = parse_FORCE_CONSTANTS(force_constants_file)
        except:
            pass

        # look at warnings
        warnings = []
        with open(out_folder.get_abs_path( self._calc._SCHED_ERROR_FILE )) as f:
            errors = f.read()
        if errors:
            warnings = [errors]

        # ====================== prepare the output node ======================

        # save the outputs
        new_nodes_list = []

        # save phonon data into node
        try:
            new_nodes_list.append(('quasiparticle_data', ParameterData(dict=quasiparticle_data)))
        except KeyError:  # keys not
            pass

        try:
            new_nodes_list.append(('thermal_properties', ParameterData(dict=thermal_properties)))
        except KeyError:  # keys not
            pass

        try:
            new_nodes_list.append(('force_constants', force_constants))
        except KeyError:  # keys not
            pass


        # add the dictionary with warnings
        new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list
