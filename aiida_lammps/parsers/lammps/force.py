import traceback
from aiida.parsers.parser import Parser
from aiida.common import exceptions

from aiida.orm import Dict, ArrayData

from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_lammps_positions_and_forces_txt, read_log_file, get_units_dict


class ForceParser(Parser):
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

        # Check that the retrieved folder is there
        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()

        output_filename = self.node.get_option('output_filename')
        if output_filename not in list_of_files:
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING

        trajectory_filename = self.node.get_option('trajectory_name')
        if trajectory_filename not in list_of_files:
            return self.exit_codes.ERROR_TRAJ_FILE_MISSING

        output_txt = out_folder.get_object_content(output_filename)

        try:
            output_data, units = read_log_file(output_txt)
        except Exception:
            traceback.print_exc()
            return self.exit_codes.ERROR_LOG_PARSING

        trajectory_txt = out_folder.get_object_content(trajectory_filename)

        positions, forces, symbols, cell2 = read_lammps_positions_and_forces_txt(trajectory_txt)

        warnings = out_folder.get_object_content(self.node.get_option("scheduler_stderr"))
        # for some reason, errors may be in the stdout, but not the log.lammps
        stdout = out_folder.get_object_content(self.node.get_option("scheduler_stdout"))
        errors = [line for line in stdout.splitlines() if line.startswith("ERROR")]

        # ====================== prepare the output node ======================

        # add units used
        output_data.update(get_units_dict(units, ["energy", "force", "distance"]))

        # save forces and stresses into node
        array_data = ArrayData()
        array_data.set_array('forces', forces)
        self.out('arrays', array_data)

        # add the dictionary with warnings and errors
        output_data.update({'warnings': warnings})
        output_data.update({'errors': errors})
        output_data["parser_class"] = self.__class__.__name__
        output_data["parser_version"] = aiida_lammps_version

        parameters_data = Dict(dict=output_data)

        # self.out(self.node.get_linkname_outparams(), parameters_data)
        self.out('results', parameters_data)

        if output_data["errors"]:
            for error in output_data["errors"]:
                self.logger.error(error)
            return self.exit_codes.ERROR_LAMMPS_RUN
