from aiida.parsers.parser import Parser
from aiida.common import OutputParsingError, exceptions
from aiida.orm import ArrayData, Dict, StructureData

from aiida_lammps import __version__ as aiida_lammps_version
from aiida_lammps.common.raw_parsers import read_log_file_long as read_log_file, read_lammps_positions_and_forces_txt, \
    get_units_dict


class OptimizeParser(Parser):
    """
    Simple Optimize Parser for LAMMPS.
    """

    def __init__(self, node):
        """
        Initialize the instance of Optimize LammpsParser
        """
        super(OptimizeParser, self).__init__(node)

    def parse(self, **kwargs):
        """
        Parses the datafolder, stores results.
        """

        # Check that the retrieved folder is there
        try:
            out_folder = self.retrieved
            # temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # check what is inside the folder
        list_of_files = out_folder.list_object_names()

        output_filename = self.node.get_option('output_filename')
        if output_filename not in list_of_files:
            raise OutputParsingError('Output file not found')

        trajectory_filename = self.node.get_option('trajectory_name')
        if trajectory_filename not in list_of_files:
            raise OutputParsingError('Trajectory file not found')

        # parser_name = self.node.options['parser_name'].default

        # Get files and do the parsing
        output_txt = out_folder.get_object_content(output_filename)
        output_data, cell, stress_tensor, units = read_log_file(output_txt)

        trajectory_txt = out_folder.get_object_content(trajectory_filename)
        positions, forces, symbols, cell2 = read_lammps_positions_and_forces_txt(trajectory_txt)

        warnings = out_folder.get_object_content('_scheduler-stderr.txt')

        # ====================== prepare the output node ======================

        # add units used
        output_data.update(get_units_dict(units, ["energy", "force", "distance"]))

        # save optimized structure into node
        structure = StructureData(cell=cell)

        for i, position in enumerate(positions[-1]):
            structure.append_atom(position=position.tolist(),
                                  symbols=symbols[i])

        self.out('structure', structure)

        # save forces and stresses into node
        array_data = ArrayData()
        array_data.set_array('forces', forces)
        array_data.set_array('stress', stress_tensor)
        array_data.set_array('positions', positions)
        self.out('arrays', array_data)

        # add the dictionary with warnings
        output_data.update({'warnings': warnings})
        output_data["parser_class"] = self.__class__.__name__
        output_data["parser_version"] = aiida_lammps_version

        parameters_data = Dict(dict=output_data)

        # self.out(self.node.get_linkname_outparams(), parameters_data)
        self.out('results', parameters_data)
