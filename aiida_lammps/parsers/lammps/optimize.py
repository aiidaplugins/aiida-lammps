from aiida.orm import ArrayData, Dict, StructureData

from aiida_lammps.parsers.lammps.base import LAMMPSBaseParser
from aiida_lammps.common.raw_parsers import read_lammps_positions_and_forces_txt, get_units_dict


class OptimizeParser(LAMMPSBaseParser):
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
        resources, exit_code = self.get_parsing_resources(kwargs)
        if exit_code is not None:
            return exit_code
        trajectory_filename, trajectory_filepath, info_filepath = resources

        output_data, cell, stress_tensor, units, exit_code = self.parse_log_file_long()
        if exit_code is not None:
            return exit_code
        output_data.update(get_units_dict(units, ["energy", "force", "distance"]))

        trajectory_txt = self.retrieved.get_object_content(trajectory_filename)
        positions, forces, symbols, cell2 = read_lammps_positions_and_forces_txt(trajectory_txt)

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

        # save results into node
        self.add_warnings_and_errors(output_data)
        self.add_standard_info(output_data)
        parameters_data = Dict(dict=output_data)
        self.out('results', parameters_data)

        if output_data["errors"]:
            return self.exit_codes.ERROR_LAMMPS_RUN
