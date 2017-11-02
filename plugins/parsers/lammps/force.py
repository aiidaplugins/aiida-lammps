from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError

from aiida.orm import DataFactory

ArrayData = DataFactory('array')
ParameterData = DataFactory('parameter')


import numpy as np

def read_log_file(logfile):

    f = open(logfile, 'r')
    data = f.readlines()

    data_dict = {}
    for i, line in enumerate(data):
        if 'Loop time' in line:
            energy = float(data[i-1].split()[4])
            data_dict['energy'] = energy

    return data_dict

def read_lammps_forces(file_name):

    import mmap
    # Time in picoseconds
    # Coordinates in Angstroms

    # Starting reading

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    cells = []

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        # Read time steps
        position_number=file_map.find('TIMESTEP')

        file_map.seek(position_number)
        file_map.readline()


        #Read number of atoms
        position_number=file_map.find('NUMBER OF ATOMS')
        file_map.seek(position_number)
        file_map.readline()
        number_of_atoms = int(file_map.readline())


        #Read cell
        position_number=file_map.find('ITEM: BOX')
        file_map.seek(position_number)
        file_map.readline()


        bounds = []
        for i in range(3):
            bounds.append(file_map.readline().split())

        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T ,axis=1)


        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy+xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy+xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi-xlo, xy,  xz],
                               [0,  yhi-ylo,  yz],
                               [0,   0,  zhi-zlo]])

        cells.append(super_cell.T)

        position_number = file_map.find('ITEM: ATOMS')
        file_map.seek(position_number)
        file_map.readline()

        #Reading forces
        forces = []
        read_elements = []
        for i in range (number_of_atoms):
            line = file_map.readline().split()[0:number_of_dimensions+1]
            forces.append(line[1:number_of_dimensions+1])
            read_elements.append(line[0])

    file_map.close()

    forces = np.array([forces], dtype=float)

    return forces


class ForceParser(Parser):
    """
    Simple Parser for LAMMPS.
    """

    def __init__(self, calc):
        """
        Initialize the instance of LammpsParser
        """
        super(ForceParser, self).__init__(calc)

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

        outputa_data = read_log_file(outfile)
        forces = read_lammps_forces(ouput_trajectory)

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

        array_data = ArrayData()
        array_data.set_array('forces', forces)
        new_nodes_list.append(('output_array', array_data))

        # add the dictionary with warnings
        outputa_data.update({'warnings': warnings})

        parameters_data = ParameterData(dict=outputa_data)
        new_nodes_list.append((self.get_linkname_outparams(), parameters_data))

        # add the dictionary with warnings
        # new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list
