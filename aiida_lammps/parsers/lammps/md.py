from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm.data.array import ArrayData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.trajectory import TrajectoryData

import numpy as np

def read_lammps_trajectory(file_name,
                           limit_number_steps=100000000,
                           initial_cut=1,
                           end_cut=None,
                           timestep=1):

    import mmap
 #Time in picoseconds
 #Coordinates in Angstroms

    #Starting reading
    print("Reading LAMMPS trajectory")
    print("This could take long, please wait..")

    #Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    step_ids = []
    data = []
    cells = []
    counter = 0
    bounds = None
    number_of_atoms = None

    lammps_labels = False

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        while True:

            counter += 1

            #Read time steps
            position_number=file_map.find('TIMESTEP')
            if position_number < 0: break

            file_map.seek(position_number)
            file_map.readline()
            step_ids.append(float(file_map.readline()))

            if number_of_atoms is None:
                #Read number of atoms
                position_number=file_map.find('NUMBER OF ATOMS')
                file_map.seek(position_number)
                file_map.readline()
                number_of_atoms = int(file_map.readline())

            if True:
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
            lammps_labels=file_map.readline()


            #Initial cut control
            if initial_cut > counter:
                continue

            #Reading coordinates
            read_coordinates = []
            read_elements = []
            for i in range (number_of_atoms):
                line = file_map.readline().split()[0:number_of_dimensions+1]
                read_coordinates.append(line[1:number_of_dimensions+1])
                read_elements.append(line[0])
            try:
                data.append(np.array(read_coordinates, dtype=float)) #in angstroms
           #     print read_coordinates
            except ValueError:
                print("Error reading step {0}".format(counter))
                break
        #        print(read_coordinates)
            #security routine to limit maximum of steps to read and put in memory
            if limit_number_steps+initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break


    file_map.close()

    data = np.array(data)
    step_ids = np.array(step_ids, dtype=int)
    cells = np.array(cells)
    elements = np.array(read_elements)
    time = np.array(step_ids)*timestep
    return data, step_ids, cells, elements, time


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

        timestep = self._calc.inp.parameters.dict.timestep
        positions, step_ids, cells, symbols, time = read_lammps_trajectory(ouput_trajectory, timestep=timestep)

        # Delete trajectory once parsed
        try:
            import os
            os.remove(ouput_trajectory)
        except:
            pass

#        force_constants = parse_FORCE_CONSTANTS(outfile)

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
        new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list

