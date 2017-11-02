from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm.data.array import ArrayData
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
    fcfile.close()
    return force_constants


def parse_quasiparticle_data(qp_file):
    import yaml

    f = open(qp_file, "r")
    quasiparticle_data = yaml.load(f)
    f.close()
    return quasiparticle_data


def parse_dynaphopy_output(file):

    thermal_properties = None
    f = open(file, 'r')
    data_lines = f.readlines()

    indices = []
    q_points = []
    for i, line in enumerate(data_lines):
        if 'Q-point' in line:
    #        print i, np.array(line.replace(']', '').replace('[', '').split()[4:8], dtype=float)
            indices.append(i)
            q_points.append(np.array(line.replace(']', '').replace('[', '').split()[4:8],dtype=float))

    indices.append(len(data_lines))

    phonons = {}
    for i, index in enumerate(indices[:-1]):

        fragment = data_lines[indices[i]: indices[i+1]]
        if 'kipped' in fragment:
            continue
 #       print q_points[i], i
        phonon_modes = {}
        for j, line in enumerate(fragment):
            if 'Peak' in line:
                number = line.split()[2]
                phonon_mode = {'width':     float(fragment[j+2].split()[1]),
                               'positions': float(fragment[j+3].split()[1]),
                               'shift':     float(fragment[j+12].split()[2])}
                phonon_modes.update({number: phonon_mode})

            if 'Thermal' in line:
                free_energy = float(fragment[j+4].split()[4])
                entropy = float(fragment[j+5].split()[3])
                cv = float(fragment[j+6].split()[3])
                total_energy = float(fragment[j+7].split()[4])

                temperature = float(fragment[j].split()[5].replace('(',''))

                thermal_properties = {'temperature': temperature,
                                      'free_energy': free_energy,
                                      'entropy': entropy,
                                      'cv': cv,
                                      'total_energy': total_energy}


        phonon_modes.update({'q_point': q_points[i].tolist()})

        phonons.update({'wave_vector_'+str(i): phonon_modes})

        f.close()

    return phonons, thermal_properties


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
        if not self._calc._OUTPUT_FILE_NAME in list_of_files:
            successful = False
            self.logger.error("Output file not found")
            return successful, ()

        # Get file and do the parsing
        outfile = out_folder.get_abs_path( self._calc._OUTPUT_FILE_NAME)
        force_constants_file = out_folder.get_abs_path(self._calc._OUTPUT_FORCE_CONSTANTS)
        qp_file = out_folder.get_abs_path(self._calc._OUTPUT_QUASIPARTICLES)

        try:
            quasiparticle_data, thermal_properties = parse_dynaphopy_output(outfile)
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
            array_data = ArrayData()
            array_data.set_array('force_constants', force_constants)
            new_nodes_list.append(('array_data', array_data))
        except KeyError:  # keys not
            pass


        # add the dictionary with warnings
        new_nodes_list.append((self.get_linkname_outparams(), ParameterData(dict={'warnings': warnings})))

        return successful, new_nodes_list
