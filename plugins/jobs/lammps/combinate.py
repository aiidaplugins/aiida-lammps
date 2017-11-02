from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty

from potentials import LammpsPotential
import numpy as np


def get_supercell(structure, supercell_shape):
    import itertools

    symbols = np.array([site.kind_name for site in structure.sites])
    positions = np.array([site.position for site in structure.sites])
    cell = np.array(structure.cell)
    supercell_shape = np.array(supercell_shape.dict.shape)

    supercell_array = np.dot(cell, np.diag(supercell_shape))

    supercell = StructureData(cell=supercell_array)
    for k in range(positions.shape[0]):
        for r in itertools.product(*[range(i) for i in supercell_shape[::-1]]):
            position = positions[k, :] + np.dot(np.array(r[::-1]), cell)
            symbol = symbols[k]
            supercell.append_atom(position=position, symbols=symbol)

    return supercell


def get_FORCE_CONSTANTS_txt(force_constants):

    force_constants = force_constants.get_array('force_constants')

    fc_shape = force_constants.shape
    fc_txt = "%4d\n" % (fc_shape[0])
    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            fc_txt += "%4d%4d\n" % (i+1, j+1)
            for vec in force_constants[i][j]:
                fc_txt +=("%22.15f"*3 + "\n") % tuple(vec)

    return fc_txt


def structure_to_poscar(structure):

    atom_type_unique = np.unique([site.kind_name for site in structure.sites], return_index=True)[1]
    labels = np.diff(np.append(atom_type_unique, [len(structure.sites)]))

    poscar = ' '.join(np.unique([site.kind_name for site in structure.sites]))
    poscar += '\n1.0\n'
    cell = structure.cell
    for row in cell:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)
    poscar += ' '.join(np.unique([site.kind_name for site in structure.sites]))+'\n'
    poscar += ' '.join(np.array(labels, dtype=str))+'\n'
    poscar += 'Cartesian\n'
    for site in structure.sites:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*site.position)

    return poscar


def parameters_to_input_file(parameters_object):

    parameters = parameters_object.get_dict()
    input_file = ('STRUCTURE FILE POSCAR\nPOSCAR\n\n')
    input_file += ('FORCE CONSTANTS\nFORCE_CONSTANTS\n\n')
    input_file += ('PRIMITIVE MATRIX\n')
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[2])
    input_file += ('\n')
    input_file += ('SUPERCELL MATRIX PHONOPY\n')
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[2])
    input_file += ('\n')

    return input_file


def generate_LAMMPS_structure(structure):
    import numpy as np

    types = [site.kind_name for site in structure.sites]

    type_index_unique = np.unique(types, return_index=True)[1]
    count_index_unique = np.diff(np.append(type_index_unique, [len(types)]))

    atom_index = []
    for i, index in enumerate(count_index_unique):
        atom_index += [i for j in range(index)]

    masses = [site.mass for site in structure.kinds]
    positions = [site.position for site in structure.sites]

    number_of_atoms = len(positions)

    lammps_data_file = 'Generated using dynaphopy\n\n'
    lammps_data_file += '{0} atoms\n\n'.format(number_of_atoms)
    lammps_data_file += '{0} atom types\n\n'.format(len(masses))

    cell = np.array(structure.cell)

    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])

    alpha = np.arccos(np.dot(cell[1], cell[2])/(c*b))
    gamma = np.arccos(np.dot(cell[1], cell[0])/(a*b))
    beta = np.arccos(np.dot(cell[2], cell[0])/(a*c))

    xhi = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    yhi = np.sqrt(pow(b,2)- pow(xy,2))
    yz = (b*c*np.cos(alpha)-xy * xz)/yhi
    zhi = np.sqrt(pow(c,2)-pow(xz,2)-pow(yz,2))

    xhi = xhi + max(0,0, xy, xz, xy+xz)
    yhi = yhi + max(0,0, yz)

    lammps_data_file += '\n{0:20.10f} {1:20.10f} xlo xhi\n'.format(0, xhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} ylo yhi\n'.format(0, yhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} zlo zhi\n'.format(0, zhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f} xy xz yz\n\n'.format(xy, xz, yz)

    lammps_data_file += 'Masses\n\n'

    for i, mass in enumerate(masses):
        lammps_data_file += '{0} {1:20.10f} \n'.format(i+1, mass)


    lammps_data_file += '\nAtoms\n\n'
    for i, row in enumerate(positions):
        lammps_data_file += '{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(i+1, atom_index[i]+1, row[0],row[1],row[2])

    return lammps_data_file


def generate_LAMMPS_input(parameters,
                          potential_obj,
                          structure_file='potential.pot',
                          trajectory_file='trajectory.lammpstr',
                          command=None):

    random_number = np.random.randint(10000000)

    names_str = ' '.join(potential_obj._names)

    lammps_input_file = 'units           metal\n'
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      atomic\n'
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    lammps_input_file += 'timestep        {}\n'.format(parameters.dict.timestep)

    lammps_input_file += 'thermo_style    custom step etotal temp vol press\n'
    lammps_input_file += 'thermo          100000\n'

    lammps_input_file += 'velocity        all create {0} {1} dist gaussian mom yes\n'.format(parameters.dict.temperature, random_number)
    lammps_input_file += 'velocity        all scale {}\n'.format(parameters.dict.temperature)

    lammps_input_file += 'fix             int all nvt temp {0} {0} {1}\n'.format(parameters.dict.temperature, parameters.dict.thermostat_variable)

    lammps_input_file += 'run             {}\n'.format(parameters.dict.equilibrium_steps)
    lammps_input_file += 'reset_timestep  0\n'

    lammps_input_file += 'dump            aiida all custom {0} {1} x y z\n'.format(parameters.dict.dump_rate, trajectory_file)
    lammps_input_file += 'dump_modify     aiida format "%16.10f %16.10f %16.10f"\n'
    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)

    lammps_input_file += 'run             {}\n'.format(parameters.dict.total_steps)

    if command:
        lammps_input_file += 'shell       {}\n'.format(command)
        lammps_input_file += 'shell       rm {}\n'.format(trajectory_file)

    lammps_input_file += 'print           "end of script" \n'.format(1000)
    lammps_input_file += 'run             {}\n'.format(1000)


    return lammps_input_file


def generate_LAMMPS_potential(pair_style):


    potential_file = '# Potential file generated by aiida plugin (please check citation in the orignal file)\n'
    for key, value in pair_style.dict.data.iteritems():
         potential_file += '{}    {}\n'.format(key, value)

    return potential_file


class CombinateCalculation(JobCalculation):
    """
    A basic plugin for calculating force constants using Lammps.

    Requirement: the node should be able to import phonopy
    """

    def _init_internal_params(self):
        super(CombinateCalculation, self)._init_internal_params()

        self._INPUT_FILE_NAME = 'input.in'
        self._INPUT_POTENTIAL = 'potential.pot'
        self._INPUT_STRUCTURE = 'input.data'

        self._OUTPUT_TRAJECTORY_FILE_NAME = 'trajectory.lampstrj'

  #      self._default_parser = "lammps.md"

        self._INPUT_CELL = 'POSCAR'
        self._INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'
        self._INPUT_FILE_NAME_DYNA = 'input_dynaphopy'
        self._OUTPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS_OUT'
        self._OUTPUT_FILE_NAME = 'OUTPUT'
        self._OUTPUT_QUASIPARTICLES = 'quasiparticles_data.yaml'

        self._default_parser = 'dynaphopy'


    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the namelists class.
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            "parameters": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters',
               'docstring': ("Use a node that specifies the lammps input data "
                             "for the namelists"),
               },
            "parameters_dynaphopy": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters_phonopy',
               'docstring': ("Use a node that specifies the dynaphopy input data "
                             "for the namelists"),
               },
            "force_constants": {
               'valid_types': ArrayData,
               'additional_parameter': None,
               'linkname': 'force_constants',
               'docstring': ("Use a node that specifies the force_constants "
                             "for the namelists"),
               },
            "potential": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'potential',
               'docstring': ("Use a node that specifies the lammps potential "
                             "for the namelists"),
               },
            "structure": {
               'valid_types': StructureData,
               'additional_parameter': None,
               'linkname': 'structure',
               'docstring': "Use a node for the structure",
               },
            "supercell_md": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'supercell_md',
               'docstring': "Use a node for the supercell MD shape",
               },
         })
        return retdict

    def _prepare_for_submission(self,tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """

        try:
            parameters_data = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError("No parameters specified for this "
                                       "calculation")

        if not isinstance(parameters_data, ParameterData):
            raise InputValidationError("parameters is not of type "
                                       "ParameterData")

        try:
            parameters_data_dynaphopy = inputdict.pop(self.get_linkname('parameters_dynaphopy'))
            force_constants = inputdict.pop(self.get_linkname('force_constants'))

        except KeyError:
           raise InputValidationError("No dynaphopy parameters specified for this "
                                       "calculation")

        try:
            potential_data = inputdict.pop(self.get_linkname('potential'))
        except KeyError:
            raise InputValidationError("No potential specified for this "
                                       "calculation")

        if not isinstance(potential_data, ParameterData):
            raise InputValidationError("potential is not of type "
                                       "ParameterData")

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError("no structure is specified for this calculation")

        try:
            supercell_shape = inputdict.pop(self.get_linkname('supercell_md'))
        except KeyError:
            raise InputValidationError("no supercell is specified for this calculation")


        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("no code is specified for this calculation")

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        time_step = parameters_data.dict.timestep

        # Dynaphopy command
        cmdline_params = ['/usr/bin/python', '/home/abel/BIN/dynaphopy', self._INPUT_FILE_NAME_DYNA,
                          self._OUTPUT_TRAJECTORY_FILE_NAME,
                          '-ts', '{}'.format(time_step), '--silent',
                          '-sfc', self._OUTPUT_FORCE_CONSTANTS, '-thm', # '--resolution 0.05',
                          '-psm', '2', '--normalize_dos', '-sdata']  # PS algorithm

        if 'temperature' in parameters_data.get_dict():
            cmdline_params.append('--temperature')
            cmdline_params.append('{}'.format(parameters_data.dict.temperature))

        if 'md_commensurate' in parameters_data.get_dict():
            if parameters_data.dict.md_commensurate:
                cmdline_params.append('--MD_commensurate')

        cmdline_params.append('> OUTPUT')

        # =================== prepare the python input files =====================

        structure_md = get_supercell(structure, supercell_shape)
        potential_object = LammpsPotential(potential_data, structure_md, potential_filename=self._INPUT_POTENTIAL)

        structure_txt = generate_LAMMPS_structure(structure_md)
        input_txt = generate_LAMMPS_input(parameters_data,
                                          potential_object,
                                          structure_file=self._INPUT_STRUCTURE,
                                          trajectory_file=self._OUTPUT_TRAJECTORY_FILE_NAME,
                                          command=' '.join(cmdline_params))

        potential_txt = potential_object.get_potential_file()

        # =========================== dump to file =============================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)
        with open(structure_filename, 'w') as infile:
            infile.write(structure_txt)

        potential_filename = tempfolder.get_abs_path(self._INPUT_POTENTIAL)
        with open(potential_filename, 'w') as infile:
            infile.write(potential_txt)

        # =+=========================  Dynaphopy =+==============================

        cell_txt = structure_to_poscar(structure)
        input_txt = parameters_to_input_file(parameters_data_dynaphopy)
        force_constants_txt = get_FORCE_CONSTANTS_txt(force_constants)

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME_DYNA)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        cell_filename = tempfolder.get_abs_path(self._INPUT_CELL)
        with open(cell_filename, 'w') as infile:
            infile.write(cell_txt)

        force_constants_filename = tempfolder.get_abs_path(self._INPUT_FORCE_CONSTANTS)
        with open(force_constants_filename, 'w') as infile:
            infile.write(force_constants_txt)

        # ============================ calcinfo ================================

        local_copy_list = []
        remote_copy_list = []
        # additional_retrieve_list = settings_dict.pop("ADDITIONAL_RETRIEVE_LIST",[])

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        # Empty command line by default
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        # Retrieve files
        calcinfo.retrieve_list = [self._OUTPUT_FORCE_CONSTANTS,
                                  self._OUTPUT_FILE_NAME,
                                  self._OUTPUT_QUASIPARTICLES]

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = ['-in', self._INPUT_FILE_NAME]

        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = True
        calcinfo.codes_info = [codeinfo]
        return calcinfo
