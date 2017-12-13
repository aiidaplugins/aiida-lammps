from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.trajectory import TrajectoryData
from aiida.orm.data.array import ArrayData
from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty

import numpy as np


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


def get_trajectory_txt(trajectory):

    cell = trajectory.get_cells()[0]

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

    xlo_bound = np.min([0.0, xy, xz, xy+xz])
    xhi_bound = xhi + np.max([0.0, xy, xz, xy+xz])
    ylo_bound = np.min([0.0, yz])
    yhi_bound = yhi + np.max([0.0, yz])
    zlo_bound = 0
    zhi_bound = zhi

    ind = trajectory.get_array('steps')
    lammps_data_file = ''
    for i, position_step in enumerate(trajectory.get_positions()):
        lammps_data_file += 'ITEM: TIMESTEP\n'
        lammps_data_file += '{}\n'.format(ind[i])
        lammps_data_file += 'ITEM: NUMBER OF ATOMS\n'
        lammps_data_file += '{}\n'.format(len(position_step))
        lammps_data_file += 'ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(xlo_bound, xhi_bound, xy)
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(ylo_bound, yhi_bound, xz)
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(zlo_bound, zhi_bound, yz)
        lammps_data_file += ('ITEM: ATOMS x y z\n')
        for position in position_step:
            lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*position)
    return lammps_data_file


def structure_to_poscar(structure):

    types = [site.kind_name for site in structure.sites]
    atom_type_unique = np.unique(types, return_index=True)
    sort_index = np.argsort(atom_type_unique[1])
    elements = np.array(atom_type_unique[0])[sort_index]
    elements_count= np.diff(np.append(np.array(atom_type_unique[1])[sort_index], [len(types)]))

    poscar = '# VASP POSCAR generated using aiida workflow '
    poscar += '\n1.0\n'
    cell = structure.cell
    for row in cell:
        poscar += '{0: 22.16f} {1: 22.16f} {2: 22.16f}\n'.format(*row)
    poscar += ' '.join([str(e) for e in elements]) + '\n'
    poscar += ' '.join([str(e) for e in elements_count]) + '\n'
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


class DynaphopyCalculation(JobCalculation):
    """
    A basic plugin for calculating force constants using Phonopy.

    Requirement: the node should be able to import phonopy
    """

    def _init_internal_params(self):
        super(DynaphopyCalculation, self)._init_internal_params()

        self._INPUT_FILE_NAME = 'input_dynaphopy'
        self._INPUT_TRAJECTORY = 'trajectory'
        self._INPUT_CELL = 'POSCAR'
        self._INPUT_FORCE_CONSTANTS = 'FORCE_CONSTANTS'

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
               'docstring': ("Use a node that specifies the dynaphopy input "
                             "for the namelists"),
               },
            "trajectory": {
               'valid_types': TrajectoryData,
               'additional_parameter': None,
               'linkname': 'trajectory',
               'docstring': ("Use a node that specifies the trajectory data "
                             "for the namelists"),
               },
            "force_constants": {
               'valid_types': ArrayData,
               'additional_parameter': None,
               'linkname': 'force_constants',
               'docstring': ("Use a node that specifies the force_constants "
                             "for the namelists"),
               },
            "structure": {
               'valid_types': StructureData,
               'additional_parameter': None,
               'linkname': 'structure',
               'docstring': "Use a node for the structure",
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
            pass
            #raise InputValidationError("No parameters specified for this "
            #                           "calculation")
        if not isinstance(parameters_data, ParameterData):
            raise InputValidationError("parameters is not of type "
                                       "ParameterData")

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError("no structure is specified for this calculation")

        try:
            trajectory = inputdict.pop(self.get_linkname('trajectory'))
        except KeyError:
            raise InputValidationError("trajectory is specified for this calculation")

        try:
            force_constants = inputdict.pop(self.get_linkname('force_constants'))
        except KeyError:
            raise InputValidationError("no force_constants is specified for this calculation")

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("no code is specified for this calculation")


        time_step = trajectory.get_times()[1]-trajectory.get_times()[0]

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        # =================== prepare the python input files =====================

        cell_txt = structure_to_poscar(structure)
        input_txt = parameters_to_input_file(parameters_data)
        force_constants_txt = get_FORCE_CONSTANTS_txt(force_constants)
        trajectory_txt = get_trajectory_txt(trajectory)

        # =========================== dump to file =============================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        cell_filename = tempfolder.get_abs_path(self._INPUT_CELL)
        with open(cell_filename, 'w') as infile:
            infile.write(cell_txt)

        force_constants_filename = tempfolder.get_abs_path(self._INPUT_FORCE_CONSTANTS)
        with open(force_constants_filename, 'w') as infile:
            infile.write(force_constants_txt)

        trajectory_filename = tempfolder.get_abs_path(self._INPUT_TRAJECTORY)
        with open(trajectory_filename, 'w') as infile:
            infile.write(trajectory_txt)

        # ============================ calcinfo ================================

        local_copy_list = []
        remote_copy_list = []
    #    additional_retrieve_list = settings_dict.pop("ADDITIONAL_RETRIEVE_LIST",[])

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        # Empty command line by default
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        # Retrieve files
        calcinfo.retrieve_list = [self._OUTPUT_FILE_NAME,
                                  self._OUTPUT_FORCE_CONSTANTS,
                                  self._OUTPUT_QUASIPARTICLES]

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = [self._INPUT_FILE_NAME, self._INPUT_TRAJECTORY,
                                   '-ts', '{}'.format(time_step), '--silent',
                                   '-sfc', self._OUTPUT_FORCE_CONSTANTS, '-thm', # '--resolution 0.01',
                                   '-psm','2', '--normalize_dos', '-sdata']

        if 'temperature' in parameters_data.get_dict():
            codeinfo.cmdline_params.append('--temperature')
            codeinfo.cmdline_params.append('{}'.format(parameters_data.dict.temperature))

        if 'md_commensurate' in parameters_data.get_dict():
            if parameters_data.dict.md_commensurate:
                codeinfo.cmdline_params.append('--MD_commensurate')

        codeinfo.stdout_name = self._OUTPUT_FILE_NAME
        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = False
        calcinfo.codes_info = [codeinfo]
        return calcinfo
