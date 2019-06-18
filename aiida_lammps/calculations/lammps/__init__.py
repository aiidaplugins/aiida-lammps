from aiida.engine import CalcJob
from aiida.common import CalcInfo, CodeInfo
from aiida.orm import StructureData, Dict
from aiida.plugins import DataFactory
from aiida_lammps.common.utils import convert_date_string
from aiida_lammps.data.potential import EmpiricalPotential
import six

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


def generate_LAMMPS_structure(structure, atom_style):
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
        if atom_style == 'charge':
            # TODO variable initial charge
            lammps_data_file += '{0} {1} 0.0 {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(i+1, atom_index[i]+1, row[0],row[1],row[2])
        else:
            lammps_data_file += '{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(i+1, atom_index[i]+1, row[0],row[1],row[2])

    return lammps_data_file


def generate_LAMMPS_potential(pair_style):

    potential_file = '# Potential file generated by aiida plugin (please check citation in the orignal file)\n'
    for key, value in pair_style.dict.data.iteritems():
        potential_file += '{}    {}\n'.format(key, value)

    return potential_file


class BaseLammpsCalculation(CalcJob):
    """
    A basic plugin for calculating force constants using Lammps.

    Requirement: the node should be able to import phonopy
    """

    _INPUT_FILE_NAME = 'input.in'
    _INPUT_STRUCTURE = 'input.data'

    _OUTPUT_FILE_NAME = 'log.lammps'

    _retrieve_list = ['log.lammps']
    _retrieve_temporary_list = []
    _cmdline_params = ['-in', _INPUT_FILE_NAME]
    _stdout_name = None

    @classmethod
    def define(cls, spec):
        super(BaseLammpsCalculation, cls).define(spec)
        spec.input('structure', valid_type=StructureData, help='the structure')
        spec.input('potential', valid_type=EmpiricalPotential,
                   help='lammps potential')
        spec.input('parameters', valid_type=Dict,
                   help='the parameters', required=False)
        spec.input('metadata.options.output_filename',
                   valid_type=six.string_types, default=cls._OUTPUT_FILE_NAME)

        spec.default_output_node = 'results'
        spec.output('results',
                    valid_type=DataFactory('dict'),
                    required=True,
                    help='the data extracted from the main output file')

        # TODO review aiidateam/aiida_core#2997, when closed, for exit code formalization

        # Unrecoverable errors: resources like the retrieved folder or its expected contents are missing
        spec.exit_code(
            200, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(
            201, 'ERROR_NO_RETRIEVED_TEMP_FOLDER',
            message='The retrieved temporary folder data node could not be accessed.')
        spec.exit_code(
            210, 'ERROR_OUTPUT_FILE_MISSING',
            message='the main output file was not found')
        spec.exit_code(
            220, 'ERROR_TRAJ_FILE_MISSING',
            message='the trajectory output file was not found')

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(
            300, 'ERROR_OUTPUT_PARSING',
            message=('An error was flagged trying to parse the '
                     'main lammps output file'))
        spec.exit_code(
            310, 'ERROR_TRAJ_PARSING',
            message=('An error was flagged trying to parse the '
                     'trajectory output file'))

        # Significant errors but calculation can be used to restart
        spec.exit_code(
            400, 'ERROR_LAMMPS_RUN',
            message='The main lammps output file flagged an error')

    def validate_parameters(self, param_data, potential_object):
        return True

    def prepare_extra_files(self, tempfolder, potential_object):
        return True

    def prepare_for_submission(self, tempfolder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param tempfolder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.CalcInfo` instance
        """

        # Setup potential
        potential_txt = self.inputs.potential.get_potential_file()

        # Setup structure
        structure_txt = generate_LAMMPS_structure(self.inputs.structure,
                                                  self.inputs.potential.atom_style)

        if "parameters" in self.inputs:
            parameters = self.inputs.parameters
        else:
            parameters = Dict()

        # Check lammps version date in parameters
        lammps_date = convert_date_string(
            parameters.get_dict().get("lammps_version", '11 Aug 2017'))

        # Setup input parameters
        input_txt = self._generate_input_function(parameters,
                                                  self.inputs.potential,
                                                  self._INPUT_STRUCTURE,
                                                  self._OUTPUT_TRAJECTORY_FILE_NAME,
                                                  version_date=lammps_date)

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)

        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        self.validate_parameters(parameters, self.inputs.potential)

        # prepare extra files if needed
        self.prepare_extra_files(tempfolder, self.inputs.potential)

        # =========================== dump to file =============================

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)
        with open(structure_filename, 'w') as infile:
            infile.write(structure_txt)

        if potential_txt is not None:
            potential_filename = tempfolder.get_abs_path(
                self.inputs.potential.potential_filename)
            with open(potential_filename, 'w') as infile:
                infile.write(potential_txt)

        # ============================ calcinfo ================================

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = self._cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.withmpi = False  # Set lammps openmpi environment properly
        codeinfo.stdout_name = self._stdout_name

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = self._retrieve_list
        calcinfo.retrieve_temporary_list = self._retrieve_temporary_list
        calcinfo.codes_info = [codeinfo]

        return calcinfo
