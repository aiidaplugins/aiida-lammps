from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty

from potentials import LammpsPotential


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


def generate_LAMMPS_input(potential_obj,
                          parameters_data,
                          structure_file='data.gan',
                          optimize_path_file='path.lammpstrj'):

    names_str = ' '.join(potential_obj._names)



    parameters = parameters_data.get_dict()

    lammps_input_file =  'units           metal\n'
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      atomic\n'
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'fix             int all box/relax {} {} vmax {}\n'.format(parameters['relaxation'],
                                                                                    parameters['pressure'] * 1000,  # pressure kb -> bar
                                                                                    parameters['vmax'])

    lammps_input_file += 'compute         stpa all stress/atom NULL\n'
                                                              #  xx,       yy,        zz,       xy,       xz,       yz
    lammps_input_file += 'compute         stgb all reduce sum c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]\n'
    lammps_input_file += 'variable        pr equal -(c_stgb[1]+c_stgb[2]+c_stgb[3])/(3*vol)\n'
    lammps_input_file += 'thermo_style    custom step temp press v_pr etotal c_stgb[1] c_stgb[2] c_stgb[3] c_stgb[4] c_stgb[5] c_stgb[6]\n'

    lammps_input_file += 'dump            aiida all custom 1 {0} element x y z  fx fy fz\n'.format(optimize_path_file)
    lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"\n'
    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)
    lammps_input_file += 'min_style       cg\n'
    lammps_input_file += 'minimize        {} {} {} {}\n'.format(parameters['energy_tolerance'],
                                                                           parameters['force_tolerance'],
                                                                           parameters['max_iterations'],
                                                                           parameters['max_evaluations'])
  #  lammps_input_file += 'print           "$(xlo - xhi) $(xy) $(xz)"\n'
  #  lammps_input_file += 'print           "0.000 $(yhi - ylo) $(yz)"\n'
  #  lammps_input_file += 'print           "0.000 0.000   $(zhi-zlo)"\n'
    lammps_input_file += 'print           "$(xlo) $(xhi) $(xy)"\n'
    lammps_input_file += 'print           "$(ylo) $(yhi) $(xz)"\n'
    lammps_input_file += 'print           "$(zlo) $(zhi) $(yz)"\n'

    return lammps_input_file


class OptimizeCalculation(JobCalculation):
    """
    A basic plugin for calculating force constants using Lammps.

    Requirement: the node should be able to import phonopy
    """

    def _init_internal_params(self):
        super(OptimizeCalculation, self)._init_internal_params()

        self._INPUT_FILE_NAME = 'input.in'
        self._INPUT_POTENTIAL = 'potential.pot'
        self._INPUT_STRUCTURE = 'input.data'

        self._OUTPUT_TRAJECTORY_FILE_NAME = 'path.lammpstrj'

        self._OUTPUT_FILE_NAME = 'log.lammps'
        self._default_parser = 'lammps.optimize'

    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the namelists class.
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            "potential": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'potential',
               'docstring': ("Use a node that specifies the lammps potential "
                             "for the namelists"),
               },
            "parameters": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'parameters',
               'docstring': ("Use a node that specifies the lammps parameters "
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
            potential_data = inputdict.pop(self.get_linkname('potential'))
        except KeyError:
            raise InputValidationError("No potential specified for this "
                                       "calculation")

        if not isinstance(potential_data, ParameterData):
            raise InputValidationError("potential is not of type "
                                       "ParameterData")

        try:
            parameters_data = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError("No parameters specified for this "
                                       "calculation")

        if not isinstance(parameters_data, ParameterData):
            raise InputValidationError("parameters is not of type "
                                       "ParameterData")


        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError("no structure is specified for this calculation")

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("no code is specified for this calculation")

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        # =================== prepare the python input files =====================

        potential_object = LammpsPotential(potential_data, structure, potential_filename=self._INPUT_POTENTIAL)


        structure_txt = generate_LAMMPS_structure(structure)
        input_txt = generate_LAMMPS_input(potential_object,
                                          parameters_data,
                                          structure_file=self._INPUT_STRUCTURE,
                                          optimize_path_file=self._OUTPUT_TRAJECTORY_FILE_NAME)

        potential_txt = potential_object.get_potential_file()

        # =========================== dump to file =============================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)
        with open(input_filename, 'w') as infile:
            infile.write(input_txt)

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)
        with open(structure_filename, 'w') as infile:
            infile.write(structure_txt)

        if potential_txt is not None:
            potential_filename = tempfolder.get_abs_path(self._INPUT_POTENTIAL)
            with open(potential_filename, 'w') as infile:
                infile.write(potential_txt)

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
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self._OUTPUT_TRAJECTORY_FILE_NAME)
        calcinfo.retrieve_list.append(self._OUTPUT_FILE_NAME)

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = ['-in', self._INPUT_FILE_NAME]
        codeinfo.code_uuid = code.uuid
        codeinfo.withmpi = False
        calcinfo.codes_info = [codeinfo]
        return calcinfo


#$MPI -n $NSLOTS $LAMMPS -sf gpu -pk gpu 2 neigh no -in in.md_data
