from aiida.common import CalcInfo, CodeInfo
from aiida.common.exceptions import ValidationError
from aiida.engine import CalcJob
from aiida.orm import Dict, StructureData
from aiida.plugins import DataFactory
import numpy as np

from aiida_lammps.common.generate_structure import generate_lammps_structure
from aiida_lammps.data.potential import EmpiricalPotential


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


def get_force_constants(force_constants):

    force_constants = force_constants.get_array("force_constants")

    fc_shape = force_constants.shape
    fc_txt = "%4d\n" % (fc_shape[0])
    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            fc_txt += "%4d%4d\n" % (i + 1, j + 1)
            for vec in force_constants[i][j]:
                fc_txt += ("%22.15f" * 3 + "\n") % tuple(vec)

    return fc_txt


def structure_to_poscar(structure):

    atom_type_unique = np.unique(
        [site.kind_name for site in structure.sites], return_index=True
    )[1]
    labels = np.diff(np.append(atom_type_unique, [len(structure.sites)]))

    poscar = " ".join(np.unique([site.kind_name for site in structure.sites]))
    poscar += "\n1.0\n"
    cell = structure.cell
    for row in cell:
        poscar += "{0: 22.16f} {1: 22.16f} {2: 22.16f}\n".format(*row)
    poscar += " ".join(np.unique([site.kind_name for site in structure.sites])) + "\n"
    poscar += " ".join(np.array(labels, dtype=str)) + "\n"
    poscar += "Cartesian\n"
    for site in structure.sites:
        poscar += "{0: 22.16f} {1: 22.16f} {2: 22.16f}\n".format(*site.position)

    return poscar


def parameters_to_input_file(parameters_object):

    parameters = parameters_object.get_dict()
    input_file = "STRUCTURE FILE POSCAR\nPOSCAR\n\n"
    input_file += "FORCE CONSTANTS\nFORCE_CONSTANTS\n\n"
    input_file += "PRIMITIVE MATRIX\n"
    input_file += ("{} {} {} \n").format(*np.array(parameters["primitive"])[0])
    input_file += ("{} {} {} \n").format(*np.array(parameters["primitive"])[1])
    input_file += ("{} {} {} \n").format(*np.array(parameters["primitive"])[2])
    input_file += "\n"
    input_file += "SUPERCELL MATRIX PHONOPY\n"
    input_file += ("{} {} {} \n").format(*np.array(parameters["supercell"])[0])
    input_file += ("{} {} {} \n").format(*np.array(parameters["supercell"])[1])
    input_file += ("{} {} {} \n").format(*np.array(parameters["supercell"])[2])
    input_file += "\n"

    return input_file


class BaseLammpsCalculation(CalcJob):
    """
    A basic plugin for calculating force constants using Lammps.

    Requirement: the node should be able to import phonopy
    """

    _INPUT_FILE_NAME = "input.in"
    _INPUT_STRUCTURE = "input.data"

    _DEFAULT_OUTPUT_FILE_NAME = "log.lammps"
    _DEFAULT_TRAJECTORY_FILE_NAME = "trajectory.lammpstrj"
    _DEFAULT_SYSTEM_FILE_NAME = "system_info.dump"
    _DEFAULT_RESTART_FILE_NAME = "lammps.restart"

    _cmdline_params = ("-in", _INPUT_FILE_NAME)
    _stdout_name = None

    @classmethod
    def define(cls, spec):
        super(BaseLammpsCalculation, cls).define(spec)
        spec.input("structure", valid_type=StructureData, help="the structure")
        spec.input("potential", valid_type=EmpiricalPotential, help="lammps potential")
        spec.input("parameters", valid_type=Dict, help="the parameters", required=False)
        spec.input(
            "metadata.options.cell_transform_filename",
            valid_type=str,
            default="cell_transform.npy",
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE_NAME,
        )
        spec.input(
            "metadata.options.trajectory_suffix",
            valid_type=str,
            default=cls._DEFAULT_TRAJECTORY_FILE_NAME,
        )
        spec.input(
            "metadata.options.system_suffix",
            valid_type=str,
            default=cls._DEFAULT_SYSTEM_FILE_NAME,
        )
        spec.input(
            "metadata.options.restart_filename",
            valid_type=str,
            default=cls._DEFAULT_RESTART_FILE_NAME,
        )

        spec.output(
            "results",
            valid_type=DataFactory("dict"),
            required=True,
            help="the data extracted from the main output file",
        )
        spec.default_output_node = "results"

        # Unrecoverable errors: resources like the retrieved folder or its expected contents are missing
        spec.exit_code(
            200,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="The retrieved folder data node could not be accessed.",
        )
        spec.exit_code(
            201,
            "ERROR_NO_RETRIEVED_TEMP_FOLDER",
            message="The retrieved temporary folder data node could not be accessed.",
        )
        spec.exit_code(
            202,
            "ERROR_LOG_FILE_MISSING",
            message="the main log output file was not found",
        )
        spec.exit_code(
            203,
            "ERROR_TRAJ_FILE_MISSING",
            message="the trajectory output file was not found",
        )
        spec.exit_code(
            204,
            "ERROR_STDOUT_FILE_MISSING",
            message="the stdout output file was not found",
        )
        spec.exit_code(
            205,
            "ERROR_STDERR_FILE_MISSING",
            message="the stderr output file was not found",
        )

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(
            300,
            "ERROR_LOG_PARSING",
            message=(
                "An error was flagged trying to parse the "
                "main lammps output log file"
            ),
        )
        spec.exit_code(
            310,
            "ERROR_TRAJ_PARSING",
            message=(
                "An error was flagged trying to parse the " "trajectory output file"
            ),
        )
        spec.exit_code(
            320,
            "ERROR_INFO_PARSING",
            message=(
                "An error was flagged trying to parse the " "system info output file"
            ),
        )

        # Significant errors but calculation can be used to restart
        spec.exit_code(
            400,
            "ERROR_LAMMPS_RUN",
            message="The main lammps output file flagged an error",
        )
        spec.exit_code(
            401,
            "ERROR_RUN_INCOMPLETE",
            message="The main lammps output file did not flag that the computation finished",
        )

    @staticmethod
    def validate_parameters(param_data, potential_object):
        return True

    def prepare_extra_files(self, tempfolder, potential_object):
        return True

    def get_retrieve_lists(self):
        return [], []

    @staticmethod
    def create_main_input_content(
        parameter_data,
        potential_data,
        kind_symbols,
        structure_filename,
        trajectory_filename,
        system_filename,
        restart_filename,
    ):
        raise NotImplementedError

    def prepare_for_submission(self, tempfolder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param tempfolder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.CalcInfo` instance
        """
        # assert that the potential and structure have the same kind elements
        if self.inputs.potential.allowed_element_names is not None and not set(
            k.symbol for k in self.inputs.structure.kinds
        ).issubset(self.inputs.potential.allowed_element_names):
            raise ValidationError(
                "the structure and potential are not compatible (different kind elements)"
            )

        # Setup structure
        structure_txt, struct_transform = generate_lammps_structure(
            self.inputs.structure, self.inputs.potential.atom_style
        )

        with open(
            tempfolder.get_abs_path(self.options.cell_transform_filename), "w+b"
        ) as handle:
            np.save(handle, struct_transform)

        if "parameters" in self.inputs:
            parameters = self.inputs.parameters
        else:
            parameters = Dict()

        # Setup input parameters
        input_txt = self.create_main_input_content(
            parameter_data=parameters,
            potential_data=self.inputs.potential,
            kind_symbols=[kind.symbol for kind in self.inputs.structure.kinds],
            structure_filename=self._INPUT_STRUCTURE,
            trajectory_filename=self.options.trajectory_suffix,
            system_filename=self.options.system_suffix,
            restart_filename=self.options.restart_filename,
        )

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)

        with open(input_filename, "w") as infile:
            infile.write(input_txt)

        self.validate_parameters(parameters, self.inputs.potential)
        retrieve_list, retrieve_temporary_list = self.get_retrieve_lists()
        retrieve_list.extend(
            [self.options.output_filename, self.options.cell_transform_filename]
        )

        # prepare extra files if needed
        self.prepare_extra_files(tempfolder, self.inputs.potential)

        # =========================== dump to file =============================

        structure_filename = tempfolder.get_abs_path(self._INPUT_STRUCTURE)
        with open(structure_filename, "w") as infile:
            infile.write(structure_txt)

        for name, content in self.inputs.potential.get_external_files().items():
            fpath = tempfolder.get_abs_path(name)
            with open(fpath, "w") as infile:
                infile.write(content)

        # ============================ calcinfo ================================

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = list(self._cmdline_params)
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.withmpi = self.metadata.options.withmpi
        codeinfo.stdout_name = self._stdout_name

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = retrieve_list
        calcinfo.retrieve_temporary_list = retrieve_temporary_list
        calcinfo.codes_info = [codeinfo]

        return calcinfo
