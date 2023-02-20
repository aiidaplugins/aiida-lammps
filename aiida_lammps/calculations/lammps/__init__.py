"""Base LAMMPS calculation for AiiDA."""
# pylint: disable=duplicate-code, duplicate-code
import itertools

from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from aiida.common.exceptions import ValidationError
from aiida.engine import CalcJob
import numpy as np

from aiida_lammps.common.generate_structure import generate_lammps_structure
from aiida_lammps.data.potential import EmpiricalPotential


def get_supercell(
    structure: orm.StructureData,
    supercell_shape: orm.Dict,
) -> orm.StructureData:
    """Generate a supercell from a given StructureData

    :param structure: original structure that will be used to generate the supercell
    :type structure: orm.StructureData
    :param supercell_shape: dictionary with the supercell information
    :type supercell_shape: orm.Dict
    :return: generated supercell
    :rtype: orm.StructureData
    """
    symbols = np.array([site.kind_name for site in structure.sites])
    positions = np.array([site.position for site in structure.sites])
    cell = np.array(structure.cell)
    supercell_shape = np.array(supercell_shape.dict.shape)

    supercell_array = np.dot(cell, np.diag(supercell_shape))

    supercell = orm.StructureData(cell=supercell_array)
    for k in range(positions.shape[0]):
        for entry in itertools.product(*[range(i) for i in supercell_shape[::-1]]):
            position = positions[k, :] + np.dot(np.array(entry[::-1]), cell)
            symbol = symbols[k]
            supercell.append_atom(position=position, symbols=symbol)

    return supercell


def get_force_constants(force_constants: orm.ArrayData) -> str:
    """Get the force constants in text format

    :param force_constants: Array with the information needed for the force constants
    :type force_constants: orm.ArrayData
    :return: force constants in text
    :rtype: str
    """
    force_constants = force_constants.get_array("force_constants")

    fc_shape = force_constants.shape
    fc_txt = "%4d\n" % (fc_shape[0])
    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            fc_txt += "%4d%4d\n" % (i + 1, j + 1)
            for vec in force_constants[i][j]:
                fc_txt += ("%22.15f" * 3 + "\n") % tuple(vec)

    return fc_txt


def structure_to_poscar(structure: orm.StructureData) -> str:
    """Write the structure into a POSCAR

    :param structure: structure used for the simulation
    :type structure: orm.StructureData
    :return: POSCAR format for the structure
    :rtype: str
    """
    atom_type_unique = np.unique(
        [site.kind_name for site in structure.sites],
        return_index=True,
    )[1]
    labels = np.diff(np.append(atom_type_unique, [len(structure.sites)]))

    poscar = " ".join(np.unique([site.kind_name for site in structure.sites]))
    poscar += "\n1.0\n"
    cell = structure.cell
    for row in cell:
        poscar += f"{row[0]: 22.16f} {row[1]: 22.16f} {row[2]: 22.16f}\n"
    poscar += " ".join(np.unique([site.kind_name for site in structure.sites])) + "\n"
    poscar += " ".join(np.array(labels, dtype=str)) + "\n"
    poscar += "Cartesian\n"
    for site in structure.sites:
        poscar += f"{site.position[0]: 22.16f} "
        poscar += f"{site.position[1]: 22.16f} "
        poscar += f"{site.position[2]: 22.16f}\n"

    return poscar


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
        super().define(spec)
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="the structure",
        )
        spec.input(
            "potential",
            valid_type=EmpiricalPotential,
            help="lammps potential",
        )
        spec.input(
            "parameters",
            valid_type=orm.Dict,
            help="the parameters",
            required=False,
        )
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
            valid_type=orm.Dict,
            required=True,
            help="the data extracted from the main output file",
        )
        spec.default_output_node = "results"

        # Unrecoverable errors: resources like the retrieved folder or
        # its expected contents are missing
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

        # Unrecoverable errors: required retrieved files could not be read,
        # parsed or are otherwise incomplete
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
    def validate_parameters(param_data, potential_object) -> bool:
        """Validate the input parameters against a schema.

        :param param_data: input parameters to be checked
        :type param_data: [type]
        :param potential_object: LAMMPS potential object
        :type potential_object: [type]
        :return: whether or not the input parameters are valid
        :rtype: bool
        """
        # pylint: disable=unused-argument
        return True

    def prepare_extra_files(self, tempfolder, potential_object) -> bool:
        """Check if extra files need to be prepared for the calculation

        :param tempfolder: temporary folder for the calculation files
        :type tempfolder: [type]
        :param potential_object: LAMMPS potential
        :type potential_object: [type]
        :return: whether or not extra files need to be prepared for the calculation
        :rtype: bool
        """
        # pylint: disable=no-self-use, unused-argument
        return True

    def get_retrieve_lists(self) -> list:
        """Get the list of files to be retrieved.

        :return: list of files that should be retrieved
        :rtype: list
        """
        # pylint: disable=no-self-use
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
        """Generate the main input file for the lammps simulation.

        :param parameter_data: Data needed to describe the simulation
        :type parameter_data: [type]
        :param potential_data: Data that describes the potential
        :type potential_data: [type]
        :param kind_symbols: Symbols of the atoms present in the simulation box
        :type kind_symbols: [type]
        :param structure_filename: Name of the structure file
        :type structure_filename: [type]
        :param trajectory_filename: Name of the trajectory file
        :type trajectory_filename: [type]
        :param system_filename: Name of the system file
        :type system_filename: [type]
        :param restart_filename: Name of the restart file
        :type restart_filename: [type]
        :raises NotImplementedError: [description]
        """
        # pylint: disable=no-self-use, too-many-arguments, unused-argument, duplicate-code, arguments-differ
        raise NotImplementedError

    def prepare_for_submission(self, tempfolder):  # pylint: disable=arguments-differ
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param tempfolder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.CalcInfo` instance
        """
        # pylint: disable=too-many-locals
        # assert that the potential and structure have the same kind elements
        if self.inputs.potential.allowed_element_names is not None and not {
            k.symbol for k in self.inputs.structure.kinds
        }.issubset(self.inputs.potential.allowed_element_names):
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
            parameters = orm.Dict()

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
