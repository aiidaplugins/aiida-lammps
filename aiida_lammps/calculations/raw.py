"""Plugin with minimal interface to run LAMMPS."""
import shutil

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob


class LammpsRawCalculation(CalcJob):
    """Plugin with minimal interface to run LAMMPS."""

    FILENAME_INPUT = "input.in"
    FILENAME_OUTPUT = "lammps.out"
    FILENAME_LOG = "lammps.log"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "script",
            valid_type=orm.SinglefileData,
            help="Complete input script to use. If specified, `structure`, `potential` and `parameters` are ignored.",
        )
        spec.input_namespace(
            "files",
            valid_type=orm.SinglefileData,
            required=False,
            help="Optional files that should be written to the working directory.",
        )
        spec.input(
            "filenames",
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            required=False,
            help="Optional namespace to specify with which filenames the files of ``files`` input should be written.",
        )
        spec.inputs["metadata"]["options"][
            "input_filename"
        ].default = cls.FILENAME_INPUT
        spec.inputs["metadata"]["options"][
            "output_filename"
        ].default = cls.FILENAME_OUTPUT
        spec.inputs["metadata"]["options"]["parser_name"].default = "lammps.raw"
        spec.inputs.validator = cls.validate_inputs

        spec.output(
            "results",
            valid_type=orm.Dict,
            required=True,
            help="The data extracted from the lammps log file",
        )
        spec.exit_code(
            351,
            "ERROR_LOG_FILE_MISSING",
            message="the file with the lammps log was not found",
            invalidates_cache=True,
        )
        spec.exit_code(
            1001,
            "ERROR_PARSING_LOGFILE",
            message="parsing the log file has failed.",
        )

    @classmethod
    def validate_inputs(cls, value, ctx):
        """Validate the top-level inputs namespace."""
        # The filename with which the file is written to the working directory is defined by the ``filenames`` input
        # namespace, falling back to the filename of the ``SinglefileData`` node if not defined.
        overrides = value["filenames"].get_dict() if "filenames" in value else {}
        filenames = [
            overrides.get(key, node.filename)
            for key, node in value.get("files", {}).items()
        ]

        if len(filenames) != len(set(filenames)):
            return (
                f"The list of filenames of the ``files`` input is not unique: {filenames}. Use the ``filenames`` input "
                "namespace to explicitly define unique filenames for each file."
            )

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission.

        :param folder: A temporary folder on the local file system.
        :returns: A :class:`aiida.common.datastructures.CalcInfo` instance.
        """
        filename_log = self.FILENAME_LOG
        filename_input = self.inputs.metadata.options.input_filename
        filename_output = self.inputs.metadata.options.output_filename
        filenames = (
            self.inputs["filenames"].get_dict() if "filenames" in self.inputs else {}
        )
        provenance_exclude_list = []

        with folder.open(filename_input, "w") as handle:
            handle.write(self.inputs.script.get_content())

        for key, node in self.inputs.get("files", {}).items():

            # The filename with which the file is written to the working directory is defined by the ``filenames`` input
            # namespace, falling back to the filename of the ``SinglefileData`` node if not defined.
            filename = filenames.get(key, node.filename)

            with folder.open(filename, "wb") as target:
                with node.open(mode="rb") as source:
                    shutil.copyfileobj(source, target)

            provenance_exclude_list.append(filename)

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = ["-in", filename_input, "-log", filename_log]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.provenance_exclude_list = provenance_exclude_list
        calcinfo.retrieve_list = [filename_output, filename_log]
        calcinfo.codes_info = [codeinfo]

        return calcinfo
