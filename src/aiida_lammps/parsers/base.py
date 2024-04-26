"""
Base parser for LAMMPS calculations.

It takes care of parsing the lammps.out file, the trajectory file and the
yaml file with the final value of the variables printed in the ``thermo_style``.
"""
import glob
import os
import time
from typing import Any, Union

from aiida import orm
from aiida.common import exceptions
from aiida.parsers.parser import Parser
import numpy as np

from aiida_lammps.data.trajectory import LammpsTrajectory
from aiida_lammps.parsers.parse_raw import parse_final_data, parse_outputfile


class LammpsBaseParser(Parser):
    """
    Base parser for LAMMPS calculations.

    It takes care of parsing the lammps.out file, the trajectory file and the
    yaml file with the final value of the variables printed in the
    ``thermo_style``.
    """

    def __init__(self, node):
        """Initialize the parser"""
        # pylint: disable=useless-super-delegation
        super().__init__(node)

    def parse(self, **kwargs):
        """
        Parse the files produced by lammps.

        It takes care of parsing the lammps.out file, the trajectory file and the
        yaml file with the final value of the variables printed in the
        ``thermo_style``.
        """
        # pylint: disable=too-many-return-statements, too-many-locals

        # Get the input parameters to see if one needs to parse the restart file
        if "parameters" in self.node.inputs:
            parameters = self.node.inputs.parameters.get_dict()
        else:
            parameters = {}
        if "settings" in self.node.inputs:
            settings = self.node.inputs.settings.get_dict()
        else:
            settings = {}

        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        list_of_files = out_folder.base.repository.list_object_names()
        # Check the output file
        outputfile_filename = self.node.get_option("output_filename")
        if outputfile_filename not in list_of_files:
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING
        parsed_data = parse_outputfile(
            file_contents=self.node.outputs.retrieved.base.repository.get_object_content(
                outputfile_filename
            )
        )

        if parsed_data["global"]["errors"]:
            # Output the data for checking what was parsed
            self.out("results", orm.Dict({"compute_variables": parsed_data["global"]}))
            for entry in parsed_data["global"]["errors"]:
                self.logger.error(f"LAMMPS emitted the error {entry}")
                return self.exit_codes.ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR.format(
                    error=entry
                )

        global_data = parsed_data["global"]
        arrays = parsed_data["time_dependent"]
        results = {"compute_variables": global_data}

        _end_file_found = "total_wall_time" in global_data

        if _end_file_found:
            try:
                parsed_time = time.strptime(global_data["total_wall_time"], "%H:%M:%S")
            except ValueError:
                pass
            else:
                total_wall_time_seconds = (
                    parsed_time.tm_hour * 3600
                    + parsed_time.tm_min * 60
                    + parsed_time.tm_sec
                )
                global_data["total_wall_time_seconds"] = total_wall_time_seconds

        if parsed_data["global"]["warnings"]:
            for entry in parsed_data["global"]["warnings"]:
                self.logger.warning(f"LAMMPS emitted the warning {entry}")

        # check final variable file
        final_variables = None
        variables_filename = self.node.get_option("variables_filename")
        if variables_filename not in list_of_files and _end_file_found:
            return self.exit_codes.ERROR_FINAL_VARIABLE_FILE_MISSING
        final_variables = parse_final_data(
            file_contents=self.node.outputs.retrieved.base.repository.get_object_content(
                variables_filename
            )
        )
        if final_variables is None:
            return self.exit_codes.ERROR_PARSING_FINAL_VARIABLES

        results.update(**final_variables)

        # Check if there is a restartfile present
        if "restart" in parameters:
            _restartfile_name = self.parse_restartfile(
                parameters=parameters,
                list_of_files=list_of_files,
                temp_folder=kwargs.get("retrieved_temporary_folder", None),
            )

            if _restartfile_name:
                results["compute_variables"]["restartfile_name"] = _restartfile_name
            if (
                not _restartfile_name
                and settings.get("store_restart", False)
                and _end_file_found
            ):
                return self.exit_codes.ERROR_RESTART_FILE_MISSING

        # Expose the results from the lammps.out outputs
        self.out("results", orm.Dict(results))

        # Get the time-dependent outputs exposed as an ArrayData
        time_dependent_computes = orm.ArrayData()

        for key, value in arrays.items():
            _data = [val if val is not None else np.nan for val in value]
            time_dependent_computes.set_array(key, np.array(_data))

        self.out("time_dependent_computes", time_dependent_computes)

        # check trajectory file
        trajectory_filename = self.node.get_option("trajectory_filename")
        if trajectory_filename not in list_of_files and _end_file_found:
            return self.exit_codes.ERROR_TRAJECTORY_FILE_MISSING
        with self.node.outputs.retrieved.base.repository.open(
            trajectory_filename
        ) as handle:
            lammps_trajectory = LammpsTrajectory(handle)

        self.out("trajectories", lammps_trajectory)
        self.out("structure", lammps_trajectory.get_step_structure(-1))

        # check stdout
        if self.node.get_option("scheduler_stdout") not in list_of_files:
            return self.exit_codes.ERROR_STDOUT_FILE_MISSING

        # check stderr
        if self.node.get_option("scheduler_stderr") not in list_of_files:
            return self.exit_codes.ERROR_STDERR_FILE_MISSING

        if not _end_file_found:
            return self.exit_codes.ERROR_OUT_OF_WALLTIME

        # Check for the convergence of the calculation
        if (
            "parameters" in self.node.inputs
            and "minimize" in self.node.inputs.parameters.get_dict()
        ):
            self.check_convergence(global_data=global_data)

        return None

    def parse_restartfile(
        self,
        parameters: dict[str, Any],
        list_of_files: list[str],
        temp_folder: Union[os.PathLike, str, None],
    ) -> str:
        """
        Parse the restartfile generated by ``LAMMPS`` and store it as a node in the database.

        ``LAMMPS`` can produce several restartfiles, where some are written
        during the simulation at regular intervals, and another that is
        stored at the end of the simulation.

        This function tries to find which of those files are written by ``LAMMPS``
        and then store them in the database as ``orm.SinglefileData``.

        :param parameters: set of variables for the lammps script generation
        :type parameters: dict
        :param list_of_files: list of files retrieved
        :type list_of_files: list
        :param temp_folder: name of the temporary folder where the temporary retrieved are
        :type temp_folder: Union[os.PathLike, str, None]

        :return: Name of the found restartfile
        :rtype: str
        """
        input_restart_filename = self.node.get_option("restart_filename")

        restart_found = False

        restart_filename = ""

        if (
            parameters.get("restart", {}).get("print_final", False)
            and input_restart_filename in list_of_files
        ):
            with self.node.outputs.retrieved.base.repository.open(
                input_restart_filename,
                mode="rb",
            ) as handle:
                restart_file = orm.SinglefileData(handle)
            self.out("restartfile", restart_file)
            restart_found = True
            restart_filename = input_restart_filename

        if (
            parameters.get("restart", {}).get("print_intermediate", False)
            and not restart_found
            and temp_folder
        ):
            restartfiles = glob.glob(f"{temp_folder}/{input_restart_filename}*")

            if restartfiles:
                _files = []
                for entry in restartfiles:
                    try:
                        _files.append(
                            int(
                                entry.replace(
                                    f"{temp_folder}/{input_restart_filename}", ""
                                ).replace(".", "")
                            )
                        )
                    except ValueError:
                        _files.append(0)

                latest_file = os.path.basename(restartfiles[np.array(_files).argmax()])
                restart_filename = latest_file
                with open(os.path.join(temp_folder, latest_file), mode="rb") as handle:
                    restart_file = orm.SinglefileData(handle)
                self.out("restartfile", restart_file)
        return restart_filename

    def check_convergence(self, global_data: dict[str, Any]):
        """Check for the convergence of the calculation in the case of a minimization run"""
        _etol = global_data.get("minimization", {}).get(
            "energy_relative_difference", None
        )
        _ftol = global_data.get("minimization", {}).get("force_two_norm", None)
        _stop_criterion = global_data.get("minimization", {}).get(
            "stop_criterion", None
        )

        _input_etol = (
            self.node.inputs.parameters.get_dict()
            .get("minimize", {})
            .get("energy_tolerance", None)
        )
        _input_ftol = (
            self.node.inputs.parameters.get_dict()
            .get("minimize", {})
            .get("force_tolerance", None)
        )

        if _stop_criterion:
            if _stop_criterion.lower() == "force tolerance" and _ftol > _input_ftol:
                raise self.exit_codes.ERROR_FORCE_NOT_CONVERGED
            if _stop_criterion.lower() == "energy tolerance" and _etol > _input_etol:
                raise self.exit_codes.ERROR_ENERGY_NOT_CONVERGED
