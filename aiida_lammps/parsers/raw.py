"""Base parser for LAMMPS output."""
import time

from aiida import orm
from aiida.parsers.parser import Parser

from aiida_lammps.calculations.raw import LammpsRawCalculation
from aiida_lammps.parsers.parse_raw import parse_outputfile


class LammpsRawParser(Parser):
    """Base parser for LAMMPS output."""

    def parse(self, **kwargs):
        """Parse the contents of the output files stored in the ``retrieved`` output node."""
        retrieved = self.retrieved
        retrieved_filenames = retrieved.base.repository.list_object_names()
        filename_out = LammpsRawCalculation.FILENAME_OUTPUT

        if filename_out not in retrieved_filenames:
            return self.exit_codes.ERROR_OUTFILE_MISSING

        parsed_data = parse_outputfile(
            file_contents=retrieved.base.repository.get_object_content(filename_out)
        )
        if parsed_data is None:
            return self.exit_codes.ERROR_PARSING_OUTFILE

        if parsed_data["global"]["errors"]:
            # Output the data for checking what was parsed
            self.out("results", orm.Dict({"compute_variables": parsed_data["global"]}))
            for entry in parsed_data["global"]["errors"]:
                self.logger.error(f"LAMMPS emitted the error {entry}")
                return self.exit_codes.ERROR_PARSER_DETECTED_LAMMPS_RUN_ERROR.format(
                    error=entry
                )

        global_data = parsed_data["global"]
        results = {"compute_variables": global_data}

        if "total_wall_time" in global_data:
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

        self.out("results", orm.Dict(results))

        return None
