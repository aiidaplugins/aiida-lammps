"""Base parser for LAMMPS log output."""
import time

from aiida import orm
from aiida.parsers.parser import Parser

from aiida_lammps.calculations.raw import LammpsRawCalculation
from aiida_lammps.parsers.parse_raw import parse_logfile


class LammpsRawParser(Parser):
    """Base parser for LAMMPS log output."""

    def parse(self, **kwargs):
        """Parse the contents of the output files stored in the ``retrieved`` output node."""
        retrieved = self.retrieved
        retrieved_filenames = retrieved.base.repository.list_object_names()
        filename_log = LammpsRawCalculation.FILENAME_LOG

        if filename_log not in retrieved_filenames:
            return self.exit_codes.ERROR_LOG_FILE_MISSING

        parsed_data = parse_logfile(
            file_contents=retrieved.base.repository.get_object_content(filename_log)
        )
        if parsed_data is None:
            return self.exit_codes.ERROR_PARSING_LOGFILE

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
