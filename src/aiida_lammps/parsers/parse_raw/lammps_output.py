"""Set of functions to parse the unformatted output file from LAMMPS"""
# pylint: disable=fixme
import ast
import re
from typing import Optional, Union

import numpy as np


def parse_outputfile(
    filename: Optional[str] = None, file_contents: Optional[str] = None
) -> Union[dict, dict]:
    """
    Parse the lammps output file file, this is the redirected screen output.

    This will gather the time dependent data stored in the output file and
    stores it as a dictionary. It will also gather single quantities and stores
    them into a different dictionary.

    :param filename: name of the lammps output file, defaults to None
    :type filename: str, optional
    :param file_contents: contents of the lammps output file, defaults to None
    :type file_contents: str, optional
    :return: dictionary with the time dependent data, dictionary with the global data
    :rtype: dict
    """
    # pylint: disable=too-many-branches, too-many-locals

    if filename is None and file_contents is None:
        return None

    if filename is not None:
        try:
            with open(filename) as handler:
                data = handler.read()
                data = data.split("\n")
        except OSError:
            return None

    if file_contents is not None:
        data = file_contents.split("\n")

    header_line_position = -1
    header_line = ""
    _data = []
    end_found = False
    parsed_data = {}
    global_parsed_data = {"warnings": [], "errors": []}

    perf_regex = re.compile(r".*Performance\:.*\,\s+([0-9\.]*)\stimesteps\/s.*")
    performance_match = perf_regex.search(file_contents or "")
    if performance_match:
        global_parsed_data["steps_per_second"] = float(performance_match.group(1))

    for index, line in enumerate(data):
        line = line.strip()

        if "ERROR" in line:
            _line = {"message": line}
            if "Last command" in data[index + 1].strip():
                _line["command"] = data[index + 1].strip()
            global_parsed_data["errors"].append(_line)
        if "WARNING" in line:
            global_parsed_data["warnings"].append(line)
        if "binsize" in line:
            global_parsed_data["binsize"] = ast.literal_eval(
                line.split()[2].replace(",", "")
            )
            global_parsed_data["bins"] = [
                ast.literal_eval(entry) for entry in line.split()[5:]
            ]
        if "ghost atom cutoff" in line:
            global_parsed_data["ghost_atom_cutoff"] = ast.literal_eval(line.split()[-1])
        if "master list distance cutoff" in line:
            global_parsed_data["master_list_distance_cutoff"] = ast.literal_eval(
                line.split()[-1]
            )
        if "max neighbors/atom" in line:
            global_parsed_data["max_neighbors_atom"] = ast.literal_eval(
                line.split()[2].replace(",", "")
            )
        if "Unit style" in line:
            global_parsed_data["units_style"] = line.split(":")[-1].strip()
        if line.startswith("units"):
            global_parsed_data["units_style"] = line.split("units")[-1].strip()

        if "Total wall time:" in line:
            global_parsed_data["total_wall_time"] = line.split()[-1]
        if "bin:" in line:
            global_parsed_data["bin"] = line.split()[-1]

        if "Minimization stats" in line:
            global_parsed_data["minimization"] = {}
        if "Stopping criterion" in line:
            global_parsed_data["minimization"]["stop_criterion"] = (
                line.strip().split("=")[-1].strip()
            )
        if "Energy initial, next-to-last, final" in line:
            global_parsed_data["minimization"][
                "energy_relative_difference"
            ] = _calculate_energy_tolerance(data[index + 1])
        if "Force two-norm initial, final" in line:
            global_parsed_data["minimization"]["force_two_norm"] = float(
                line.strip().split("=")[-1].split()[-1].strip()
            )

        if line.startswith("Step"):
            header_line_position = index
            header_line = [
                re.sub("[^a-zA-Z0-9_]", "__", entry) for entry in line.split()
            ]
        if (
            header_line_position > 0
            and index != header_line_position
            and not end_found
            and not line.split()[0].replace(".", "", 1).isdigit()
        ):
            end_found = True
        if header_line_position > 0 and index != header_line_position and not end_found:
            _data.append([ast.literal_eval(entry) for entry in line.split()])
    _data = np.asarray(_data)
    for index, entry in enumerate(header_line):
        parsed_data[entry] = _data[:, index].tolist()

    return {"time_dependent": parsed_data, "global": global_parsed_data}


def _calculate_energy_tolerance(line: str) -> float:
    """Determine the energy tolerance found in the minimization step"""
    energy = [float(entry) for entry in line.split()]
    return (energy[-1] - energy[-2]) / energy[-1]
