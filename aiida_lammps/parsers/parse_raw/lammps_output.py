"""Set of functions to parse the unformatted output file from LAMMPS"""


def parse_output_file(filename: str = None, file_contents: str = None) -> dict:
    """Parse the output file from lammps to discover if there are any errors or warnings

    :param filename: name of the lammps_output file, defaults to None
    :type filename: str, optional
    :param file_contents: contents of the lammps_output file, defaults to None
    :type file_contents: str, optional
    :return: dictionary with any error or warning found
    :rtype: dict
    """
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

    output = {"warnings": [], "errors": []}
    for index, line in enumerate(data):
        line = line.strip()

        if "ERROR" in line:
            _line = {"message": line}
            if "Last command" in data[index + 1].strip():
                _line["command"] = data[index + 1].strip()
            output["errors"].append(_line)
        if "WARNING" in line:
            output["warnings"].append(line)

    return output
