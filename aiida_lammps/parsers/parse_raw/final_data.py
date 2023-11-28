"""Set of functions to parse the files containing the final variables printed by LAMMPS"""

from typing import Optional

import yaml


def parse_final_data(
    filename: Optional[str] = None, file_contents: Optional[str] = None
) -> dict:
    """
    Read the yaml file with the global final data.

    The final iteration for each of computed variables is sotred into a yaml
    file which is then read and stored as a dictionary.

    :param filename: name of the yaml file where the variables are stored,
        defaults to None
    :type filename: str, optional
    :param file_contents: contents of the yaml file where the variables are stored,
        defaults to None
    :type file_contents: str, optional
    :return: dictionary with the final compute variables
    :rtype: dict
    """

    if filename is None and file_contents is None:
        return None
    if filename is not None:
        try:
            with open(filename) as handle:
                data = yaml.load(handle, Loader=yaml.Loader)
        except OSError:
            data = None
    if file_contents is not None:
        data = yaml.load(file_contents, Loader=yaml.Loader)
    return data
