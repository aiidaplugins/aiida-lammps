"""General utility functions for aiida-lammps"""
import enum

from aiida import __version__ as aiida_version_
from packaging import version


def aiida_version():
    """get the version of aiida in use

    :returns: packaging.version.Version
    """
    return version.parse(aiida_version_)


def cmp_version(string):
    """convert a version string to a packaging.version.Version"""
    return version.parse(string)


class RestartTypes(enum.Enum):
    """Enumeration of the known relax types"""

    FROM_SCRATCH = "from_scratch"
    FROM_RESTARTFILE = "from_restartfile"
    FROM_REMOTEFOLDER = "from_remotefolder"
    FROM_STRUCTURE = "from_structure"
