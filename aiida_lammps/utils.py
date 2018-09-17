def aiida_version():
    """get the version of aiida in use

    :returns: packaging.version.Version
    """
    from aiida import __version__ as aiida_version_
    from packaging import version
    return version.parse(aiida_version_)


def cmp_version(string):
    """convert a version string to a packaging.version.Version"""
    from packaging import version
    return version.parse(string)