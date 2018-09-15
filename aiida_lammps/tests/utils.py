import os
import sys

TEST_COMPUTER = 'localhost-test'


def get_computer(name=TEST_COMPUTER, workdir=None):
    """Get local computer.

    Sets up local computer with 'name' or reads it from database,
    if it exists.

    :param name: Name of local computer
    :param workdir: path to work directory (required if creating a new computer)

    :return: The computer node
    :rtype: :py:class:`aiida.orm.Computer`
    """
    from aiida.orm import Computer
    from aiida.common.exceptions import NotExistent

    try:
        computer = Computer.get(name)
    except NotExistent:

        if workdir is None:
            raise ValueError(
                "to create a new computer, a work directory must be supplied")

        computer = Computer(
            name=name,
            description='localhost computer set up by aiida_lammps tests',
            hostname=name,
            workdir=workdir,
            transport_type='local',
            scheduler_type='direct',
            enabled_state=True)
        computer.store()

    return computer


def get_path_to_executable(executable):
    """ Get path to local executable.

    :param executable: Name of executable in the $PATH variable
    :type executable: str

    :return: path to executable
    :rtype: str
    """
    path = None

    # issue with distutils finding scripts within the python path (i.e. those created by pip install)
    script_path = os.path.join(os.path.dirname(sys.executable), executable)
    if os.path.exists(script_path):
        path = script_path

    if path is None:
        # pylint issue https://github.com/PyCQA/pylint/issues/73
        import distutils.spawn  # pylint: disable=no-name-in-module,import-error
        path = distutils.spawn.find_executable(executable)

    if path is None:
        raise ValueError("{} executable not found in PATH.".format(executable))

    return os.path.abspath(path)