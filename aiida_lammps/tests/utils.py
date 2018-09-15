import os
import sys

TEST_COMPUTER = 'localhost-test'


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


def get_computer(name=TEST_COMPUTER, workdir=None, configure=False):
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

        if configure:
            try:
                # aiida-core v1
                from aiida.control.computer import configure_computer
                configure_computer(computer)
            except ImportError:
                configure_computer_v012(computer)

    return computer


def get_backend():
    """ Return database backend.

    Reads from 'TEST_AIIDA_BACKEND' environment variable.
    Defaults to django backend.
    """
    from aiida.backends.profile import BACKEND_DJANGO, BACKEND_SQLA
    if os.environ.get('TEST_AIIDA_BACKEND') == BACKEND_SQLA:
        return BACKEND_SQLA
    return BACKEND_DJANGO


def is_sqla_backend():
    """return True if the backend is sqlalchemy"""
    from aiida.backends.profile import BACKEND_SQLA
    return get_backend() == BACKEND_SQLA


def configure_computer_v012(computer):
    """Configure the authentication information for a given computer

    adapted from aiida-core v0.12.2:
    aiida_core.aiida.cmdline.commands.computer.Computer.computer_configure

    :param computer: the computer to authenticate against
    :param authparams: a dictionary of additional authorisation parameters to use (in string format)
    :return:
    """
    from aiida.backends.utils import get_automatic_user

    user = get_automatic_user()

    authinfo = get_auth_info_v012(computer, user)

    authinfo.set_auth_params({})
    authinfo.save()


def get_auth_info_v012(computer, user):
    from aiida.backends.profile import BACKEND_DJANGO, BACKEND_SQLA
    from django.core.exceptions import ObjectDoesNotExist

    BACKEND = get_backend()
    if BACKEND == BACKEND_DJANGO:
        from aiida.backends.djsite.db.models import DbAuthInfo

        try:
            authinfo = DbAuthInfo.objects.get(
                dbcomputer=computer.dbcomputer, aiidauser=user)

        except ObjectDoesNotExist:
            authinfo = DbAuthInfo(
                dbcomputer=computer.dbcomputer, aiidauser=user)

    elif BACKEND == BACKEND_SQLA:
        raise NotImplementedError()

    return authinfo


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


def run_get_node(process, inputs_dict):
    """ an implementation of run_get_node which is compatible with both aiida v0.12 and v1.0.0

    it will also convert "options" "label" and "description" to/from the _ variant

    :param process: a process
    :param inputs_dict: a dictionary of inputs
    :type inputs_dict: dict
    :return: the calculation Node
    """
    if aiida_version() < cmp_version("1.0.0a1"):
        for key in ["options", "label", "description"]:
            if key in inputs_dict:
                inputs_dict["_" + key] = inputs_dict.pop(key)
        workchain = process.new_instance(inputs=inputs_dict)
        workchain.run_until_complete()
        calcnode = workchain.calc
    else:
        from aiida.work.launch import run_get_node  # pylint: disable=import-error
        for key in ["_options", "_label", "_description"]:
            if key in inputs_dict:
                inputs_dict[key[1:]] = inputs_dict.pop(key)
        _, calcnode = run_get_node(process, **inputs_dict)

    return calcnode


def get_calc_log(calcnode):
    """get a formatted string of the calculation log"""
    from aiida.backends.utils import get_log_messages
    import json
    import datetime

    def default(o):
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()

    log_string = "- Calc State:\n{0}\n- Scheduler Out:\n{1}\n- Scheduler Err:\n{2}\n- Log:\n{3}".format(
        calcnode.get_state(),
        calcnode.get_scheduler_output(), calcnode.get_scheduler_error(),
        json.dumps(get_log_messages(calcnode), default=default, indent=2))
    return log_string
