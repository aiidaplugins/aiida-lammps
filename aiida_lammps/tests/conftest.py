"""
initialise a text database and profile
"""
import tempfile
import shutil
import os

from aiida.utils.fixtures import fixture_manager
import pytest

from aiida_lammps.utils import aiida_version, cmp_version


@pytest.fixture(scope='session')
def aiida_profile():
    """setup a test profile for the duration of the tests"""
    with fixture_manager() as fixture_mgr:
        yield fixture_mgr


@pytest.fixture(scope='function')
def new_workdir():
    """get a new temporary folder to use as the computer's wrkdir"""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture(scope='function')
def new_database(aiida_profile):
    """clear the database after each test"""
    yield aiida_profile
    aiida_profile.reset_db()


@pytest.fixture(scope='function')
def new_database_with_daemon(aiida_profile):
    """When you run something in aiida v1, it will be done in its own runner,
    which is a mini daemon in and of its own.
    However, in 0.12 the global daemon is the only one that can
    submit, update and retrieve job calculations.
    Therefore, we must configure and start it before running JobProcesses
    """
    if aiida_version() < cmp_version('1.0.0a1'):
        from aiida.backends.utils import set_daemon_user
        from aiida.cmdline.commands.daemon import Daemon
        from aiida.common import setup

        set_daemon_user(aiida_profile.email)
        daemon = Daemon()
        daemon.logfile = os.path.join(aiida_profile.config_dir,
                                      setup.LOG_SUBDIR, setup.CELERY_LOG_FILE)
        daemon.pidfile = os.path.join(aiida_profile.config_dir,
                                      setup.LOG_SUBDIR, setup.CELERY_PID_FILE)
        daemon.celerybeat_schedule = os.path.join(aiida_profile.config_dir,
                                                  setup.DAEMON_SUBDIR,
                                                  'celerybeat-schedule')

        if daemon.get_daemon_pid() is None:
            daemon.daemon_start()
        else:
            daemon.daemon_restart()
        yield aiida_profile
        daemon.kill_daemon()
        aiida_profile.reset_db()
    else:
        yield aiida_profile
        aiida_profile.reset_db()

