from io import StringIO
from textwrap import dedent

from aiida.cmdline.utils.common import get_calcjob_report
from aiida.orm import FolderData
from aiida.plugins import ParserFactory
import pytest


def get_log():
    return dedent(
        """\
        units metal
        final_energy: 2.0
        final_cell: 0 1 0 0 1 0 0 1 0
        final_stress: 0 0 0 0 0 0
            """
    )


def get_traj_force():
    return dedent(
        """\
        ITEM: TIMESTEP
        0
        ITEM: NUMBER OF ATOMS
        6
        ITEM: BOX BOUNDS pp pp pp
        0 4.44
        0 5.39
        0 3.37
        ITEM: ATOMS element fx fy fz
        Fe     0.0000000000     0.0000000000    -0.0000000000
        Fe     0.0000000000    -0.0000000000     0.0000000000
        S   -25.5468278966    20.6615772179    -0.0000000000
        S   -25.5468278966   -20.6615772179    -0.0000000000
        S    25.5468278966    20.6615772179    -0.0000000000
        S    25.5468278966   -20.6615772179     0.0000000000
        """
    )


@pytest.mark.parametrize(
    "plugin_name", ["lammps.force", "lammps.optimize", "lammps.md", "lammps.md.multi"]
)
def test_missing_log(db_test_app, plugin_name):

    retrieved = FolderData()

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)
    with db_test_app.sandbox_folder() as temp_path:
        results, calcfunction = parser.parse_from_node(
            calc_node, retrieved_temporary_folder=temp_path.abspath
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (
        calcfunction.exit_status
        == calc_node.process_class.exit_codes.ERROR_LOG_FILE_MISSING.status
    )


@pytest.mark.parametrize(
    "plugin_name", ["lammps.force", "lammps.optimize", "lammps.md", "lammps.md.multi"]
)
def test_missing_traj(db_test_app, plugin_name):

    retrieved = FolderData()
    retrieved.put_object_from_filelike(StringIO(get_log()), "log.lammps")
    retrieved.put_object_from_filelike(StringIO(""), "_scheduler-stdout.txt")
    retrieved.put_object_from_filelike(StringIO(""), "_scheduler-stderr.txt")

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)
    with db_test_app.sandbox_folder() as temp_path:
        results, calcfunction = parser.parse_from_node(
            calc_node, retrieved_temporary_folder=temp_path.abspath
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (
        calcfunction.exit_status
        == calc_node.process_class.exit_codes.ERROR_TRAJ_FILE_MISSING.status
    )


@pytest.mark.parametrize(
    "plugin_name", ["lammps.force", "lammps.optimize", "lammps.md", "lammps.md.multi"]
)
def test_empty_log(db_test_app, plugin_name):

    retrieved = FolderData()
    for filename in [
        "log.lammps",
        "trajectory.lammpstrj",
        "_scheduler-stdout.txt",
        "_scheduler-stderr.txt",
    ]:
        retrieved.put_object_from_filelike(StringIO(""), filename)

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)

    with db_test_app.sandbox_folder() as temp_path:
        with temp_path.open("x-trajectory.lammpstrj", "w"):
            pass
        results, calcfunction = parser.parse_from_node(
            calc_node, retrieved_temporary_folder=temp_path.abspath
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (
        calcfunction.exit_status
        == calc_node.process_class.exit_codes.ERROR_LOG_PARSING.status
    )


@pytest.mark.parametrize(
    "plugin_name", ["lammps.force", "lammps.optimize", "lammps.md", "lammps.md.multi"]
)
def test_empty_traj(db_test_app, plugin_name):

    retrieved = FolderData()
    retrieved.put_object_from_filelike(StringIO(get_log()), "log.lammps")
    for filename in [
        "trajectory.lammpstrj",
        "_scheduler-stdout.txt",
        "_scheduler-stderr.txt",
    ]:
        retrieved.put_object_from_filelike(StringIO(""), filename)

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)
    with db_test_app.sandbox_folder() as temp_path:
        with temp_path.open("x-trajectory.lammpstrj", "w"):
            pass
        results, calcfunction = parser.parse_from_node(
            calc_node, retrieved_temporary_folder=temp_path.abspath
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (
        calcfunction.exit_status
        == calc_node.process_class.exit_codes.ERROR_TRAJ_PARSING.status
    )


@pytest.mark.parametrize(
    "plugin_name", ["lammps.force", "lammps.optimize", "lammps.md", "lammps.md.multi"]
)
def test_run_error(db_test_app, plugin_name):

    retrieved = FolderData()
    retrieved.put_object_from_filelike(StringIO(get_log()), "log.lammps")
    retrieved.put_object_from_filelike(
        StringIO(get_traj_force()), "x-trajectory.lammpstrj"
    )
    retrieved.put_object_from_filelike(
        StringIO("ERROR description"), "_scheduler-stdout.txt"
    )
    retrieved.put_object_from_filelike(StringIO(""), "_scheduler-stderr.txt")

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)

    with db_test_app.sandbox_folder() as temp_path:
        with temp_path.open("x-trajectory.lammpstrj", "w") as handle:
            handle.write(get_traj_force())
        results, calcfunction = parser.parse_from_node(
            calc_node, retrieved_temporary_folder=temp_path.abspath
        )

    print(get_calcjob_report(calc_node))

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (
        calcfunction.exit_status
        == calc_node.process_class.exit_codes.ERROR_LAMMPS_RUN.status
    )
