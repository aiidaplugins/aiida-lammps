from textwrap import dedent
import pytest
import six
from aiida.orm import FolderData
from aiida.cmdline.utils.common import get_calcjob_report


def get_log():
    return six.ensure_text(dedent("""\
        units metal
        final_energy: 2.0
        final_cell: 0 1 0 0 1 0 0 1 0
        final_stress: 0 0 0 0 0 0
            """))


def get_traj_force():
    return six.ensure_text(dedent("""\
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
        """))


@pytest.mark.parametrize('plugin_name', [
    "lammps.force",
    "lammps.optimize",
    # "lammps.md", # requires retrieved_temporary_folder (awaiting aiidateam/aiida_core#3061)
])
def test_missing_log(db_test_app, plugin_name):

    retrieved = FolderData()

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = db_test_app.get_parser_cls(plugin_name)
    results, calcfunction = parser.parse_from_node(calc_node)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == calc_node.process_class.exit_codes.ERROR_LOG_FILE_MISSING.status


@pytest.mark.parametrize('plugin_name', [
    "lammps.force",
    "lammps.optimize",
    # "lammps.md", # requires retrieved_temporary_folder (awaiting aiidateam/aiida_core#3061)
])
def test_missing_traj(db_test_app, plugin_name):

    retrieved = FolderData()
    with retrieved.open('log.lammps', 'w'):
        pass

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = db_test_app.get_parser_cls(plugin_name)
    results, calcfunction = parser.parse_from_node(calc_node)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == calc_node.process_class.exit_codes.ERROR_TRAJ_FILE_MISSING.status


@pytest.mark.parametrize('plugin_name', [
    "lammps.force",
    "lammps.optimize",
    # "lammps.md", # requires retrieved_temporary_folder (awaiting aiidateam/aiida_core#3061)
])
def test_empty_log(db_test_app, plugin_name):

    retrieved = FolderData()
    with retrieved.open('log.lammps', 'w'):
        pass
    with retrieved.open('trajectory.lammpstrj', 'w'):
        pass
    with retrieved.open('_scheduler-stdout.txt', 'w'):
        pass
    with retrieved.open('_scheduler-stderr.txt', 'w'):
        pass

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = db_test_app.get_parser_cls(plugin_name)
    results, calcfunction = parser.parse_from_node(calc_node)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == calc_node.process_class.exit_codes.ERROR_LOG_PARSING.status


@pytest.mark.parametrize('plugin_name', [
    "lammps.force",
    "lammps.optimize",
    # "lammps.md", # requires retrieved_temporary_folder (awaiting aiidateam/aiida_core#3061)
])
def test_empty_traj(db_test_app, plugin_name):

    retrieved = FolderData()
    with retrieved.open('log.lammps', 'w') as handle:
        handle.write(get_log())
    with retrieved.open('trajectory.lammpstrj', 'w') as handle:
        pass
    with retrieved.open('_scheduler-stdout.txt', 'w'):
        pass
    with retrieved.open('_scheduler-stderr.txt', 'w'):
        pass

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = db_test_app.get_parser_cls(plugin_name)
    results, calcfunction = parser.parse_from_node(calc_node)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == calc_node.process_class.exit_codes.ERROR_TRAJ_PARSING.status


@pytest.mark.parametrize('plugin_name', [
    "lammps.force",
    # "lammps.optimize",
    # "lammps.md", # requires retrieved_temporary_folder (awaiting aiidateam/aiida_core#3061)
])
def test_run_error(db_test_app, plugin_name):

    retrieved = FolderData()
    with retrieved.open('log.lammps', 'w') as handle:
        handle.write(get_log())
    with retrieved.open('trajectory.lammpstrj', 'w') as handle:
        handle.write(get_traj_force())
    with retrieved.open('_scheduler-stdout.txt', 'w') as handle:
        handle.write(six.ensure_text('ERROR description'))
    with retrieved.open('_scheduler-stderr.txt', 'w'):
        pass

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = db_test_app.get_parser_cls(plugin_name)
    results, calcfunction = parser.parse_from_node(calc_node)

    print(get_calcjob_report(calc_node))

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == calc_node.process_class.exit_codes.ERROR_LAMMPS_RUN.status
