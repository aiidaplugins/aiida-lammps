"""
Tests to aiida-lammps parsers.
"""
from textwrap import dedent
import io
import os
import pytest
import yaml
from aiida.cmdline.utils.common import get_calcjob_report
from aiida.orm import FolderData
from aiida.plugins import ParserFactory
from aiida_lammps.tests.utils import TEST_DIR
from aiida_lammps.common.raw_parsers import parse_logfile, parse_final_data


def get_log():
    """Get the reference values for the log parser"""
    return dedent("""\
        units metal
        final_energy: 2.0
        final_cell: 0 1 0 0 1 0 0 1 0
        final_stress: 0 0 0 0 0 0
            """)


def get_traj_force():
    """Get the reference values for the trajectory parser"""
    return dedent("""\
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
        """)


@pytest.mark.parametrize(
    'plugin_name',
    ['lammps.force', 'lammps.optimize', 'lammps.md', 'lammps.md.multi'])
def test_missing_log(db_test_app, plugin_name):
    """Check if the log file is produced during calculation."""
    retrieved = FolderData()

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)
    with db_test_app.sandbox_folder() as temp_path:
        results, calcfunction = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (calcfunction.exit_status ==
            calc_node.process_class.exit_codes.ERROR_LOG_FILE_MISSING.status)


@pytest.mark.parametrize(
    'plugin_name',
    ['lammps.force', 'lammps.optimize', 'lammps.md', 'lammps.md.multi'])
def test_missing_traj(db_test_app, plugin_name):
    """Check if the trajectory file is produced during calculation."""
    retrieved = FolderData()
    retrieved.put_object_from_filelike(io.StringIO(get_log()), 'log.lammps')
    retrieved.put_object_from_filelike(io.StringIO(''),
                                       '_scheduler-stdout.txt')
    retrieved.put_object_from_filelike(io.StringIO(''),
                                       '_scheduler-stderr.txt')

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)
    with db_test_app.sandbox_folder() as temp_path:
        results, calcfunction = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (calcfunction.exit_status ==
            calc_node.process_class.exit_codes.ERROR_TRAJ_FILE_MISSING.status)


@pytest.mark.parametrize(
    'plugin_name',
    ['lammps.force', 'lammps.optimize', 'lammps.md', 'lammps.md.multi'])
def test_empty_log(db_test_app, plugin_name):
    """Check if the lammps log is empty."""
    retrieved = FolderData()
    for filename in [
            'log.lammps',
            'trajectory.lammpstrj',
            '_scheduler-stdout.txt',
            '_scheduler-stderr.txt',
    ]:
        retrieved.put_object_from_filelike(io.StringIO(''), filename)

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)

    with db_test_app.sandbox_folder() as temp_path:
        with temp_path.open('x-trajectory.lammpstrj', 'w'):
            pass
        results, calcfunction = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (calcfunction.exit_status ==
            calc_node.process_class.exit_codes.ERROR_LOG_PARSING.status)


@pytest.mark.parametrize(
    'plugin_name',
    ['lammps.force', 'lammps.optimize', 'lammps.md', 'lammps.md.multi'])
def test_empty_traj(db_test_app, plugin_name):
    """Check if the lammps trajectory file is empty."""
    retrieved = FolderData()
    retrieved.put_object_from_filelike(io.StringIO(get_log()), 'log.lammps')
    for filename in [
            'trajectory.lammpstrj',
            '_scheduler-stdout.txt',
            '_scheduler-stderr.txt',
    ]:
        retrieved.put_object_from_filelike(io.StringIO(''), filename)

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)
    with db_test_app.sandbox_folder() as temp_path:
        with temp_path.open('x-trajectory.lammpstrj', 'w'):
            pass
        results, calcfunction = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (calcfunction.exit_status ==
            calc_node.process_class.exit_codes.ERROR_TRAJ_PARSING.status)


@pytest.mark.parametrize(
    'plugin_name',
    ['lammps.force', 'lammps.optimize', 'lammps.md', 'lammps.md.multi'])
def test_run_error(db_test_app, plugin_name):
    """Check if the parser runs without producing errors."""
    retrieved = FolderData()
    retrieved.put_object_from_filelike(
        io.StringIO(get_log()),
        'log.lammps',
    )
    retrieved.put_object_from_filelike(
        io.StringIO(get_traj_force()),
        'x-trajectory.lammpstrj',
    )
    retrieved.put_object_from_filelike(
        io.StringIO('ERROR description'),
        '_scheduler-stdout.txt',
    )
    retrieved.put_object_from_filelike(
        io.StringIO(''),
        '_scheduler-stderr.txt',
    )

    calc_node = db_test_app.generate_calcjob_node(plugin_name, retrieved)
    parser = ParserFactory(plugin_name)

    with db_test_app.sandbox_folder() as temp_path:
        with temp_path.open('x-trajectory.lammpstrj', 'w') as handle:
            handle.write(get_traj_force())
        results, calcfunction = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    print(get_calcjob_report(calc_node))

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert (calcfunction.exit_status ==
            calc_node.process_class.exit_codes.ERROR_LAMMPS_RUN.status)


def test_parser_log():
    """
    Test the parser for the ``log.lammps`` file.
    """
    filename = os.path.join(
        TEST_DIR,
        'input_files',
        'parsers',
        'log.lammps',
    )

    parsed_data = parse_logfile(filename=filename)

    reference_filename = os.path.join(
        TEST_DIR,
        'test_raw_parsers',
        'test_parse_log.yaml',
    )

    with io.open(reference_filename) as handle:
        reference_data = yaml.load(handle, Loader=yaml.Loader)

    assert parsed_data == reference_data, 'content of "log.lammps" differs from reference'


def test_parse_final_variables():
    """
    Test the parser for the final variables
    """
    filename = os.path.join(
        TEST_DIR,
        'input_files',
        'parsers',
        'aiida_lammps.yaml',
    )

    parsed_data = parse_final_data(filename=filename)

    assert isinstance(parsed_data,
                      dict), 'the parsed data is not of the correct format'

    assert 'final_step' in parsed_data, 'no step information present'
    assert 'final_etotal' in parsed_data, 'no total energy information present'
