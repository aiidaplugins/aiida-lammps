"""
Tests to aiida-lammps parsers.
"""
import io
import os
from textwrap import dedent

from aiida.orm import FolderData, SinglefileData
from aiida.plugins import ParserFactory

from aiida_lammps.parsers.parse_raw import parse_final_data, parse_logfile
from .utils import TEST_DIR


def get_log():
    """Get the reference values for the log parser"""
    return dedent(
        """\
        units metal
        final_energy: 2.0
        final_cell: 0 1 0 0 1 0 0 1 0
        final_stress: 0 0 0 0 0 0
            """
    )


def get_traj_force():
    """Get the reference values for the trajectory parser"""
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


def test_lammps_base(db_test_app, data_regression):
    """Check if the log file is produced during calculation."""
    filename = os.path.join(
        TEST_DIR,
        "input_files",
        "parsers",
        "log.lammps",
    )
    retrieved = FolderData()
    retrieved.base.repository.put_object_from_file(filename, "log.lammps")
    retrieved.base.repository.put_object_from_filelike(
        io.StringIO(""), "_scheduler-stdout.txt"
    )
    retrieved.base.repository.put_object_from_filelike(
        io.StringIO(""), "_scheduler-stderr.txt"
    )
    inputs = {"script": SinglefileData(io.StringIO(""))}
    calc_node = db_test_app.generate_calcjob_node(
        "lammps.base", retrieved, inputs=inputs
    )
    parser = ParserFactory("lammps.base")
    with db_test_app.sandbox_folder() as temp_path:
        (
            results,
            calcfunction,
        ) = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    assert calcfunction.is_finished_ok
    assert "results" in calcfunction.outputs
    data_regression.check({"results": calcfunction.outputs.results.get_dict()})


def test_lammps_base_timing_info(db_test_app, data_regression):
    """Test that the timing performance is properly parsed."""
    filename = os.path.join(
        TEST_DIR,
        "input_files",
        "parsers",
        "log_alt.lammps",
    )
    retrieved = FolderData()
    retrieved.base.repository.put_object_from_file(filename, "log.lammps")
    retrieved.base.repository.put_object_from_filelike(
        io.StringIO(""), "_scheduler-stdout.txt"
    )
    retrieved.base.repository.put_object_from_filelike(
        io.StringIO(""), "_scheduler-stderr.txt"
    )
    inputs = {"script": SinglefileData(io.StringIO(""))}
    calc_node = db_test_app.generate_calcjob_node(
        "lammps.base", retrieved, inputs=inputs
    )
    parser = ParserFactory("lammps.base")
    with db_test_app.sandbox_folder() as temp_path:
        (
            results,
            calcfunction,
        ) = parser.parse_from_node(  # pylint: disable=unused-variable
            calc_node,
            retrieved_temporary_folder=temp_path.abspath,
        )

    assert calcfunction.is_finished_ok
    assert "results" in calcfunction.outputs
    data_regression.check({"results": calcfunction.outputs.results.get_dict()})


def test_parser_log(data_regression):
    """
    Test the parser for the ``log.lammps`` file.
    """
    filename = os.path.join(
        TEST_DIR,
        "input_files",
        "parsers",
        "log.lammps",
    )

    parsed_data = parse_logfile(filename=filename)
    data_regression.check(parsed_data)


def test_parse_final_variables():
    """
    Test the parser for the final variables
    """
    filename = os.path.join(
        TEST_DIR,
        "input_files",
        "parsers",
        "aiida_lammps.yaml",
    )

    parsed_data = parse_final_data(filename=filename)

    assert isinstance(parsed_data, dict), "the parsed data is not of the correct format"

    assert "final_step" in parsed_data, "no step information present"
    assert "final_etotal" in parsed_data, "no total energy information present"
