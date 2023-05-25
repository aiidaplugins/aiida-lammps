import io
import textwrap

from aiida import orm
import pytest

from aiida_lammps.calculations.raw import LammpsRawCalculation


def test_script(generate_calc_job, aiida_local_code_factory):
    """Test the ``script`` input."""
    content = textwrap.dedent(
        """
        "velocity      all create 1.44 87287 loop geom
        "pair_style    lj/cut 2.5
        "pair_coeff    1 1 1.0 1.0 2.5
        "neighbor      0.3 bin
        "neigh_modify  delay 0 every 20 check no
        "fix           1 all nve
        "run           10000
        """
    )
    inputs = {
        "code": aiida_local_code_factory("lammps.raw", "bash"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "script": orm.SinglefileData(io.StringIO(content)),
    }

    tmp_path, calc_info = generate_calc_job("lammps.raw", inputs)
    assert (tmp_path / LammpsRawCalculation.FILENAME_INPUT).read_text() == content


def test_files_invalid(generate_calc_job, aiida_local_code_factory):
    """Test the ``files`` input valdiation.

    The list of filenames that will be used to write to the working directory needs to be unique.
    """
    # Create two ``SinglefileData`` nodes without specifying an explicit filename. This will cause the default to be
    # used, and so both will have the same filename, which should trigger the validation error.
    inputs = {
        "code": aiida_local_code_factory("lammps.raw", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "files": {
            "file_a": orm.SinglefileData(io.StringIO("content")),
            "file_b": orm.SinglefileData(io.StringIO("content")),
        },
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    with pytest.raises(
        ValueError,
        match=r"The list of filenames of the ``files`` input is not unique:.*",
    ):
        generate_calc_job("lammps.raw", inputs)


def test_files(generate_calc_job, aiida_local_code_factory):
    """Test the ``files`` input."""
    inputs = {
        "code": aiida_local_code_factory("lammps.raw", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "files": {
            "file_a": orm.SinglefileData(
                io.StringIO("content a"), filename="file_a.txt"
            ),
            "file_b": orm.SinglefileData(
                io.StringIO("content b"), filename="file_b.txt"
            ),
        },
        "filenames": {"file_b": "custom_filename.txt"},
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    tmp_path, calc_info = generate_calc_job("lammps.raw", inputs)
    assert sorted(calc_info.provenance_exclude_list) == [
        "custom_filename.txt",
        "file_a.txt",
    ]
    assert (tmp_path / "file_a.txt").read_text() == "content a"
    assert (tmp_path / "custom_filename.txt").read_text() == "content b"


def test_filenames_invalid(generate_calc_job, aiida_local_code_factory):
    """Test the ``filenames`` input valdiation.

    The list of filenames that will be used to write to the working directory needs to be unique.
    """
    # Create two ``SinglefileData`` nodes with unique filenames but override them using the ``filenames`` input to use
    # the same filename, and so both will have the same filename, which should trigger the validation error.
    inputs = {
        "code": aiida_local_code_factory("lammps.raw", "bash"),
        "script": orm.SinglefileData(io.StringIO("")),
        "files": {
            "file_a": orm.SinglefileData(io.StringIO("content"), filename="file_a.txt"),
            "file_b": orm.SinglefileData(io.StringIO("content"), filename="file_b.txt"),
        },
        "filenames": {
            "file_a": "file.txt",
            "file_b": "file.txt",
        },
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    with pytest.raises(
        ValueError,
        match=r"The list of filenames of the ``files`` input is not unique:.*",
    ):
        generate_calc_job("lammps.raw", inputs)
