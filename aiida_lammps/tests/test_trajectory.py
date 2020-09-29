import io
import os

from aiida_lammps.common.parse_trajectory import create_structure, iter_trajectories
from aiida_lammps.data.trajectory import LammpsTrajectory
from aiida_lammps.tests.utils import TEST_DIR, recursive_round


def test_iter_trajectories(data_regression):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    output = []
    with io.open(path) as handle:
        for tstep in iter_trajectories(handle):
            dct = dict(tstep._asdict())
            dct.pop("cell")
            dct.pop("lines")
            output.append(dct)
    data_regression.check(output)


def test_create_structure(db_test_app, data_regression):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    with io.open(path) as handle:
        traj_block = next(iter_trajectories(handle))

    structure = create_structure(traj_block)
    data_regression.check(recursive_round(structure.attributes, 2, apply_lists=True))


def test_lammps_trajectory_data(db_test_app, data_regression):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    data = LammpsTrajectory(path)
    data_regression.check(data.attributes)


def test_lammpstraj_get_step_string(db_test_app, file_regression):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    data = LammpsTrajectory(path)
    file_regression.check(data.get_step_string(-1))


def test_lammpstraj_get_step_struct(db_test_app, data_regression):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    data = LammpsTrajectory(path)
    data_regression.check(
        recursive_round(data.get_step_structure(-1).attributes, 2, apply_lists=True)
    )


def test_lammpstraj_timesteps(db_test_app):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    data = LammpsTrajectory(path)
    assert data.time_steps == [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]
