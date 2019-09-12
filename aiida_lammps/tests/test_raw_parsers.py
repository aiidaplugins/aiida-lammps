import os

import numpy as np

from aiida_lammps.tests.utils import TEST_DIR
from aiida_lammps.common.raw_parsers import (
    iter_lammps_trajectories,
    parse_trajectory_file,
)


def test_iter_lammps_trajectories():
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    with open(path) as handle:
        time_steps = [t.timestep for t in iter_lammps_trajectories(handle)]
    assert time_steps == [
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


def test_parse_trajectory_file(data_regression):
    path = os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
    variables = parse_trajectory_file(path)
    data_regression.check({k: np.array(v).shape for k, v in variables.items()})
