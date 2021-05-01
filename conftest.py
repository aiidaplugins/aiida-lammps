"""
initialise a text database and profile
"""
from collections import namedtuple
import io
import os
import shutil
import tempfile

from aiida.plugins import DataFactory
import numpy as np
import pytest

from aiida_lammps.tests.utils import TEST_DIR, AiidaTestApp

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


def pytest_addoption(parser):
    """Define pytest command-line."""
    group = parser.getgroup("aiida_lammps")

    group.addoption(
        "--lammps-workdir",
        dest="lammps_workdir",
        default=None,
        help=(
            "Specify a work directory path for aiida calcjob execution. "
            "If not specified, "
            "a temporary directory is used and deleted after tests execution."
        ),
    )
    group.addoption(
        "--lammps-exec",
        dest="lammps_exec",
        default=None,
        help=("Specify a the lammps executable to run (default: lammps)."),
    )


def get_work_directory(config):
    """Return the aiida work directory to use."""
    if config.getoption("lammps_workdir") is not None:
        return config.getoption("lammps_workdir")
    return None


def pytest_report_header(config):
    """Add header information for pytest execution."""
    return [
        "LAMMPS Executable: {}".format(
            shutil.which(config.getoption("lammps_exec") or "lammps")
        ),
        "LAMMPS Work Directory: {}".format(
            config.getoption("lammps_workdir") or "<TEMP>"
        ),
    ]


@pytest.fixture(scope="function")
def db_test_app(aiida_profile, pytestconfig):
    """Clear the database after each test."""
    exec_name = pytestconfig.getoption("lammps_exec") or "lammps"
    executables = {
        "lammps.md": exec_name,
        "lammps.md.multi": exec_name,
        "lammps.optimize": exec_name,
        "lammps.force": exec_name,
        "lammps.combinate": exec_name,
    }

    test_workdir = get_work_directory(pytestconfig)
    if test_workdir:
        work_directory = test_workdir
    else:
        work_directory = tempfile.mkdtemp()

    yield AiidaTestApp(work_directory, executables, environment=aiida_profile)
    aiida_profile.reset_db()

    if not test_workdir:
        shutil.rmtree(work_directory)


@pytest.fixture(scope="function")
def get_structure_data():
    def _get_structure_data(pkey):
        """return test structure data"""
        if pkey == "Fe":

            cell = [
                [2.848116, 0.000000, 0.000000],
                [0.000000, 2.848116, 0.000000],
                [0.000000, 0.000000, 2.848116],
            ]

            positions = [
                (0.0000000, 0.0000000, 0.0000000),
                (0.5000000, 0.5000000, 0.5000000),
            ]
            fractional = True

            symbols = ["Fe", "Fe"]
            names = ["Fe1", "Fe2"]

        elif pkey == "Ar":

            cell = [
                [3.987594, 0.000000, 0.000000],
                [-1.993797, 3.453358, 0.000000],
                [0.000000, 0.000000, 6.538394],
            ]

            symbols = names = ["Ar"] * 2

            positions = [(0.33333, 0.66666, 0.25000), (0.66667, 0.33333, 0.75000)]
            fractional = True

        elif pkey == "GaN":

            cell = [
                [3.1900000572, 0, 0],
                [-1.5950000286, 2.762621076, 0],
                [0.0, 0, 5.1890001297],
            ]

            positions = [
                (0.6666669, 0.3333334, 0.0000000),
                (0.3333331, 0.6666663, 0.5000000),
                (0.6666669, 0.3333334, 0.3750000),
                (0.3333331, 0.6666663, 0.8750000),
            ]
            fractional = True

            symbols = names = ["Ga", "Ga", "N", "N"]

        elif pkey == "pyrite":

            cell = [
                [5.38, 0.000000, 0.000000],
                [0.000000, 5.38, 0.000000],
                [0.000000, 0.000000, 5.38],
            ]

            positions = [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.338, 0.338, 0.338],
                [0.662, 0.662, 0.662],
                [0.162, 0.662, 0.838],
                [0.838, 0.338, 0.162],
                [0.662, 0.838, 0.162],
                [0.338, 0.162, 0.838],
                [0.838, 0.162, 0.662],
                [0.162, 0.838, 0.338],
            ]
            fractional = True

            symbols = names = ["Fe"] * 4 + ["S"] * 8

        elif pkey == "fes_cubic-zincblende":
            cell = [[2.71, -2.71, 0.0], [2.71, 0.0, 2.71], [0.0, -2.71, 2.71]]
            symbols = names = ["Fe", "S"]
            positions = [[0, 0, 0], [4.065, -4.065, 4.065]]
            fractional = False
        elif pkey == "greigite":
            cell = [[0.0, 4.938, 4.938], [4.938, 0.0, 4.938], [4.938, 4.938, 0.0]]
            positions = [
                (1.2345, 1.2345, 1.2345),
                (8.6415, 8.6415, 8.6415),
                (4.938, 4.938, 4.938),
                (2.469, 4.938, 2.469),
                (4.938, 2.469, 2.469),
                (2.469, 2.469, 4.938),
                (2.473938, 2.473938, 2.473938),
                (4.942938, 7.402062, 4.942938),
                (4.933062, 2.473938, 4.933062),
                (2.473938, 4.933062, 4.933062),
                (7.402062, 4.942938, 4.942938),
                (7.402062, 7.402062, 7.402062),
                (4.942938, 4.942938, 7.402062),
                (4.933062, 4.933062, 2.473938),
            ]
            fractional = False
            symbols = names = [
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "S",
                "S",
                "S",
                "S",
                "S",
                "S",
                "S",
                "S",
            ]

        else:
            raise ValueError("Unknown structure key: {}".format(pkey))

        # create structure
        structure = DataFactory("structure")(cell=cell)
        for position, symbol, name in zip(positions, symbols, names):
            if fractional:
                position = np.dot(position, cell).tolist()
            structure.append_atom(position=position, symbols=symbol, name=name)

        return structure

    return _get_structure_data


potential_data = namedtuple(
    "PotentialTestData", ["type", "data", "structure", "output"]
)


@pytest.fixture(scope="function")
def get_potential_data(get_structure_data):
    def _get_potential_data(pkey):
        """return data to create a potential,
        and accompanying structure data and expected output data to test it with
        """
        if pkey == "eam":
            pair_style = "eam"
            with io.open(
                os.path.join(TEST_DIR, "input_files", "Fe_mm.eam.fs")
            ) as handle:
                potential_dict = {
                    "type": "fs",
                    "file_contents": handle.readlines(),
                    "element_names": ["Fe"],
                }
            structure = get_structure_data("Fe")
            output_dict = {"initial_energy": -8.2441284, "energy": -8.2448702}

        elif pkey == "lennard-jones":

            structure = get_structure_data("Ar")

            # Example LJ parameters for Argon. These may not be accurate at all
            pair_style = "lennard_jones"
            potential_dict = {
                "1  1": "0.01029   3.4    3.5",
                # '2  2':   '1.0      1.0    2.5',
                # '1  2':   '1.0      1.0    2.5'
            }

            output_dict = {
                "initial_energy": 0.0,
                "energy": 0.0,  # TODO should LJ energy be 0?
            }

        elif pkey == "tersoff":

            structure = get_structure_data("GaN")

            potential_dict = {
                "Ga Ga Ga": "1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199",
                "N  N  N": "1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77",
                "Ga Ga N": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000",
                "Ga N  N": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44",
                "N  Ga Ga": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44",
                "N  Ga N ": "1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000",
                "N  N  Ga": "1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000",
                "Ga N  Ga": "1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000",
            }

            pair_style = "tersoff"

            output_dict = {"initial_energy": -18.109886, "energy": -18.110852}

        elif pkey == "reaxff":

            from aiida_lammps.common.reaxff_convert import (
                filter_by_species,
                read_lammps_format,
            )

            pair_style = "reaxff"
            with io.open(
                os.path.join(TEST_DIR, "input_files", "FeCrOSCH.reaxff")
            ) as handle:
                potential_dict = read_lammps_format(
                    handle.read().splitlines(), tolerances={"hbonddist": 7.0}
                )
                potential_dict = filter_by_species(
                    potential_dict, ["Fe core", "S core"]
                )
                for n in ["anglemin", "angleprod", "hbondmin", "torsionprod"]:
                    potential_dict["global"].pop(n)
                potential_dict["control"] = {"safezone": 1.6}
                # potential_dict = {
                #     "file_contents": handle.readlines(),
                #     "control": {"safezone": 1.6},
                #     "global": {"hbonddist": 7.0},
                # }

            structure = get_structure_data("pyrite")

            output_dict = {
                "initial_energy": -1027.9739,
                "energy": -1030.3543,
                "units": "real",
            }

        else:
            raise ValueError("Unknown potential key: {}".format(pkey))

        return potential_data(pair_style, potential_dict, structure, output_dict)

    return _get_potential_data
