"""
initialise a test database and profile
"""
from __future__ import annotations

import collections
import os
import pathlib
import shutil
import tempfile
from typing import Any

from aiida import orm
from aiida.common.datastructures import CalcInfo
from aiida.common.links import LinkType
from aiida.engine import CalcJob
import numpy as np
import pytest
import yaml

from tests.utils import TEST_DIR, AiidaTestApp

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
        f'LAMMPS Executable: {shutil.which(config.getoption("lammps_exec") or "lammps")}',
        f'LAMMPS Work Directory: {config.getoption("lammps_workdir") or "<TEMP>"}',
    ]


@pytest.fixture
def filepath_tests() -> pathlib.Path:
    """Return the path to the tests folder."""
    return pathlib.Path(__file__).resolve().parent / "tests"


@pytest.fixture(scope="function")
def db_test_app(aiida_profile, pytestconfig):
    """Clear the database after each test."""
    exec_name = pytestconfig.getoption("lammps_exec") or "lammps"
    executables = {
        "lammps.base": exec_name,
    }

    test_workdir = get_work_directory(pytestconfig)
    if test_workdir:
        work_directory = test_workdir
    else:
        work_directory = tempfile.mkdtemp()

    yield AiidaTestApp(work_directory, executables, environment=aiida_profile)
    aiida_profile.clear_profile()

    if not test_workdir:
        shutil.rmtree(work_directory)


@pytest.fixture
def generate_calc_job(tmp_path):
    """Create a :class:`aiida.engine.CalcJob` instance with the given inputs.

    The fixture will call ``prepare_for_submission`` and return a tuple of the temporary folder that was passed to it,
    as well as the ``CalcInfo`` instance that it returned.
    """

    def factory(
        entry_point_name: str,
        inputs: dict[str, Any] | None = None,
        return_process: bool = False,
    ) -> tuple[pathlib.Path, CalcInfo] | CalcJob:
        """Create a :class:`aiida.engine.CalcJob` instance with the given inputs.

        :param entry_point_name: The entry point name of the calculation job plugin to run.
        :param inputs: The dictionary of inputs for the calculation job.
        :param return_process: Flag, if ``True``, return the constructed ``CalcJob`` instance instead of the tuple of
            the temporary folder and ``CalcInfo`` instance.
        """
        from aiida.common.folders import Folder
        from aiida.engine.utils import instantiate_process
        from aiida.manage import get_manager
        from aiida.plugins import CalculationFactory

        runner = get_manager().get_runner()
        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs or {})
        calc_info = process.prepare_for_submission(Folder(tmp_path))

        if return_process:
            return process

        return tmp_path, calc_info

    return factory


@pytest.fixture
def generate_calc_job_node(filepath_tests, aiida_computer_local, tmp_path):
    """Create and return a :class:`aiida.orm.CalcJobNode` instance."""

    def flatten_inputs(inputs, prefix=""):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, collections.abc.Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + "__"))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def factory(
        entry_point: str,
        test_name: str,
        inputs: dict = None,
        retrieve_temporary_list: list[str] | None = None,
    ):
        """Create and return a :class:`aiida.orm.CalcJobNode` instance."""
        node = orm.CalcJobNode(
            computer=aiida_computer_local(),
            process_type=f"aiida.calculations:{entry_point}",
        )

        if inputs:
            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(
                    input_node, link_type=LinkType.INPUT_CALC, link_label=link_label
                )

        node.store()

        filepath_retrieved = (
            filepath_tests
            / "parsers"
            / "fixtures"
            / entry_point.split(".")[-1]
            / test_name
        )

        retrieved = orm.FolderData()
        retrieved.base.repository.put_object_from_tree(filepath_retrieved)
        retrieved.base.links.add_incoming(
            node, link_type=LinkType.CREATE, link_label="retrieved"
        )
        retrieved.store()

        if retrieve_temporary_list:
            for pattern in retrieve_temporary_list:
                for filename in filepath_retrieved.glob(pattern):
                    filepath = tmp_path / filename.relative_to(filepath_retrieved)
                    filepath.write_bytes(filename.read_bytes())

            return node, tmp_path

        return node

    return factory


@pytest.fixture(scope="function")
def get_structure_data():
    """get the structure data for the simulation."""

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

            positions = [
                (0.33333, 0.66666, 0.25000),
                (0.66667, 0.33333, 0.75000),
            ]
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
            raise ValueError(f"Unknown structure key: {pkey}")

        # create structure
        structure = orm.StructureData(cell=cell)
        for position, symbol, name in zip(positions, symbols, names):
            if fractional:
                position = np.dot(position, cell).tolist()
            structure.append_atom(position=position, symbols=symbol, name=name)

        return structure

    return _get_structure_data


@pytest.fixture(scope="function")
def get_lammps_potential_data(get_structure_data):
    """Get the potential information for different types of potentials.

    :param get_structure_data: Structure to be used in the simulation
    :type get_structure_data: orm.StructureData
    """

    def _get_lammps_potential_data(pkey):
        """return data to create a potential,
        and accompanying structure data and expected output data to test it with
        """
        output_dict = {}
        if pkey == "eam_alloy":
            output_dict["filename"] = os.path.join(
                "tests",
                "input_files",
                "potentials",
                "FeW_MO_737567242631_000.eam.alloy",
            )

            filename_parameters = os.path.join(
                "tests",
                "input_files",
                "parameters",
                "FeW_MO_737567242631_000.eam.alloy.yaml",
            )

            with open(filename_parameters, encoding="utf8") as handle:
                output_dict["parameters"] = yaml.load(handle, yaml.SafeLoader)

            with open(output_dict["filename"], encoding="utf8") as handle:
                output_dict["potential_data"] = handle.read()
            output_dict["structure"] = get_structure_data("Fe")

        if pkey == "tersoff":
            output_dict["filename"] = os.path.join(
                "tests",
                "input_files",
                "potentials",
                "Fe_MO_137964310702_004.tersoff",
            )

            filename_parameters = os.path.join(
                "tests",
                "input_files",
                "parameters",
                "Fe_MO_137964310702_004.tersoff.yaml",
            )

            with open(filename_parameters, encoding="utf8") as handle:
                output_dict["parameters"] = yaml.load(handle, yaml.SafeLoader)

            with open(output_dict["filename"], encoding="utf8") as handle:
                output_dict["potential_data"] = handle.read()
            output_dict["structure"] = get_structure_data("Fe")

        if pkey == "meam":
            output_dict["filename"] = os.path.join(
                "tests",
                "input_files",
                "potentials",
                "Fe_MO_492310898779_001.meam",
            )

            filename_parameters = os.path.join(
                "tests",
                "input_files",
                "parameters",
                "Fe_MO_492310898779_001.meam.yaml",
            )

            with open(filename_parameters, encoding="utf8") as handle:
                output_dict["parameters"] = yaml.load(handle, yaml.SafeLoader)

            with open(output_dict["filename"], encoding="utf8") as handle:
                output_dict["potential_data"] = handle.read()
            output_dict["structure"] = get_structure_data("Fe")

        if pkey == "morse":
            output_dict["filename"] = os.path.join(
                "tests",
                "input_files",
                "potentials",
                "Fe_MO_331285495617_004.morse",
            )

            filename_parameters = os.path.join(
                "tests",
                "input_files",
                "parameters",
                "Fe_MO_331285495617_004.morse.yaml",
            )

            with open(filename_parameters, encoding="utf8") as handle:
                output_dict["parameters"] = yaml.load(handle, yaml.SafeLoader)

            with open(output_dict["filename"], encoding="utf8") as handle:
                output_dict["potential_data"] = handle.read()
            output_dict["structure"] = get_structure_data("Fe")

        return output_dict

    return _get_lammps_potential_data
