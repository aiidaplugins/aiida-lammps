"""
initialise a test database and profile
"""

# pylint: disable=redefined-outer-name
from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
import os
import pathlib
import shutil
import tempfile
from typing import Any

from aiida import orm
from aiida.common import AttributeDict, CalcInfo, LinkType, exceptions
from aiida.engine import Process
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.plugins import WorkflowFactory
from aiida.plugins.entry_point import format_entry_point_string
from aiida_lammps.calculations.base import LammpsBaseCalculation
from aiida_lammps.data.potential import LammpsPotentialData
from aiida_lammps.data.trajectory import LammpsTrajectory
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
def structure_parameters() -> AttributeDict:
    parameteters = AttributeDict({"dimension": 2, "boundary": ["p", "p", "f"]})
    return parameteters


@pytest.fixture
def filepath_tests() -> pathlib.Path:
    """Return the path to the tests folder."""
    return pathlib.Path(__file__).resolve().parent / "tests"


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return an ``InstalledCode`` instance configured to run calculations of given entry point on localhost."""

    def _fixture_code(entry_point_name):
        label = f"test.{entry_point_name}"

        try:
            return orm.load_code(label=label)
        except exceptions.NotExistent:
            return orm.InstalledCode(
                label=label,
                computer=fixture_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point_name,
            )

    return _fixture_code


@pytest.fixture
def parameters_minimize() -> AttributeDict:
    """
    Set of parameters for a minimization calculation

    :return: parameters for a minimization calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        "property/atom": [
            {
                "type": [
                    {"keyword": " ", "value": "fx"},
                    {"keyword": " ", "value": "fy"},
                ],
                "group": "all",
            }
        ],
    }

    parameters.minimize = {
        "style": "cg",
        "energy_tolerance": 1e-5,
        "force_tolerance": 1e-5,
        "max_evaluations": 5000,
        "max_iterations": 5000,
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 100,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    parameters.fix = {
        "box/relax": [{"group": "all", "type": ["iso", 0.0, "vmax", 0.001]}]
    }
    parameters.restart = {"print_final": True}

    return parameters


@pytest.fixture
def parameters_minimize_groups() -> AttributeDict:
    """
    Set of parameters for a minimization calculation using groups

    :return: parameters for a minimization calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        "ke": [{"type": [{"keyword": " ", "value": " "}], "group": "test"}],
        "property/atom": [
            {
                "type": [
                    {"keyword": " ", "value": "fx"},
                    {"keyword": " ", "value": "fy"},
                ],
                "group": "all",
            }
        ],
    }

    parameters.minimize = {
        "style": "cg",
        "energy_tolerance": 1e-5,
        "force_tolerance": 1e-5,
    }

    parameters.structure = {
        "atom_style": "atomic",
        "groups": [{"name": "test", "args": ["type", 1]}],
    }
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 100,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


@pytest.fixture
def parameters_md_nve() -> AttributeDict:
    """
    Set of parameters for a md calculation using the nve integration

    :return: parameters for a md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        "property/atom": [
            {
                "type": [
                    {"keyword": " ", "value": "fx"},
                    {"keyword": " ", "value": "fy"},
                ],
                "group": "all",
            }
        ],
    }
    parameters.md = {
        "integration": {
            "style": "nve",
        },
        "max_number_steps": 5000,
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 1000,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


@pytest.fixture
def parameters_md_nvt() -> AttributeDict:
    """
    Set of parameters for a md calculation using the nvt integration

    :return: parameters for a md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        "property/atom": [
            {
                "type": [
                    {"keyword": " ", "value": "fx"},
                    {"keyword": " ", "value": "fy"},
                ],
                "group": "all",
            }
        ],
    }
    parameters.md = {
        "integration": {
            "style": "nvt",
            "constraints": {
                "temp": [400, 400, 100],
            },
        },
        "max_number_steps": 5000,
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 1000,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


@pytest.fixture
def parameters_md_npt() -> AttributeDict:
    """
    Set of parameters for a md calculation using the npt integration

    :return: parameters for a md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        "property/atom": [
            {
                "type": [
                    {"keyword": " ", "value": "fx"},
                    {"keyword": " ", "value": "fy"},
                ],
                "group": "all",
            }
        ],
    }
    parameters.md = {
        "integration": {
            "style": "npt",
            "constraints": {
                "temp": [400, 400, 100],
                "iso": [0.0, 0.0, 1000.0],
            },
        },
        "max_number_steps": 5000,
        "velocity": [{"create": {"temp": 300, "seed": 1}, "group": "all"}],
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 1000,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


@pytest.fixture
def parameters_restart_full() -> AttributeDict:
    """Get the parameters when all the restart possibilities are considered

    :return: get the parameters controlling the restart file generation
    :rtype: AttributeDict
    """
    data = AttributeDict()
    data.restart = AttributeDict()
    data.restart.print_final = True
    data.restart.print_intermediate = True
    data.restart.num_steps = 1000
    data.settings = AttributeDict()
    data.settings.store_restart = True
    return data


@pytest.fixture
def parameters_restart_full_no_storage() -> AttributeDict:
    """Get the parameters when one is not storing the restartfile

    :return: get the parameters controlling the restart file generation
    :rtype: AttributeDict
    """
    data = AttributeDict()
    data.restart = AttributeDict()
    data.restart.print_final = True
    data.restart.print_intermediate = True
    data.restart.num_steps = 1000
    return data


@pytest.fixture
def parameters_restart_final() -> AttributeDict:
    """Get the parameters for the restart of only the final file

    :return: get the parameters controlling the restart file generation
    :rtype: AttributeDict
    """
    data = AttributeDict()
    data.restart = AttributeDict()
    data.restart.print_final = True
    data.settings = AttributeDict()
    data.settings.store_restart = True
    return data


@pytest.fixture
def parameters_restart_intermediate() -> AttributeDict:
    """Get the parameters for the restartfile of only the intermediate file

    :return: get the parameters controlling the restart file generation
    :rtype: AttributeDict
    """
    data = AttributeDict()
    data.restart = AttributeDict()
    data.restart.print_intermediate = True
    data.restart.num_steps = 1000
    data.settings = AttributeDict()
    data.settings.store_restart = True
    return data


@pytest.fixture
def generate_structure() -> orm.StructureData:
    """
    Generates the structure for the calculation.

    It will create a bcc structure in a square lattice.

    :return: structure to be used in the calculation
    :rtype: orm.StructureData
    """

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

    structure = orm.StructureData(cell=cell)
    for position, symbol, name in zip(positions, symbols, names):
        if fractional:
            position = np.dot(position, cell).tolist()
        structure.append_atom(position=position, symbols=symbol, name=name)

    return structure


@pytest.fixture
def get_potential_fe_eam() -> LammpsPotentialData:
    """
    Generate the potential to be used in the calculation.

    Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

    :return: potential to do the calculation
    :rtype: LammpsPotentialData
    """

    potential_parameters = {
        "species": ["Fe"],
        "atom_style": "atomic",
        "pair_style": "eam/fs",
        "units": "metal",
        "extra_tags": {
            "publication_year": 2018,
            "developer": ["Ronald E. Miller"],
            "title": "EAM potential (LAMMPS cubic hermite tabulation) for Fe developed by Mendelev et al. (2003) v000",
            "content_origin": "NIST IPRP: https: // www.ctcms.nist.gov/potentials/Fe.html",
            "content_other_locations": None,
            "data_method": "unknown",
            "description": """This Fe EAM potential parameter file is from the NIST repository,
            \"Fe_2.eam.fs\" as of the March 9, 2009 update.
            It is similar to \"Fe_mm.eam.fs\" in the LAMMPS distribution dated 2007-06-11,
            but gives different results for very small interatomic distances
            (The LAMMPS potential is in fact the deprecated potential referred to in the March 9,
            2009 update on the NIST repository).
            The file header includes a note from the NIST contributor:
            \"The potential was taken from v9_4_bcc (in C:\\SIMULATION.MD\\Fe\\Results\\ab_initio+Interstitials)\"
            """,
            "disclaimer": """According to the developer Giovanni Bonny
            (as reported by the NIST IPRP), this potential was not stiffened and cannot
            be used in its present form for collision cascades.
            """,
            "properties": None,
            "source_citations": [
                {
                    "abstract": None,
                    "author": "Mendelev, MI and Han, S and Srolovitz, DJ and Ackland, GJ and Sun, DY and Asta, M",
                    "doi": "10.1080/14786430310001613264",
                    "journal": "{Phil. Mag.}",
                    "number": "{35}",
                    "pages": "{3977-3994}",
                    "recordkey": "MO_546673549085_000a",
                    "recordprimary": "recordprimary",
                    "recordtype": "article",
                    "title": "{Development of new interatomic potentials appropriate for crystalline and liquid iron}",
                    "volume": "{83}",
                    "year": "{2003}",
                }
            ],
        },
    }

    source = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "tests/input_files/potentials/Fe_mm.eam.fs",
    )

    potential = LammpsPotentialData.get_or_create(
        source=source,
        **potential_parameters,
    )

    return potential


@pytest.fixture(scope="function")
def db_test_app(aiida_profile, pytestconfig):
    """Clear the database after each test."""
    exec_name = pytestconfig.getoption("lammps_exec") or "lammps"
    executables = {
        "lammps.base": exec_name,
    }

    test_workdir = get_work_directory(pytestconfig)
    work_directory = test_workdir if test_workdir else tempfile.mkdtemp()

    yield AiidaTestApp(work_directory, executables, environment=aiida_profile)
    aiida_profile.clear_profile()

    if not test_workdir:
        shutil.rmtree(work_directory)


@pytest.fixture
def generate_remote_data():
    """Return a `RemoteData` node."""

    def _generate_remote_data(computer, remote_path, entry_point_name=None):
        """Return a `KpointsData` with a mesh of npoints in each direction."""

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        remote = orm.RemoteData(remote_path=remote_path)
        remote.computer = computer

        if entry_point_name is not None:
            creator = orm.CalcJobNode(computer=computer, process_type=entry_point)
            creator.set_option(
                "resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1}
            )
            remote.base.links.add_incoming(
                creator, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            creator.store()

        return remote

    return _generate_remote_data


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
    ) -> tuple[pathlib.Path, CalcInfo] | Process:
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
def generate_calc_job_node(fixture_localhost):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def flatten_inputs(inputs, prefix=""):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + "__"))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name="base",
        computer=None,
        test_name=None,
        inputs=None,
        attributes=None,
        retrieve_temporary=None,
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder.
        :param inputs: any optional nodes to add as input links to the current CalcJobNode
        :param attributes: any optional attributes to set on the node
        :param retrieve_temporary: optional tuple of an absolute filepath of a temporary directory and a list of
            filenames that should be written to this directory, which will serve as the `retrieved_temporary_folder`.
            For now this only works with top-level files and does not support files nested in directories.
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """

        if computer is None:
            computer = fixture_localhost

        filepath_folder = None

        if test_name is not None:
            os.path.dirname(os.path.abspath(__file__))
            filepath_folder = os.path.join(TEST_DIR, test_name)
        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set("input_filename", "input.in")
        node.base.attributes.set("output_filename", "lammps.out")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("max_wallclock_seconds", 1800)
        node.set_metadata_inputs(
            LammpsBaseCalculation._DEFAULT_VARIABLES  # pylint: disable=protected-access
        )

        if attributes:
            node.base.attributes.set_many(attributes)

        if inputs:
            metadata = inputs.pop("metadata", {})
            options = metadata.get("options", {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(
                    input_node,
                    link_type=LinkType.INPUT_CALC,
                    link_label=link_label,
                )

        node.store()

        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            for filename in filenames:
                with suppress(FileNotFoundError):
                    # To test the absence of files in the retrieve_temporary folder
                    shutil.copy(
                        os.path.join(filepath_folder, filename),
                        os.path.join(dirpath, filename),
                    )

        if filepath_folder:
            retrieved = orm.FolderData()
            retrieved.base.repository.put_object_from_tree(filepath_folder)
            # Remove files that are supposed to be only present in the retrieved temporary folder
            if retrieve_temporary:
                for filename in filenames:
                    with suppress(OSError):
                        # To test the absence of files in the retrieve_temporary folder
                        retrieved.base.repository.delete_object(filename)

            retrieved.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="retrieved"
            )
            retrieved.store()

            remote_folder = orm.RemoteData(computer=computer, remote_path="/tmp")
            remote_folder.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            remote_folder.store()

        return node

    return _generate_calc_job_node


@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain


@pytest.fixture
def generate_inputs_minimize(
    fixture_code,
    generate_structure,
    get_potential_fe_eam,
    parameters_minimize,
):
    """Generate default inputs for a `LammpsBaseCalculation` doing a minimization calculation."""

    def _generate_inputs_minimize():
        """Generate default inputs for a `LammpsBaseCalculation` doing a minimization calculation."""

        options = AttributeDict()
        options.resources = AttributeDict()
        # Total number of machines used
        options.resources.num_machines = 1
        # Total number of mpi processes
        options.resources.tot_num_mpiprocs = 2
        options.max_wallclock_seconds = 1

        inputs = {
            "code": fixture_code("lammps.base"),
            "structure": generate_structure,
            "parameters": orm.Dict(dict=parameters_minimize),
            "potential": get_potential_fe_eam,
            "metadata": {"options": options},
        }
        return inputs

    return _generate_inputs_minimize


@pytest.fixture
def generate_inputs_md(
    fixture_code, generate_structure, get_potential_fe_eam, parameters_md_nve
):
    """Generate default inputs for a `LammpsBaseCalculation` doing a NVE MD calculation."""

    def _generate_inputs_md():
        """Generate default inputs for a `LammpsBaseCalculation` doing a NVE MD calculation."""
        options = AttributeDict()
        options.resources = AttributeDict()
        options.resources.num_machines = 1
        options.resources.tot_num_mpiprocs = 2
        options.max_wallclock_seconds = 1
        inputs = {
            "code": fixture_code("lammps.base"),
            "structure": generate_structure,
            "parameters": orm.Dict(dict=parameters_md_nve),
            "potential": get_potential_fe_eam,
            "metadata": {"options": options},
        }
        return inputs

    return _generate_inputs_md


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


@pytest.fixture
def generate_singlefile_data():
    """Return a SinglefileData node"""

    def _generate_singlefile_data(
        computer: orm.Computer,
        label: str = "restartfile",
        entry_point_name: str | None = None,
    ):
        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        with open("README.md", mode="rb") as handler:
            single_file = orm.SinglefileData(handler)

        if entry_point_name is not None:
            creator = orm.CalcJobNode(computer=computer, process_type=entry_point)
            creator.set_option(
                "resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1}
            )
            single_file.base.links.add_incoming(
                creator, link_type=LinkType.CREATE, link_label=label
            )
            creator.store()

        return single_file

    return _generate_singlefile_data


@pytest.fixture
def generate_lammps_trajectory():
    """Return a LammpsTrajectory node"""

    def _generate_lammps_trajectory(
        computer: orm.Computer,
        label: str = "trajectory",
        entry_point_name: str | None = None,
    ):
        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        with open(
            os.path.join(TEST_DIR, "input_files", "trajectory.lammpstrj")
        ) as handler:
            trajectory = LammpsTrajectory(handler)

        if entry_point_name is not None:
            creator = orm.CalcJobNode(computer=computer, process_type=entry_point)
            creator.set_option(
                "resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1}
            )
            trajectory.base.links.add_incoming(
                creator, link_type=LinkType.CREATE, link_label=label
            )
            creator.store()

        return trajectory

    return _generate_lammps_trajectory


@pytest.fixture
def generate_lammps_results():
    """Return a Lammps results node"""

    def _generate_lammps_results(
        computer: orm.Computer,
        label: str = "results",
        entry_point_name: str | None = None,
        data: dict | None = None,
    ):
        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        _results = data if data else {"compute_variables": {}}
        results = orm.Dict(_results)
        if entry_point_name is not None:
            creator = orm.CalcJobNode(computer=computer, process_type=entry_point)
            creator.set_option(
                "resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1}
            )
            results.base.links.add_incoming(
                creator, link_type=LinkType.CREATE, link_label=label
            )
            creator.store()
        return results

    return _generate_lammps_results
