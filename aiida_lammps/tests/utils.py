from collections.abc import Mapping
from contextlib import contextmanager
import distutils.spawn
import os
import re
import subprocess
import sys

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


def lammps_version(executable="lammps"):
    """Get the version of lammps.

    we assume `lammps -h` returns e.g. 'LAMMPS (10 Feb 2015)' or
    'Large-scale Atomic/Molecular Massively Parallel Simulator - 5 Jun 2019'
    """
    out_text = subprocess.check_output([executable, "-h"]).decode("utf8")
    match = re.search(r"LAMMPS \((.*)\)", out_text)
    if match:
        return match.group(1)
    regex = re.compile(
        r"^Large-scale Atomic/Molecular Massively Parallel Simulator - (.*)$",
        re.MULTILINE,
    )
    match = re.search(regex, out_text)
    if match:
        return match.group(1).strip()

    raise IOError("Could not find version from `{} -h`".format(executable))


def get_path_to_executable(executable):
    """Get path to local executable.

    :param executable: Name of executable in the $PATH variable
    :type executable: str

    :return: path to executable
    :rtype: str

    """
    path = None

    # issue with distutils finding scripts within the python path
    # (i.e. those created by pip install)
    script_path = os.path.join(os.path.dirname(sys.executable), executable)
    if os.path.exists(script_path):
        path = script_path
    if path is None:
        path = distutils.spawn.find_executable(executable)
    if path is None:
        raise ValueError("{} executable not found in PATH.".format(executable))

    return os.path.abspath(path)


def get_or_create_local_computer(work_directory, name="localhost"):
    """Retrieve or setup a local computer

    Parameters
    ----------
    work_directory : str
        path to a local directory for running computations in
    name : str
        name of the computer

    Returns
    -------
    aiida.orm.computers.Computer

    """
    from aiida.common import NotExistent
    from aiida.orm import Computer

    try:
        computer = Computer.objects.get(label=name)
    except NotExistent:
        computer = Computer(
            label=name,
            hostname="localhost",
            description=("localhost computer, " "set up by aiida_lammps tests"),
            transport_type="local",
            scheduler_type="direct",
            workdir=os.path.abspath(work_directory),
        )
        computer.store()
        computer.configure()

    return computer


def get_or_create_code(entry_point, computer, executable, exec_path=None):
    """Setup code on localhost computer"""
    from aiida.common import NotExistent
    from aiida.orm import Code, Computer

    if isinstance(computer, str):
        computer = Computer.objects.get(label=computer)

    try:
        code = Code.objects.get(
            label="{}-{}-{}".format(entry_point, executable, computer.label)
        )
    except NotExistent:
        if exec_path is None:
            exec_path = get_path_to_executable(executable)
        code = Code(
            input_plugin_name=entry_point, remote_computer_exec=[computer, exec_path]
        )
        code.label = "{}-{}-{}".format(entry_point, executable, computer.label)
        code.store()

    return code


def get_default_metadata(
    max_num_machines=1,
    max_wallclock_seconds=1800,
    with_mpi=False,
    num_mpiprocs_per_machine=1,
):
    """
    Return an instance of the metadata dictionary with the minimally required parameters
    for a CalcJob and set to default values unless overridden

    :param max_num_machines: set the number of nodes, default=1
    :param max_wallclock_seconds: set the maximum number of wallclock seconds, default=1800
    :param with_mpi: whether to run the calculation with MPI enabled
    :param num_mpiprocs_per_machine: set the number of cpus per node, default=1

    :rtype: dict
    """
    return {
        "options": {
            "resources": {
                "num_machines": int(max_num_machines),
                "num_mpiprocs_per_machine": int(num_mpiprocs_per_machine),
            },
            "max_wallclock_seconds": int(max_wallclock_seconds),
            "withmpi": with_mpi,
        }
    }


def recursive_round(ob, dp, apply_lists=False):
    """map a function on to all values of a nested dictionary"""
    if isinstance(ob, Mapping):
        return {k: recursive_round(v, dp, apply_lists) for k, v in ob.items()}
    elif apply_lists and isinstance(ob, (list, tuple)):
        return [recursive_round(v, dp, apply_lists) for v in ob]
    elif isinstance(ob, float):
        return round(ob, dp)
    else:
        return ob


class AiidaTestApp(object):
    def __init__(self, work_directory, executable_map, environment=None):
        """a class providing methods for testing purposes

        Parameters
        ----------
        work_directory : str
            path to a local work directory (used when creating computers)
        executable_map : dict
            mapping of computation entry points to the executable name
        environment : None or aiida.manage.fixtures.FixtureManager
            manager of a temporary AiiDA environment

        """
        self._environment = environment
        self._work_directory = work_directory
        self._executables = executable_map

    @property
    def work_directory(self):
        """return path to the work directory"""
        return self._work_directory

    @property
    def environment(self):
        """return manager of a temporary AiiDA environment"""
        return self._environment

    def get_or_create_computer(self, name="localhost"):
        """Setup localhost computer"""
        return get_or_create_local_computer(self.work_directory, name)

    def get_or_create_code(self, entry_point, computer_name="localhost"):
        """Setup code on localhost computer"""

        computer = self.get_or_create_computer(computer_name)

        try:
            executable = self._executables[entry_point]
        except KeyError:
            raise KeyError(
                "Entry point {} not recognized. Allowed values: {}".format(
                    entry_point, self._executables.keys()
                )
            )

        return get_or_create_code(entry_point, computer, executable)

    @staticmethod
    def get_default_metadata(
        max_num_machines=1, max_wallclock_seconds=1800, with_mpi=False
    ):
        return get_default_metadata(max_num_machines, max_wallclock_seconds, with_mpi)

    @staticmethod
    def get_parser_cls(entry_point_name):
        """load a parser class

        Parameters
        ----------
        entry_point_name : str
            entry point name of the parser class

        Returns
        -------
        aiida.parsers.parser.Parser

        """
        from aiida.plugins import ParserFactory

        return ParserFactory(entry_point_name)

    @staticmethod
    def get_data_node(entry_point_name, **kwargs):
        """load a data node instance

        Parameters
        ----------
        entry_point_name : str
            entry point name of the data node class

        Returns
        -------
        aiida.orm.nodes.data.Data

        """
        from aiida.plugins import DataFactory

        return DataFactory(entry_point_name)(**kwargs)

    @staticmethod
    def get_calc_cls(entry_point_name):
        """load a data node class

        Parameters
        ----------
        entry_point_name : str
            entry point name of the data node class

        """
        from aiida.plugins import CalculationFactory

        return CalculationFactory(entry_point_name)

    def generate_calcjob_node(
        self, entry_point_name, retrieved, computer_name="localhost", attributes=None
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        Parameters
        ----------
        entry_point_name : str
            entry point name of the calculation class
        retrieved : aiida.orm.FolderData
            containing the file(s) to be parsed
        computer_name : str
            used to get or create a ``Computer``, by default 'localhost'
        attributes : None or dict
            any additional attributes to set on the node

        Returns
        -------
        aiida.orm.CalcJobNode
            instance with the `retrieved` node linked as outgoing

        """
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode
        from aiida.plugins.entry_point import format_entry_point_string

        process = self.get_calc_cls(entry_point_name)
        computer = self.get_or_create_computer(computer_name)
        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        node = CalcJobNode(computer=computer, process_type=entry_point)
        node.set_options(
            {
                k: v.default() if callable(v.default) else v.default
                for k, v in process.spec_options.items()
                if v.has_default()
            }
        )
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("max_wallclock_seconds", 1800)

        if attributes:
            node.set_attributes(attributes)

        node.store()

        retrieved.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
        retrieved.store()

        return node

    @contextmanager
    def sandbox_folder(self):
        """AiiDA folder object context.

        Yields
        ------
        aiida.common.folders.SandboxFolder

        """
        from aiida.common.folders import SandboxFolder

        with SandboxFolder() as folder:
            yield folder

    @staticmethod
    def generate_calcinfo(entry_point_name, folder, inputs=None):
        """generate a `CalcInfo` instance for testing calculation jobs.

        A new `CalcJob` process instance is instantiated,
        and `prepare_for_submission` is called to populate the supplied folder,
        with raw inputs.

        Parameters
        ----------
        entry_point_name: str
        folder: aiida.common.folders.Folder
        inputs: dict or None

        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info
