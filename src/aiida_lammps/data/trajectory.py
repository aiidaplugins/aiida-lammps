"""
Data structure for storing LAMMPS trajectories.

The idea is that each of the steps of the simulation are stored as ZIP files
which can then be easily accessed by the user.
"""
# pylint: disable=too-many-ancestors
import io
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from aiida import orm
from aiida.common.exceptions import ValidationError

from aiida_lammps.parsers.parse_raw import (
    create_structure,
    iter_trajectories,
    parse_step,
)


class LammpsTrajectory(orm.Data):
    """Store a lammps trajectory file.

    Each trajectory step is stored as a separate file, within a compressed zip folder.
    This reduces storage space, and allows for fast access to each step.

    """

    _zip_prefix = "step-"
    _trajectory_filename = "trajectory.zip"
    _timestep_filename = "timesteps.txt"
    _compression_method = ZIP_DEFLATED

    def __init__(self, fileobj=None, aliases=None, **kwargs):
        """Store a lammps trajectory file.

        :param fileobj: the file or path to the file, defaults to None
        :type fileobj: str, optional
        :param aliases: mapping of variable names to one or more lammps variables,
            e.g. {"position": ["x", "y", "z"]}, defaults to None
        :type aliases: dict[str, list], optional
        """

        super().__init__(**kwargs)

        if fileobj is not None:
            if isinstance(fileobj, str):
                with open(fileobj) as handle:
                    self.set_from_fileobj(handle.readlines(), aliases)
            else:
                self.set_from_fileobj(fileobj, aliases)

    def _validate(self):
        """Validate that a trajectory has been set, before storing."""

        super()._validate()
        if self.base.attributes.get("number_steps", None) is None:
            raise ValidationError("trajectory has not yet been set")

    def set_from_fileobj(self, fileobj, aliases=None):
        """Store a lammps trajectory file.

        :param fileobj: the file or path to the file
        :type fileobj: str
        :param aliases:  mapping of variable names to one or more lammps variables,
            e.g. {"position": ["x", "y", "z"]}, defaults to None
        :type aliases: dict[str, list], optional
        :raises ValueError: if the aliases are not of the correct type
        :raises IOError: if a given step has more atoms than supposed to
        :raises IOError: if a given step has incompatible field names
        :raises IOError: if the timesteps are not present in the trajectory file
        """

        time_steps = []
        elements = set()
        field_names = None
        number_atoms = None

        self.base.attributes.reset({})

        if not (aliases is None or isinstance(aliases, dict)):
            raise ValueError("aliases must be None or dict")

        # Write the zip to a temporary file, and then add it to the node repository
        with tempfile.NamedTemporaryFile() as temp_handle:
            with ZipFile(
                temp_handle,
                "w",
                self._compression_method,
            ) as zip_file:
                for step_id, trajectory_step in enumerate(iter_trajectories(fileobj)):
                    # extract data to store in attributes
                    time_steps.append(trajectory_step.timestep)
                    if number_atoms is None:
                        number_atoms = trajectory_step.natoms
                    elif trajectory_step.natoms != number_atoms:
                        raise OSError(
                            f"step {step_id} contains different number of"
                            f" atoms: {trajectory_step.natoms}"
                        )
                    if field_names is None:
                        field_names = list(trajectory_step.atom_fields.keys())
                    elif field_names != list(trajectory_step.atom_fields.keys()):
                        raise OSError(
                            f"step {step_id} contains different field names:"
                            f" {list(trajectory_step.atom_fields.keys())}"
                        )
                    if "element" in trajectory_step.atom_fields:
                        elements.update(trajectory_step.atom_fields["element"])

                    # save content
                    content = "\n".join(trajectory_step.lines)
                    zip_name = f"{self._zip_prefix}{step_id}"
                    zip_file.writestr(zip_name, content)

            if not time_steps:
                raise OSError("The trajectory file does not contain any timesteps")

            # Flush and rewind the temporary handle,
            # otherwise the command to store it in the repo will write an empty file
            temp_handle.flush()
            temp_handle.seek(0)

            self.base.repository.put_object_from_filelike(
                temp_handle,
                self._trajectory_filename,
            )

        self.base.repository.put_object_from_filelike(
            io.StringIO(" ".join([str(entry) for entry in time_steps])),
            self._timestep_filename,
        )

        self.base.attributes.set("number_steps", len(time_steps))
        self.base.attributes.set("number_atoms", number_atoms)
        self.base.attributes.set("field_names", sorted(field_names))
        self.base.attributes.set("trajectory_filename", self._trajectory_filename)
        self.base.attributes.set("timestep_filename", self._timestep_filename)
        self.base.attributes.set("zip_prefix", self._zip_prefix)
        self.base.attributes.set("compression_method", self._compression_method)
        self.base.attributes.set("aliases", aliases)
        self.base.attributes.set("elements", sorted(elements))

    @property
    def number_steps(self):
        """Get the number of steps stored in the data.

        :return: number of steps stored in the data
        :rtype: int
        """
        return self.base.attributes.get("number_steps")

    @property
    def time_steps(self):
        """Get the simulation time steps stored in the data.

        :return: time steps stored in the data.
        :rtype: list
        """
        with self.base.repository.open(
            self.base.attributes.get("timestep_filename"), "r"
        ) as handle:
            output = [int(i) for i in handle.readline().split()]
        return output

    @property
    def number_atoms(self):
        """Get the number of atoms present in the simulation box.

        :return: number of atoms in the simulation box
        :rtype: int
        """
        return self.base.attributes.get("number_atoms")

    @property
    def field_names(self):
        """Get the name of the fields as written to file.

        :return: list of field names as written to file.
        :rtype: list
        """
        return self.base.attributes.get("field_names")

    @property
    def aliases(self):
        """Get the mapping of one or more lammps variables.

        :return: mapping of one or more lammps variables.
        :rtype: list
        """
        return self.base.attributes.get("aliases")

    def get_step_string(self, step_idx):
        """Return the content string, for a specific trajectory step."""
        step_idx = list(range(self.number_steps))[step_idx]
        zip_name = f'{self.base.attributes.get("zip_prefix")}{step_idx}'
        with self.base.repository.open(
            self.base.attributes.get("trajectory_filename"),
            mode="rb",
        ) as handle, ZipFile(
            handle,
            "r",
            self.base.attributes.get("compression_method"),
        ) as zip_file, zip_file.open(zip_name, "r") as step_file:
            content = step_file.read()
        return content.decode("utf8")

    def get_step_data(self, step_idx):
        """Return parsed data, for a specific trajectory step."""
        return parse_step(self.get_step_string(step_idx).splitlines())

    def iter_step_strings(self, steps=None):
        """Yield the content string, for each trajectory step."""

        if steps is None:
            steps = range(self.number_steps)
        elif isinstance(steps, int):
            steps = range(0, self.number_steps, steps)

        with self.base.repository.open(
            self.base.attributes.get("trajectory_filename"),
            mode="rb",
        ) as handle, ZipFile(
            handle,
            "r",
            self.base.attributes.get("compression_method"),
        ) as zip_file:
            for step_idx in steps:
                zip_name = f'{self.base.attributes.get("zip_prefix")}{step_idx}'
                with zip_file.open(zip_name) as step_file:
                    content = step_file.read()
                    yield content

    def get_step_structure(
        self,
        step_idx: int,
        symbol_field: str = "element",
        position_fields: tuple = ("x", "y", "z"),
        original_structure: orm.StructureData = None,
    ) -> orm.StructureData:
        """Return a StructureData object, for a specific trajectory step.

        :param step_idx: trajectory step to be looked at
        :type step_idx: int
        :param symbol_field: the variable field denoting the symbol
            for each atom, defaults to 'element'
        :type symbol_field: str, optional
        :param position_fields: tuple, the variable fields denoting
            the x, y, z position for each atom, defaults to ('x', 'y', 'z')
        :type position_fields: tuple, optional
        :param original_structure: a structure that will be used to
            define kinds for each atom, defaults to None
        :type original_structure: orm.StructureData, optional
        :return: structure of the simulation at the given time step
        :rtype: orm.StructureData
        """

        data = self.get_step_data(step_idx)
        return create_structure(
            data,
            symbol_field=symbol_field,
            position_fields=position_fields,
            original_structure=original_structure,
        )

    def write_as_lammps(self, handle, steps=None):
        """Write out the lammps trajectory to file.

        :param handle: a file handle, opened in "wb" mode
        :param steps: a list of steps to write (default to all)
        """
        for string in self.iter_step_strings(steps=steps):
            handle.write(string)
            handle.write(b"\n")
