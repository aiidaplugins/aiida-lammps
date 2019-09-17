import io
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import six

from aiida.orm import Data

from aiida_lammps.common.parse_trajectory import (
    iter_trajectories,
    parse_step,
    create_structure,
)


class LammpsTrajectory(Data):
    """Store a lammps trajectory file.

    Each trajectory step is stored as a separate file, within a compressed zip folder.
    This reduces storage space, and allows for fast access to each step.

    """

    _zip_prefix = "step-"
    _traj_filename = "trajectory.zip"
    _timestep_filename = "timesteps.txt"
    _compression_method = ZIP_DEFLATED

    def __init__(self, fileobj=None, aliases=None, **kwargs):
        super(LammpsTrajectory, self).__init__(**kwargs)

        if fileobj is not None:
            if isinstance(fileobj, six.string_types):
                with io.open(fileobj) as handle:
                    self.set_from_fileobj(handle, aliases)
            else:
                self.set_from_fileobj(fileobj, aliases)

    def _validate(self):
        from aiida.common.exceptions import ValidationError

        super(LammpsTrajectory, self)._validate()
        if self.get_attribute("number_steps", None) is None:
            raise ValidationError("trajectory has not yet been set")

    def set_from_fileobj(self, fileobj, aliases=None):

        time_steps = []
        elements = set()
        field_names = None
        number_atoms = None

        self.reset_attributes({})

        if not (aliases is None or isinstance(aliases, dict)):
            raise ValueError("aliases must be None or dict")

        # Write the zip to a temporary file, and then add it to the node repository
        with tempfile.NamedTemporaryFile() as temp_handle:
            with ZipFile(temp_handle, "w", self._compression_method) as zip_file:
                for step_id, traj_step in enumerate(iter_trajectories(fileobj)):

                    # extract data to store in attributes
                    time_steps.append(traj_step.timestep)
                    if number_atoms is None:
                        number_atoms = traj_step.natoms
                    elif traj_step.natoms != number_atoms:
                        raise IOError(
                            "step {} contains different number of atoms: {}".format(
                                step_id, traj_step.natoms
                            )
                        )
                    if field_names is None:
                        field_names = list(traj_step.atom_fields.keys())
                    elif field_names != list(traj_step.atom_fields.keys()):
                        raise IOError(
                            "step {} contains different field names: {}".format(
                                step_id, list(traj_step.atom_fields.keys())
                            )
                        )
                    if "element" in traj_step.atom_fields:
                        elements.update(traj_step.atom_fields["element"])

                    # save content
                    content = "\n".join(traj_step.lines)
                    zip_name = "{}{}".format(self._zip_prefix, step_id)
                    zip_file.writestr(zip_name, content)

            if not time_steps:
                raise IOError("The trajectory file does not contain any timesteps")

            # Flush and rewind the temporary handle,
            # otherwise the command to store it in the repo will write an empty file
            temp_handle.flush()
            temp_handle.seek(0)

            self.put_object_from_filelike(
                temp_handle, self._traj_filename, mode="wb", encoding=None
            )

        with self.open(self._timestep_filename, "w") as handle:
            handle.write(six.ensure_text(" ".join([str(t) for t in time_steps])))

        self.set_attribute("number_steps", len(time_steps))
        self.set_attribute("number_atoms", number_atoms)
        self.set_attribute("field_names", list(sorted(field_names)))
        self.set_attribute("traj_filename", self._traj_filename)
        self.set_attribute("timestep_filename", self._timestep_filename)
        self.set_attribute("zip_prefix", self._zip_prefix)
        self.set_attribute("compression_method", self._compression_method)
        self.set_attribute("aliases", aliases)
        self.set_attribute("elements", list(sorted(elements)))

    @property
    def number_steps(self):
        return self.get_attribute("number_steps")

    @property
    def time_steps(self):
        with self.open(self.get_attribute("timestep_filename"), "r") as handle:
            output = [int(i) for i in handle.readline().split()]
        return output

    @property
    def number_atoms(self):
        return self.get_attribute("number_atoms")

    @property
    def field_names(self):
        return self.get_attribute("field_names")

    @property
    def aliases(self):
        return self.get_attribute("aliases")

    def get_step_string(self, step_idx):
        step_idx = list(range(self.number_steps))[step_idx]
        zip_name = "{}{}".format(self.get_attribute("zip_prefix"), step_idx)
        with self.open(self.get_attribute("traj_filename"), mode="rb") as handle:
            with ZipFile(
                handle, "r", self.get_attribute("compression_method")
            ) as zip_file:
                with zip_file.open(zip_name) as step_file:
                    content = step_file.read()
        return six.ensure_text(content)

    def get_step_data(self, step_idx):
        return parse_step(self.get_step_string(step_idx).splitlines())

    def get_step_structure(
        self,
        step_idx,
        symbol_field="element",
        position_fields=("x", "y", "z"),
        original_structure=None,
    ):
        data = self.get_step_data(step_idx)
        return create_structure(
            data,
            symbol_field=symbol_field,
            position_fields=position_fields,
            original_structure=original_structure,
        )
