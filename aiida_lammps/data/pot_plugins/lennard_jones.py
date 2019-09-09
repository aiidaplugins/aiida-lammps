import numpy as np

from aiida_lammps.data.pot_plugins.base_plugin import PotentialAbstract


class LennardJones(PotentialAbstract):
    """Class for creation of Lennard-Jones potential inputs."""

    def validate_data(self, data):
        """Validate the input data."""
        pass

    def get_external_content(self):
        return None

    def get_input_potential_lines(self):

        cut = np.max([float(i.split()[2]) for i in self.data.values()])

        lammps_input_text = "pair_style  lj/cut {}\n".format(cut)

        # TODO how to map kinds to pair coefficient for lj?

        for key in sorted(self.data.keys()):
            lammps_input_text += "pair_coeff {}    {}\n".format(key, self.data[key])
        return lammps_input_text

    @property
    def allowed_element_names(self):
        return None

    @property
    def atom_style(self):
        return "atomic"

    @property
    def default_units(self):
        return "metal"
