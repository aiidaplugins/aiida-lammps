import numpy as np

from aiida_lammps.data.pot_plugins.base_plugin import PotentialAbstract


class LennardJones(PotentialAbstract):
    """Class for creation of Lennard-Jones potential inputs."""

    def get_potential_file_content(self, data):
        return None

    def get_input_potential_lines(
        self, data, kind_elements=None, potential_filename="potential.pot"
    ):

        cut = np.max([float(i.split()[2]) for i in data.values()])

        lammps_input_text = "pair_style  lj/cut {}\n".format(cut)

        for key in sorted(data.keys()):
            lammps_input_text += "pair_coeff {}    {}\n".format(key, data[key])
        return lammps_input_text

    @property
    def atom_style(self):
        return "atomic"

    @property
    def default_units(self):
        return "metal"
