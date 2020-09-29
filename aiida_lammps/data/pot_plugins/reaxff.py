from aiida_lammps.common.reaxff_convert import write_lammps_format
from aiida_lammps.data.pot_plugins.base_plugin import PotentialAbstract
from aiida_lammps.validation import validate_against_schema


class Reaxff(PotentialAbstract):
    """Class for creation of ReaxFF potential inputs.

    NOTE: to use c_reax[1] - c_reax[14],
    c_reax[1] must be added to ``thermo_style custom``, to trigger the compute

    The array values correspond to:

    - eb = bond energy
    - ea = atom energy
    - elp = lone-pair energy
    - emol = molecule energy (always 0.0)
    - ev = valence angle energy
    - epen = double-bond valence angle penalty
    - ecoa = valence angle conjugation energy
    - ehb = hydrogen bond energy
    - et = torsion energy
    - eco = conjugation energy
    - ew = van der Waals energy
    - ep = Coulomb energy
    - efi = electric field energy (always 0.0)
    - eqeq = charge equilibration energy

    """

    potential_fname = "potential.pot"
    control_fname = "potential.control"

    def validate_data(self, data):
        """Validate the input data."""
        validate_against_schema(data, "reaxff.schema.json")

    def get_potential_file_content(self):
        if "file_contents" in self.data:
            content = ""
            for line in self.data["file_contents"]:
                content += "{}".format(line)
        else:
            content = write_lammps_format(self.data)
        return content

    def get_control_file_content(self):
        control = self.data.get("control", {})
        global_dict = self.data.get("global", {})
        content = []

        tolerances = {
            "hbonddist": "hbond_cutoff",
            "nbrhood_cutoff": "nbrhood_cutoff",
            "anglemin": "thb_cutoff",
            "angleprod": "thb_cutoff_sq",
        }

        for key, name in tolerances.items():
            if key in global_dict:
                content.append("{} {}".format(name, global_dict[key]))

        control_variables = [
            "simulation_name",
            "traj_title",
            "tabulate_long_range",
            "energy_update_freq",
            "write_freq",
            "bond_graph_cutoff",
        ]

        for name in control_variables:
            if name in control:
                content.append("{} {}".format(name, control[name]))

        bool_to_int = {
            "print_atom_info": "atom_info",
            "print_atom_forces": "atom_forces",
            "print_atom_velocities": "atom_velocities",
            "print_bond_info": "bond_info",
            "print_angle_info": "angle_info",
        }

        for key, name in bool_to_int.items():
            if key in control:
                content.append("{} {}".format(name, 1 if control[key] else 0))

        if content:
            return "\n".join(content)
        return None

    def get_external_content(self):

        fmap = {self.potential_fname: self.get_potential_file_content()}
        content = self.get_control_file_content()
        if content:
            fmap[self.control_fname] = content
        return fmap

    def get_input_potential_lines(self):

        control = self.data.get("control", {})

        lammps_input_text = "pair_style reax/c {} ".format(
            self.control_fname if self.get_control_file_content() else "NULL"
        )
        if "safezone" in control:
            lammps_input_text += "safezone {0} ".format(control["safezone"])
        lammps_input_text += "\n"
        lammps_input_text += "pair_coeff      * * {} {{kind_symbols}}\n".format(
            self.potential_fname
        )
        lammps_input_text += "fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
        if control.get("fix_modify_qeq", True):
            # TODO #15 in conda-forge/osx-64::lammps-2019.06.05-py36_openmpi_5,
            # an error is raised: ERROR: Illegal fix_modify command (src/fix.cpp:147)
            # posted question to lammps-users@lists.sourceforge.net
            # 'Using qeq/reax fix_modify energy in recent versions of LAMMPS'
            # lammps_input_text += "fix_modify qeq energy yes\n"
            pass
        lammps_input_text += "compute reax all pair reax/c\n"

        return lammps_input_text

    @property
    def allowed_element_names(self):
        elements = self.data.get("species", None)
        if elements:
            # strip core/shell
            elements = [e.split()[0] for e in elements]
        return elements

    @property
    def atom_style(self):
        return "charge"

    @property
    def default_units(self):
        return "real"
