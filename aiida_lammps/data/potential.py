from hashlib import md5
from io import StringIO

from aiida.orm import Data
from aiida.plugins.entry_point import get_entry_point_names, load_entry_point


class EmpiricalPotential(Data):
    """
    Store the empirical potential data
    """

    entry_name = "lammps.potentials"
    pot_lines_fname = "potential_lines.txt"

    @classmethod
    def list_types(cls):
        return get_entry_point_names(cls.entry_name)

    @classmethod
    def load_type(cls, entry_name):
        return load_entry_point(cls.entry_name, entry_name)

    def __init__(self, type, data=None, **kwargs):
        """Empirical potential data, used to create LAMMPS input files.

        Parameters
        ----------
        type: str
            the type of potential (should map to a `lammps.potential` entry point)
        data: dict
            data required to create the potential file and input lines

        """
        super(EmpiricalPotential, self).__init__(**kwargs)
        self.set_data(type, data)

    def set_data(self, potential_type, data=None):
        """Store the potential type (e.g. Tersoff, EAM, LJ, ..)."""
        if potential_type is None:
            raise ValueError("'potential_type' must be provided")
        if potential_type not in self.list_types():
            raise ValueError(
                "'potential_type' must be in: {}".format(self.list_types())
            )
        pot_class = self.load_type(potential_type)(data or {})

        atom_style = pot_class.atom_style
        default_units = pot_class.default_units
        allowed_element_names = pot_class.allowed_element_names

        external_contents = pot_class.get_external_content() or {}
        pot_lines = pot_class.get_input_potential_lines()

        self.set_attribute("potential_type", potential_type)
        self.set_attribute("atom_style", atom_style)
        self.set_attribute("default_units", default_units)
        self.set_attribute(
            "allowed_element_names",
            sorted(allowed_element_names)
            if allowed_element_names
            else allowed_element_names,
        )

        # store potential section of main input file
        self.set_attribute(
            "md5|input_lines", md5(pot_lines.encode("utf-8")).hexdigest()
        )
        self.put_object_from_filelike(StringIO(pot_lines), self.pot_lines_fname)

        # store external files required by the potential
        external_files = []
        for fname, content in external_contents.items():
            self.set_attribute(
                "md5|{}".format(fname.replace(".", "_")),
                md5(content.encode("utf-8")).hexdigest(),
            )
            self.put_object_from_filelike(StringIO(content), fname)
            external_files.append(fname)
        self.set_attribute("external_files", sorted(external_files))

        # delete any previously stored files that are no longer required
        for fname in self.list_object_names():
            if fname not in external_files + [self.pot_lines_fname]:
                self.delete_object(fname)

    @property
    def potential_type(self):
        """Return lammps atom style."""
        return self.get_attribute("potential_type")

    @property
    def atom_style(self):
        """Return lammps atom style."""
        return self.get_attribute("atom_style")

    @property
    def default_units(self):
        """Return lammps default units."""
        return self.get_attribute("default_units")

    @property
    def allowed_element_names(self):
        """Return available atomic symbols."""
        return self.get_attribute("allowed_element_names")

    def get_input_lines(self, kind_symbols=None):
        """Return the command(s) required to setup the potential.

        The placeholder ``{kind_symbols}`` will be replaced,
        with a list of symbols for each kind in the structure.

        e.g.::

             pair_style      eam
             pair_coeff      * *  {kind_symbols}

        get_input_lines(["S", "Cr"])::

             pair_style      eam
             pair_coeff      * *  S Cr

        """
        content = self.get_object_content(self.pot_lines_fname, "r")
        if kind_symbols:
            content = content.replace("{kind_symbols}", " ".join(kind_symbols))
        return content

    def get_external_files(self):
        """Return the mapping of external filenames to content."""
        fmap = {}
        for fname in self.get_attribute("external_files"):
            fmap[fname] = self.get_object_content(fname, "r")
        return fmap
