import abc

import six


@six.add_metaclass(abc.ABCMeta)
class PotentialAbstract:
    """Abstract class for potential plugins."""

    @abc.abstractmethod
    def get_potential_file_content(self, data):
        """Return the content of the input potential file.

        Parameters
        ----------
        data : dict

        Returns
        -------
        None or str

        """
        pass

    @abc.abstractmethod
    def get_input_potential_lines(
        self, data, kind_elements=None, potential_filename="potential.pot"
    ):
        """Return the potential section of the main input file.

        Parameters
        ----------
        data : dict
        kind_elements : None or list[str]
        potential_filename : str

        Returns
        -------
        str

        """
        pass

    @abc.abstractproperty
    def default_units(self):
        """Return the default unit style to use."""
        pass

    @abc.abstractproperty
    def atom_style(self):
        """Return the atomic style to use."""
        pass
