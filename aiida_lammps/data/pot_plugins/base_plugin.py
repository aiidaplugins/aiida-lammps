"""Abstract class for potential plugins."""
import abc


class PotentialAbstract(abc.ABC):
    """Abstract class for potential plugins."""
    def __init__(self, data):
        self.validate_data(data)
        self.data = data

    @abc.abstractmethod
    def validate_data(self, data):
        """Validate the input data."""
        # pylint: disable=unnecessary-pass
        pass

    @abc.abstractmethod
    def get_external_content(self):
        """Return the mapping of external filenames to content.

        Parameters
        ----------
        data : dict

        Returns
        -------
        None or dict

        """
        # pylint: disable=unnecessary-pass
        pass

    @abc.abstractmethod
    def get_input_potential_lines(self, kind_elements=None):
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
        # pylint: disable=unnecessary-pass
        pass

    @property
    def default_units(self):
        """Return the default unit style to use."""
        # pylint: disable=unnecessary-pass
        pass

    @property
    def atom_style(self):
        """Return the atomic style to use."""
        # pylint: disable=unnecessary-pass
        pass

    @property
    def allowed_element_names(self):
        """Return the allowed element names.

        (used in ``pair_coeff`` to map atom types to elements).
        """
        # pylint: disable=unnecessary-pass
        pass
