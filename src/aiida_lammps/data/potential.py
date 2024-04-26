"""
Base class for the LAMMPS potentials.

This class allows the storage/recovering of LAMMPS potentials, it allows
one to store any LAMMPS potential as a :py:class:`~aiida.orm.nodes.data.singlefile.SinglefileData` object, which can
then be either written to the LAMMPS input script and/or to a file, where
it can be easily read by LAMMPS. This distinction depends on the assigned
``pair_style`` which require different ``pair_coeff`` entries in the input
file.

Based on the
`pseudo <https://github.com/aiidateam/aiida-pseudo/blob/master/aiida_pseudo/data/pseudo/pseudo.py>`_
class written by Sebastiaan Huber.

The potentials are also tagged by following the KIM-API
`schema <https://openkim.org/doc/schema/kimspec/>`_, as to make them more easy
to track and as compatible as possible to the KIM schema.
"""
import datetime

# pylint: disable=arguments-differ, too-many-public-methods
import io
import json
import os
import pathlib
from typing import Any, BinaryIO, ClassVar, Optional, Union
import warnings

from aiida import orm, plugins
from aiida.common.constants import elements
from aiida.common.exceptions import StoringNotAllowed
from aiida.common.files import md5_from_filelike


def _validate_string(data: str) -> str:
    """
    Validate if the given data is a string

    :param data: data to be validated
    :type data: str
    :raises TypeError: Raised if the data is not of the correct type
    :return: validated string
    :rtype: str
    """
    if not isinstance(data, str):
        raise TypeError(f'"{data}" is not of type str')
    return data


def _validate_string_list(data: Union[str, list[str]]) -> list[str]:
    """
    Validate the a list of strings

    :param data: string or list of strings
    :type data: Union[str, List[str]]
    :raises TypeError: raise if the data is not of type str or list
    :raises TypeError: raise if an entry in the list is not fo type str
    :return: the data as a list of strings
    :rtype: List[str]
    """
    if not isinstance(data, (list, str)):
        raise TypeError(f'"{data}" is not of type str or list')
    if isinstance(data, list) and not all(isinstance(entry, str) for entry in data):
        raise TypeError(f'Not all entries in "{data}" are of type str')
    if isinstance(data, str):
        data = [data]
    return data


def _validate_datetime(data: Union[str, int, float, datetime.datetime]) -> int:
    """
    Validate and transform dates into integers

    :param data: representation of a year
    :type data: Union[str,int,float, datetime.datetime]
    :raises TypeError: raise if the data is not of type str, int, float or datetime.datetime
    :return: integer representing a year
    :rtype: int
    """
    if not isinstance(data, (int, float, str, datetime.datetime)):
        raise TypeError(f'"{data}" is not of type (str,int,float,datetime.datetime)')
    if isinstance(data, (float, str)):
        data = int(str(data).strip())
    if isinstance(data, datetime.datetime):
        data = data.year
    return data


def _validate_sources(data: Union[dict, list[dict]]) -> list[dict]:
    """
    Validate the sources for the potential.

    This checks whether the entry is a dictionary that can be used to describe
    the citation for a potential.

    :param data: citation data for a potential
    :type data: Union[dict, List[dict]]
    :raises TypeError: raises if the data is not a dict or list of dicts
    :raises TypeError: raises if not all the entries in the list are dicts
    :return: list of references for a potential
    :rtype: List[dict]
    """

    def _validate_single_source(source: dict) -> dict:
        """
        Validate a single potential citation data

        This will check if certain keys exists and add them to the citation data

        :param source: citation data for a potential
        :type source: dict
        :return: validated potential data
        :rtype: dict
        """
        _required_keys = ["author", "journal", "title", "volume", "year"]

        for key in _required_keys:
            if key not in source:
                source[key] = None
                warnings.warn(
                    f'The required key "{key}" is not found, setting its value to None'
                )
        return source

    if not isinstance(data, (dict, list)):
        raise TypeError("The data is not of type dict or list")
    if isinstance(data, list) and not all(isinstance(entry, dict) for entry in data):
        raise TypeError(f'Not all entries in "{data}" are not of type dict')
    if isinstance(data, dict):
        data = _validate_single_source(data)
        data = [data]
    if isinstance(data, list):
        for index, _source in enumerate(data):
            data[index] = _validate_single_source(_source)
    return data


class LammpsPotentialData(orm.SinglefileData):
    """
    Base class for the LAMMPS potentials.

    This class allows the storage/recovering of LAMMPS potentials, it allows
    one to store any LAMMPS potential as a :py:class:`~aiida.orm.nodes.data.singlefile.SinglefileData` object, which can
    then be either written to the LAMMPS input script and/or to a file, where
    it can be easily read by LAMMPS. This distinction depends on the assigned
    ``pair_style`` which require different ``pair_coeff`` entries in the input
    file.

    Based on the
    `pseudo <https://github.com/aiidateam/aiida-pseudo/blob/master/aiida_pseudo/data/pseudo/pseudo.py>`_
    class written by Sebaastian Huber.

    The potentials are also tagged by following the KIM-API
    `schema <https://openkim.org/doc/schema/kimspec/>`_, as to make them more easy
    to track and as compatible as possible to the KIM schema.
    """  # pylint: disable=line-too-long

    # pylint: disable=too-many-arguments, too-many-ancestors
    _key_element = "element"
    _key_md5 = "md5"

    _schema_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lammps_potentials.json",
    )

    _extra_keys: ClassVar[dict[str, Any]] = {
        "title": {"validator": _validate_string},
        "developer": {"validator": _validate_string_list},
        "publication_year": {"validator": _validate_datetime},
        "content_origin": {"validator": _validate_string},
        "content_other_locations": {"validator": _validate_string_list},
        "data_method": {
            "values": ["experiment", "computation", "unknown"],
            "validator": _validate_string,
        },
        "description": {"validator": _validate_string},
        "disclaimer": {"validator": _validate_string},
        "generation_method": {"validator": _validate_string},
        "properties": {"validator": _validate_string_list},
        "source_citations": {"validator": _validate_sources},
    }

    with open(_schema_file, encoding="utf8") as handler:
        _defaults = json.load(handler)
        default_potential_info = _defaults["pair_style"]
        default_atom_style_info = _defaults["atom_style"]

    @classmethod
    def get_or_create(
        cls,
        source: Union[str, pathlib.Path, BinaryIO],
        filename: Optional[str] = None,
        pair_style: Optional[str] = None,
        species: Optional[list] = None,
        atom_style: Optional[str] = None,
        units: Optional[str] = None,
        extra_tags: Optional[dict] = None,
    ):
        """
        Get lammps potential data node from database or create a new one.

        This will check if there is a potential data node with matching md5
        checksum and use that or create a new one if not existent.

        :param source: the source potential content, either a binary stream,
            or a ``str`` or ``Path`` to the path of the file on disk,
            which can be relative or absolute.
        :type source: Union[str, pathlib.Path, BinaryIO]
        :param filename: optional explicit filename to give to the file stored in the repository.
        :type filename: str
        :param pair_style: Type of potential according to LAMMPS
        :type pair_style: str
        :param species: Species that can be used for this potential.
        :type species: list
        :param atom_style: Type of treatment of the atoms according to LAMMPS.
        :type atom_style: str
        :param units: Default units to be used with this potential.
        :type units:  str
        :param extra_tags: Dictionary with extra information to tag the
            potential, based on the KIM schema.
        :type extra_tags: dict
        :return: instance of ``LammpsPotentialData``, stored if taken from
            database, unstored otherwise.
        :raises TypeError: if the source is not a ``str``, ``pathlib.Path``
            instance or binary stream.
        :raises FileNotFoundError: if the source is a filepath but does not exist.
        """
        # pylint: disable=too-many-arguments
        source = cls.prepare_source(source)

        query = orm.QueryBuilder()
        query.append(
            cls,
            subclassing=False,
            filters={f"attributes.{cls._key_md5}": md5_from_filelike(source)},
        )

        existing = query.first()

        if existing:
            potential = existing[0]
        else:
            cls.pair_style = pair_style
            cls.species = species
            cls.atom_style = atom_style
            cls.units = units
            cls.extra_tags = extra_tags
            source.seek(0)
            potential = cls(source, filename)

        return potential

    @classmethod
    def get_entry_point_name(cls):
        """Return the entry point name associated with this data class.
        :return: the entry point name.
        """
        _, entry_point = plugins.entry_point.get_entry_point_from_class(
            cls.__module__,
            cls.__name__,
        )
        return entry_point.name

    @staticmethod
    def is_readable_byte_stream(stream) -> bool:
        """Return if object is a readable filelike object in binary mode or stream of bytes.

        :param stream: the object to analyze.
        :returns: True if ``stream`` appears to be a readable filelike object
            in binary mode, False otherwise.
        """
        return isinstance(stream, io.BytesIO) or (
            hasattr(stream, "read") and hasattr(stream, "mode") and "b" in stream.mode
        )

    @classmethod
    def prepare_source(cls, source: Union[str, pathlib.Path, BinaryIO]) -> BinaryIO:
        """Validate the ``source`` representing a file on disk or a byte stream.

        .. note:: if the ``source`` is a valid file on disk, its content is
            read and returned as a stream of bytes.

        :raises TypeError: if the source is not a ``str``, ``pathlib.Path``
            instance or binary stream.
        :raises FileNotFoundError: if the source is a filepath but does not exist.
        """
        if not isinstance(
            source, (str, pathlib.Path)
        ) and not cls.is_readable_byte_stream(source):
            raise TypeError(
                "`source` should be a `str` or `pathlib.Path` filepath on "
                f"disk or a stream of bytes, got: {source}"
            )

        if isinstance(source, (str, pathlib.Path)):
            filename = pathlib.Path(source).name
            with open(source, "rb") as handle:
                source = io.BytesIO(handle.read())
                source.name = filename

        return source

    def validate_md5(self, md5: str):
        """Validate that the md5 checksum matches that of the currently stored file.

        :param value: the md5 checksum.
        :raises ValueError: if the md5 does not match that of the currently stored file.
        """
        with self.open(mode="rb") as handle:
            md5_file = md5_from_filelike(handle)
            if md5 != md5_file:
                raise ValueError(
                    f"md5 does not match that of stored file: {md5} != {md5_file}"
                )

    def validate_pair_style(self, pair_style: str):
        """
        Validate that the given `pair_style` is a valid lammps `pair_style`

        Takes the given `pair_style` and compares it with those supported by lammps
        and see if there it a match.

        :param pair_style: Name of the LAMMPS potential `pair_style`
        :type pair_style: str
        :raises TypeError: If the `pair_style` is None.
        :raises KeyError: If the `pair_style` is not supported by LAMMPS.
        """
        if pair_style is None:
            raise TypeError("The pair_style of the potential must be provided.")
        if pair_style not in self.default_potential_info:
            raise KeyError(f'The pair_style "{pair_style}" is not valid')
        self.base.attributes.set("pair_style", pair_style)

    def validate_species(self, species: list):
        """
        Validate that the given species are actual atomic species.

        This checks that each of the entries in the species list are actual
        elements.

        :param species: list of atomic species for this potential
        :type species: list
        :raises TypeError: If the list of species is not provided
        """
        if species is None:
            raise TypeError("The species for this potential must be provided.")
        for _specie in species:
            self.validate_element(_specie)
        self.base.attributes.set("species", species)

    def validate_atom_style(self, atom_style: str, pair_style: str):
        """
        Validate that the given `atom_style` is a proper LAMMPS `atom_style`

        The idea is to check if the given `atom_style` is supported by LAMMPS
        if no `atom_style` is given a default one is assigned.

        :param atom_style: Name of the given `atom_style`
        :type atom_style: str
        :param pair_style: Name of the current `pair_style`
        :type pair_style: str
        :raises ValueError: If the `atom_style` is not supported by LAMMPS
        """
        if atom_style is None:
            atom_style = self.default_potential_info[pair_style]["atom_style"]
        if atom_style not in self.default_atom_style_info:
            raise ValueError(f'The atom_style "{atom_style}" is not valid')
        self.base.attributes.set("atom_style", atom_style)

    @classmethod
    def validate_element(cls, element: str):
        """Validate the given element symbol.

        :param element: the symbol of the element following the IUPAC naming standard.
        :raises ValueError: if the element symbol is invalid.
        """
        if element not in [values["symbol"] for values in elements.values()]:
            raise ValueError(f"`{element}` is not a valid element.")

    def validate_units(self, units: str, pair_style: str):
        """
        Validate the LAMMPS units.

        Checks that the provided units for the potential are LAMMPS compatible,
        if no units are given default values are used instead.

        :param units: Name of the given units for the calculation.
        :type units: str
        :param pair_style: Name of the used pair_style
        :type pair_style: str
        :raises ValueError: If the `units` are not LAMMPS compatible.
        """
        if units is None:
            units = self.default_potential_info[pair_style]["units"]
        if units not in [
            "si",
            "lj",
            "real",
            "metal",
            "cgs",
            "electron",
            "micro",
            "nano",
        ]:
            raise ValueError(f'The units "{units}" is not valid')
        self.base.attributes.set("default_units", units)

    def validate_extra_tags(self, extra_tags: dict):
        """
        Validate the dictionary with the extra tags for the potential.

        It will take the given dictionary and check that the keys provided
        correspond to the ones accepted by the class. It will also check that
        the type of the entries are of the appropriate python type. If the
        entries can take only a subset of values they are checked against them.

        :param extra_tags: dictionary with the extra tags that can be used to tag the potential
        :type extra_tags: dict
        :raises TypeError: If the type of the entry does not matches the expected
        :raises ValueError: If the value of the entry does not match the possible values
        """
        for key, value in self._extra_keys.items():
            _value = extra_tags.get(key, None)
            _accepted_values = value.get("values", None)
            _validator = value.get("validator", None)
            if _value is not None and _validator is not None:
                _value = _validator(_value)
            if (
                _accepted_values is not None
                and _value is not None
                and _value not in _accepted_values
            ):
                raise ValueError(
                    f'The value "{_value}" not in the "{_accepted_values}"'
                )

            self.base.attributes.set(key, _value)

    def set_file(
        self,
        source: Union[str, pathlib.Path, BinaryIO],
        filename: Optional[str] = None,
        pair_style: Optional[str] = None,
        species: Optional[list] = None,
        atom_style: Optional[str] = None,
        units: Optional[str] = None,
        extra_tags: Optional[dict] = None,
        **kwargs,
    ):
        """Set the file content.

        .. note:: this method will first analyze the type of the ``source``
            and if it is a filepath will convert it to a binary stream of the
            content located at that filepath, which is then passed on to the
            superclass. This needs to be done first, because it will properly
            set the file and filename attributes that are expected by other
            methods. Straight after the superclass call, the source seeker
            needs to be reset to zero if it needs to be read again, because the
            superclass most likely will have read the stream to the end.
            Finally it is important that the ``prepare_source`` is called here
            before the superclass invocation, because this way the conversion
            from filepath to byte stream will be performed only once.
            Otherwise, each subclass would perform the conversion over and over again.

        :param source: the source lammps potential content, either a binary
            stream, or a ``str`` or ``Path`` to the path of the file on disk,
            which can be relative or absolute.
        :type source: Union[str, pathlib.Path, BinaryIO]
        :param filename: optional explicit filename to give to the file stored in the repository.
        :type filename: str
        :param pair_style: Type of potential according to LAMMPS
        :type pair_style: str
        :param species: Species that can be used for this potential.
        :type species: list
        :param atom_style: Type of treatment of the atoms according to LAMMPS.
        :type atom_style: str
        :param units: Default units to be used with this potential.
        :type units:  str
        :param extra_tags: Dictionary with extra information to tag the
            potential, based on the KIM schema.
        :type extra_tags: dict

        :raises TypeError: if the source is not a ``str``, ``pathlib.Path``
            instance or binary stream.
        :raises FileNotFoundError: if the source is a filepath but does not exist.
        """
        # pylint: disable=too-many-arguments

        source = self.prepare_source(source)

        if self.pair_style is not None and pair_style is None:
            pair_style = self.pair_style
        if self.species is not None and species is None:
            species = self.species
        if self.atom_style is not None and atom_style is None:
            atom_style = self.atom_style
        if self.units is not None and units is None:
            units = self.units
        if self.extra_tags is not None and extra_tags is None:
            extra_tags = self.extra_tags

        self.validate_pair_style(pair_style=pair_style)
        self.validate_species(species=species)
        self.validate_atom_style(atom_style=atom_style, pair_style=pair_style)
        self.validate_units(units=units, pair_style=pair_style)

        if extra_tags is None:
            extra_tags = {}
        self.validate_extra_tags(extra_tags=extra_tags)

        super().set_file(source, filename, **kwargs)
        source.seek(0)
        self.md5 = md5_from_filelike(source)

    def store(self, **kwargs):
        """Store the node verifying first that all required attributes are set.
        :raises :py:exc:`~aiida.common.StoringNotAllowed`: if no valid element has been defined.
        """

        try:
            self.validate_md5(self.md5)
        except ValueError as exception:
            raise StoringNotAllowed(exception) from exception

        return super().store(**kwargs)

    @property
    def atom_style(self) -> str:
        """
        Return the default `atomic_style` of this potential.
        :return: the default `atomic_style` of this potential
        :rtype: str
        """
        return self.base.attributes.get("atom_style")

    @property
    def pair_style(self) -> str:
        """
        Return the `pair_style` of this potential according to the LAMMPS notation
        :return: the `pair_style` of the potential
        :rtype: str
        """
        return self.base.attributes.get("pair_style")

    @property
    def species(self) -> list:
        """Return the list of species which this potential can be used for.
        :return: The list of chemical species which are contained in this potential.
        :rtype: list
        """
        return self.base.attributes.get("species")

    @property
    def default_units(self) -> str:
        """
        Return the default units associated with this potential.
        :return: the default units associated with this potential
        :rtype: str
        """
        return self.base.attributes.get("default_units")

    @property
    def content_origin(self) -> str:
        """
        Return the place where this potential information can be found.

        As based in the KIM schema. A description and/or web address to the
        online source where the material was obtained.
        Possible examples include 'Original content',
        'Obtained from developer', 'Included in LAMMPS',
        a link to the relevant NIST IPR page, or the URL/ID/Access Date of a
        Materials Project entry.

        :return: the place where this potential information can be found.
        :rtype: str
        """
        return self.base.attributes.get("content_origin")

    @property
    def content_other_locations(self) -> Union[str, list]:
        """
        Return other locations where the potential can be found.

        As based in the KIM schema. A description of and/or web address(es)
        to other location(s) where the content is available.

        :return: other locations where the potential can be found.
        :rtype: Union[str, list]
        """
        return self.base.attributes.get("content_other_locations")

    @property
    def data_method(self) -> str:
        """
        Return the data method used to generate the potential.

        As based in the KIM schema. The method used to generate an instance
        of Reference Data.
        Must be one of: experiment, computation, or unknown

        :return: data_method used to generate the potential
        :rtype: str
        """
        return self.base.attributes.get("data_method")

    @property
    def description(self) -> str:
        """
        Return a description of the potential.

        As based in the KIM schema. A short description describing its key
        features including for example: type of model
        (pair potential, 3-body potential, EAM, etc.), modeled elements
        (Ac, Ag, …, Zr), intended purpose, origin, and so on.

        :return: description of the potential
        :rtype: str
        """
        return self.base.attributes.get("description")

    @property
    def developer(self) -> Union[str, list]:
        """
        Return the developer information of this potential.

        As based in the KIM schema. An array of strings, each of which is a KIM
        user uuid corresponding to a "developer" of the item.
        A developer of an item is someone who participated in creating the core
        intellectual content of the digital object, e.g. the functional form
        of an interatomic model or a specific parameter set for it.

        :return: developer information of this potential
        :rtype: Union[str, list]
        """
        return self.base.attributes.get("developer")

    @property
    def disclaimer(self) -> str:
        """
        Return a disclaimer regarding the usage of the potential.

        As based in the KIM schema. A short statement of applicability which
        will accompany any results computed using it. A developer can use the
        disclaimer to inform users of the intended use of this KIM Item.

        :return: disclaimer regarding the usage of this potential
        :rtype: str
        """
        return self.base.attributes.get("disclaimer")

    @property
    def properties(self) -> Union[str, list]:
        """
        Return the properties for which this potential was devised.

        As based in the KIM schema. A list of properties reported by a KIM Item.

        :return: properties fow which this potential was devised.
        :rtype: Union[str, list]
        """
        return self.base.attributes.get("properties")

    @property
    def publication_year(self) -> Union[str, datetime.datetime, int]:
        """
        Return the year of publication of this potential.

        As based in the KIM schema. Year this item was published on openkim.org.

        :return: year of publication of this potential
        :rtype: Union[str, datetime.datetime, int]
        """
        return self.base.attributes.get("publication_year")

    @property
    def source_citations(self) -> Union[str, list]:
        """
        Return the citation where the potential was originally published.

        As based in the KIM schema. An array of BibTeX-style EDN dictionaries
        corresponding to primary published work(s) describing the KIM Item.

        :return: the citation where the potential was originally published.
        :rtype: Union[str, list]
        """
        return self.base.attributes.get("source_citations")

    @property
    def title(self) -> str:
        """
        Return the title of the potential.

        As based in the KIM schema. Used when displaying a KIM Item on
        openkim.org, as well as autogenerating its citation.
        The title should not include an ending period.

        :return: the title of the potential
        :rtype: str
        """
        return self.base.attributes.get("title")

    @property
    def md5(self) -> Optional[int]:
        """Return the md5.
        :return: the md5 of the stored file.
        """
        return self.base.attributes.get(self._key_md5, None)

    @property
    def generation_method(self) -> str:
        """
        Return the generation method of the potential.

        In here one can describe how the potential itself was generated, if it
        was done via ML, fitting via specific codes, analytical fitting, etc.

        :return: the generation method of the potential
        :rtype: str
        """
        return self.base.attributes.get("generation_method")

    @md5.setter
    def md5(self, value: str):
        """Set the md5.
        :param value: the md5 checksum.
        :raises ValueError: if the md5 does not match that of the currently stored file.
        """
        self.validate_md5(value)
        self.base.attributes.set(self._key_md5, value)
