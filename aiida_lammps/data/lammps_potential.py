"""
Base class for the LAMMPS potentials.

This class allows the storage/recovering of LAMMPS potentials, it allows
one to store any LAMMPS potential as a ``SinglefileData`` object, which can
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
"""
# pylint: disable=arguments-differ, too-many-public-methods
import io
import os
import pathlib
import typing
import json
import datetime

from aiida import orm
from aiida import plugins
from aiida.common.constants import elements
from aiida.common.exceptions import StoringNotAllowed
from aiida.common.files import md5_from_filelike


class LammpsPotentialData(orm.SinglefileData):  # pylint: disable=too-many-arguments, too-many-ancestors
    """
    Base class for the LAMMPS potentials.

    This class allows the storage/recovering of LAMMPS potentials, it allows
    one to store any LAMMPS potential as a ``SinglefileData`` object, which can
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

    _key_element = 'element'
    _key_md5 = 'md5'

    _schema_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'lammps_potentials.json',
    )

    _extra_keys = {
        'content_origin': {
            'type': str
        },
        'content_other_locations': {
            'type': (str, list)
        },
        'data_method': {
            'type': str,
            'values': ['experiment', 'computation', 'unknown']
        },
        'description': {
            'type': str
        },
        'developer': {
            'type': (str, list)
        },
        'disclaimer': {
            'type': str
        },
        'generation_method': {
            'type': str
        },
        'properties': {
            'type': (str, list)
        },
        'publication_year': {
            'type': (str, datetime.datetime, int)
        },
        'source_citations': {
            'type': (str, list)
        },
        'title': {
            'type': str
        },
    }

    with open(_schema_file, 'r') as handler:
        _defaults = json.load(handler)
        default_potential_info = _defaults['pair_style']
        default_atom_style_info = _defaults['atom_style']

    @classmethod
    def get_or_create(
        cls,
        source: typing.Union[str, pathlib.Path, typing.BinaryIO],
        filename: str = None,
        pair_style: str = None,
        species: list = None,
        atom_style: str = None,
        units: str = None,
        extra_tags: dict = None,
    ):  # pylint: disable=too-many-arguments
        """Get lammps potential data node from database or create a new one.

        This will check if there is a potential data node with matching md5
        checksum and use that or create a new one if not existent.

        :param source: the source potential content, either a binary stream,
            or a ``str`` or ``Path`` to the path of the file on disk,
            which can be relative or absolute.
        :param filename: optional explicit filename to give to the file stored in the repository.
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
        source = cls.prepare_source(source)

        query = orm.QueryBuilder()
        query.append(
            cls,
            subclassing=False,
            filters={f'attributes.{cls._key_md5}': md5_from_filelike(source)},
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
        :param stream: the object to analyse.
        :returns: True if ``stream`` appears to be a readable filelike object
            in binary mode, False otherwise.
        """
        return (isinstance(stream, io.BytesIO)
                or (hasattr(stream, 'read') and hasattr(stream, 'mode')
                    and 'b' in stream.mode))

    @classmethod
    def prepare_source(
        cls, source: typing.Union[str, pathlib.Path, typing.BinaryIO]
    ) -> typing.BinaryIO:
        """Validate the ``source`` representing a file on disk or a byte stream.
        .. note:: if the ``source`` is a valid file on disk, its content is
            read and returned as a stream of bytes.
        :raises TypeError: if the source is not a ``str``, ``pathlib.Path``
        instance or binary stream.
        :raises FileNotFoundError: if the source is a filepath but does not exist.
        """
        if not isinstance(
                source,
            (str, pathlib.Path)) and not cls.is_readable_byte_stream(source):
            raise TypeError(
                '`source` should be a `str` or `pathlib.Path` filepath on ' +
                f'disk or a stream of bytes, got: {source}')

        if isinstance(source, (str, pathlib.Path)):
            filename = pathlib.Path(source).name
            with open(source, 'rb') as handle:
                source = io.BytesIO(handle.read())
                source.name = filename

        return source

    def validate_md5(self, md5: str):
        """Validate that the md5 checksum matches that of the currently stored file.
        :param value: the md5 checksum.
        :raises ValueError: if the md5 does not match that of the currently stored file.
        """
        with self.open(mode='rb') as handle:
            md5_file = md5_from_filelike(handle)
            if md5 != md5_file:
                raise ValueError(
                    f'md5 does not match that of stored file: {md5} != {md5_file}'
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
            raise TypeError(
                'The pair_style of the potential must be provided.')
        if pair_style not in self.default_potential_info.keys():
            raise KeyError(f'The pair_style "{pair_style}" is not valid')
        self.set_attribute('pair_style', pair_style)

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
            raise TypeError('The species for this potential must be provided.')
        for _specie in species:
            self.validate_element(_specie)
        self.set_attribute('species', species)

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
            atom_style = self.default_potential_info[pair_style]['atom_style']
        if atom_style not in self.default_atom_style_info:
            raise ValueError(f'The atom_style "{atom_style}" is not valid')
        self.set_attribute('atom_style', atom_style)

    @classmethod
    def validate_element(cls, element: str):
        """Validate the given element symbol.
        :param element: the symbol of the element following the IUPAC naming standard.
        :raises ValueError: if the element symbol is invalid.
        """
        if element not in [values['symbol'] for values in elements.values()]:
            raise ValueError(f'`{element}` is not a valid element.')

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
            units = self.default_potential_info[pair_style]['units']
        if units not in [
                'si', 'lj', 'real', 'metal', 'cgs', 'electron', 'micro', 'nano'
        ]:
            raise ValueError(f'The units "{units}" is not valid')
        self.set_attribute('default_units', units)

    def validate_extra_tags(self, extra_tags: dict):
        """
        Validate the dictionary with the extra tags for the potential.

        It will take the given dictionary and check that the keys provided
        correspond to the ones accepted by the class. It will also check that
        the type of the entries are of the appropriate python type. If the
        entries can take only a subset of values they are checked against them.
        :param extra_tags: dictionary with the extra tags that can be used to tag the potential
        :type extra_tags: dict
        :raises ValueError: If the type of the entry does not matches the expected
        :raises ValueError: If the value of the entry does not match the possible values
        """
        for key, value in self._extra_keys.items():
            _value = extra_tags.get(key, None)
            _types = value.get('type', None)
            _values = value.get('values', None)
            if _value is not None:
                if not isinstance(_value, _types):
                    raise ValueError(
                        f'Tag "{key}" with value "{_value}" is not of type "{_types}"'
                    )
                if _values is not None and _value not in _values:
                    raise ValueError(
                        f'Tag "{key}" is not in the accepted values "{_values}"'
                    )

    def set_file(
        self,
        source: typing.Union[str, pathlib.Path, typing.BinaryIO],
        filename: str = None,
        pair_style: str = None,
        species: list = None,
        atom_style: str = None,
        units: str = None,
        extra_tags: dict = None,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        """Set the file content.

        .. note:: this method will first analyse the type of the ``source``
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
        :type source: typing.Union[str, pathlib.Path, typing.BinaryIO]
        :param filename: optional explicit filename to give to the file stored in the repository.
        :type filename: str
        :param pair_style: Type of potential according to LAMMPS
        :type pair_style: str
        :param species: Species that can be used for this potential.
        :type species: list
        :param atom_style: Type of treatment of the atoms according to LAMMPS.
        :type atom_style: str
        :param units: Default units to be used with this potential.
        :type unite:  str
        :param extra_tags: Dictionary with extra information to tag the
            potential, based on the KIM schema.
        :type extra_tags: dict

        :raises TypeError: if the source is not a ``str``, ``pathlib.Path``
            instance or binary stream.
        :raises FileNotFoundError: if the source is a filepath but does not exist.
        """
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
            extra_tags = dict()
        if extra_tags is not None:
            self.validate_extra_tags(extra_tags=extra_tags)
        for key in self._extra_keys:
            self.set_attribute(key, extra_tags.get(key, None))

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
        return self.get_attribute('atom_style')

    @property
    def pair_style(self) -> str:
        """
        Return the `pair_style` of this potential according to the LAMMPS notation
        :return: the `pair_style` of the potential
        :rtype: str
        """
        return self.get_attribute('pair_style')

    @property
    def species(self) -> list:
        """Return the list of species which this potential can be used for.
        :return: The list of chemical species which are contained in this potential.
        :rtype: list
        """
        return self.get_attribute('species')

    @property
    def default_units(self) -> str:
        """
        Return the default units associated with this potential.
        :return: the default units associated with this potential
        :rtype: str
        """
        return self.get_attribute('default_units')

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
        return self.get_attribute('content_origin')

    @property
    def content_other_locations(self) -> typing.Union[str, list]:
        """
        Return other locations where the potential can be found.

        As based in the KIM schema. A description of and/or web address(es)
        to other location(s) where the content is available.

        :return: other locations where the potential can be found.
        :rtype: typing.Union[str, list]
        """
        return self.get_attribute('content_other_locations')

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
        return self.get_attribute('data_method')

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
        return self.get_attribute('description')

    @property
    def developer(self) -> typing.Union[str, list]:
        """
        Return the developer information of this potential.

        As based in the KIM schema. An array of strings, each of which is a KIM
        user uuid corresponding to a "developer" of the item.
        A developer of an item is someone who participated in creating the core
        intellectual content of the digital object, e.g. the functional form
        of an interatomic model or a specific parameter set for it.

        :return: developer information of this potential
        :rtype: typing.Union[str, list]
        """
        return self.get_attribute('developer')

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
        return self.get_attribute('disclaimer')

    @property
    def properties(self) -> typing.Union[str, list]:
        """
        Return the properties for which this potential was devised.

        As based in the KIM schema. A list of properties reported by a KIM Item.

        :return: properties fow which this potential was devised.
        :rtype: typing.Union[str, list]
        """
        return self.get_attribute('properties')

    @property
    def publication_year(self) -> typing.Union[str, datetime.datetime, int]:
        """
        Return the year of publication of this potential.

        As based in the KIM schema. Year this item was published on openkim.org.

        :return: year of publication of this potential
        :rtype: typing.Union[str, datetime.datetime, int]
        """
        return self.get_attribute('publication_year')

    @property
    def source_citations(self) -> typing.Union[str, list]:
        """
        Return the citation where the potential was originally published.

        As based in the KIM schema. An array of BibTeX-style EDN dictionaries
        corresponding to primary published work(s) describing the KIM Item.

        :return: the citation where the potential was originally published.
        :rtype: typing.Union[str, list].
        """
        return self.get_attribute('source_citations')

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
        return self.get_attribute('title')

    @property
    def md5(self) -> typing.Optional[int]:
        """Return the md5.
        :return: the md5 of the stored file.
        """
        return self.get_attribute(self._key_md5, None)

    @property
    def generation_method(self) -> str:
        """
        Return the geneation method of the potential.

        In here one can describe how the potential itself was generated, if it
        was done via ML, fitting via specific codes, analytical fitting, etc.

        :return: the generation method of the potential
        :rtype: str
        """
        return self.get_attribute('generation_method')

    @md5.setter
    def md5(self, value: str):
        """Set the md5.
        :param value: the md5 checksum.
        :raises ValueError: if the md5 does not match that of the currently stored file.
        """
        self.validate_md5(value)
        self.set_attribute(self._key_md5, value)
