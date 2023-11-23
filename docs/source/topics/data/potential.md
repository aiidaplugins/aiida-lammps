# ``LammpsPotentialData``

The potential is one of the most important pieces of data in a MD simulation, since it controls how the atoms interact with each other.
In ``aiida-lammps`` the potential file is stored in the `LammpsPotentialData` data type, which will store the entire potential file in the database, and add certain attributes so that the data node is easily queryable for later usage. These attributes have been chosen so that they resemble the [OpenKIM](https://openkim.org/doc/schema/kimspec/) standard as much as possible.

To demonstrate how this works one can [download](https://openkim.org/id/EAM_Dynamo_Mendelev_2003_Fe__MO_546673549085_000) a potential from the OpenKIM database, after the file has been downloaded one can generate a dictionary with the metadata of the potential to tag it in the ``AiiDA`` database.

```python
potential_parameters = {
    'species': ['Fe'],  # Which species can be treated by this potential (required)
    'atom_style': 'atomic', # Which kind of atomic style is associated with this potential (required)
    'pair_style': 'eam/fs', # LAMMPS pair style (required)
    'units': 'metal', # Default units of this potential (required)
    'extra_tags': {
        'content_origin': 'NIST IPRP: https: // www.ctcms.nist.gov/potentials/Fe.html', # Where the file was original found
        'content_other_locations': None, # If the file can be found somewhere else
        'data_method': 'unknown', # How was the data generated
        'description': """
        This Fe EAM potential parameter file is from the NIST repository, \"Fe_2.eam.fs\" as of the March 9, 2009 update.
        It is similar to \"Fe_mm.eam.fs\" in the LAMMPS distribution dated 2007-06-11,
        but gives different results for very small interatomic distances
        (The LAMMPS potential is in fact the deprecated potential referred to in the March 9, 2009 update on the NIST repository).
        The file header includes a note from the NIST contributor:
        \"The potential was taken from v9_4_bcc (in C:\\SIMULATION.MD\\Fe\\Results\\ab_initio+Interstitials)\"
        """, # Short description of the potential
        'developer': ['Ronald E. Miller'], # Name of the developer that uploaded it to OpenKIM
        'disclaimer': """
        According to the developer Giovanni Bonny (as reported by the NIST IPRP),
        this potential was not stiffened and cannot be used in its present form for collision cascades.
        """, # Any known issues with the potential
        'properties': None, # If any specific properties are associated to the potential
        'publication_year': 2018, # Year of publication to OpenKIM
        'source_citations': [{
            'abstract': None,
            'author':
            'Mendelev, MI and Han, S and Srolovitz, DJ and Ackland, GJ and Sun, DY and Asta, M',
            'doi': '10.1080/14786430310001613264',
            'journal': '{Phil. Mag.}',
            'number': '{35}',
            'pages': '{3977-3994}',
            'recordkey': 'MO_546673549085_000a',
            'recordprimary': 'recordprimary',
            'recordtype': 'article',
            'title':
            '{Development of new interatomic potentials appropriate for crystalline and liquid iron}',
            'volume': '{83}',
            'year': '{2003}'
        }],
        'title': 'EAM potential (LAMMPS cubic hermite tabulation) for Fe developed by Mendelev et al. (2003) v000' # Title of the potential
    }
}
```
Certain tags are required, and must be provided to be able to upload the potential to the database. This is because they identify which ``pair_style`` is associated with the potential, which atomic species can be treated with it, etc. The rest of the tags, in this example are filled so that they follow the [OpenKIM](https://openkim.org/doc/schema/kimspec/) standard as that is the place where the potential was obtained. If another database is used or if it is a homemade potential, these tags can be used to facilitate the querying of the potential.

Then the potential can be uploaded to the database
```python
from aiida_lamps.data.potential import LammpsPotentialData

potential = LammpsPotentialData.get_or_create(
    source='Fe_2.eam.fs', # Relative path to the potential file
    **potential_parameters, # Parameters to tag the potential
)

```

The ``get_or_create`` method is based on the one by [aiida-pseudo](https://github.com/aiidateam/aiida-pseudo/blob/master/aiida_pseudo/data/pseudo/pseudo.py), which will calculate the md5 sum of the file and check the database for another file with the same [md5 hash](https://en.wikipedia.org/wiki/MD5), if such entry is found, that potential is used instead. This avoids the unnecessary replication of potential data nodes whenever one tries to upload a previously uploaded potential.

:::{note}
When calculating the md5 hash the program will look at the contents of the file, so that even if a minor change is done (that should not affect the result of a calculation), the checksum will be different and hence a new potential node will be created.
:::

## Potentials without files
In ``LAMMPS`` certain [pair_style](https://docs.lammps.org/pair_style.html) such as the Lenard-Johns potential do not read their parameters from an auxiliary file, if not they are directly written to the main input file. In ``aiida-lammps`` to standardize the potential storage in the database these kinds of potentials are expected to be also be stored as a file. The format expected for these kinds of potentials is simply the coefficients that one would normally write the in the ``LAMMPS`` input file. The input file generator will then generate the necessary lines for the input file depending on the potential ``pair_style``.


## Potentials with multiple files
In ``LAMMPS`` it is in principle possible to give several potential files to treat different atoms. Currently this is **not** supported in the plugin. As only one potential file can be give as to treat the entire system. This is a situation that is aimed to be solved in future releases.
