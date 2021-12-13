"""
Sets up an example for the calculation of bcc Fe using ``aiida-lammps``.
"""
import numpy as np
from aiida import orm
from aiida_lammps.data.lammps_potential import LammpsPotentialData


def generate_structure() -> orm.StructureData:
    """
    Generates the structure for the calculation.

    It will create a bcc structure in a square lattice.

    :return: structure to be used in the calculation
    :rtype: orm.StructureData
    """

    cell = [
        [2.848116, 0.000000, 0.000000],
        [0.000000, 2.848116, 0.000000],
        [0.000000, 0.000000, 2.848116],
    ]

    positions = [
        (0.0000000, 0.0000000, 0.0000000),
        (0.5000000, 0.5000000, 0.5000000),
    ]
    fractional = True

    symbols = ['Fe', 'Fe']
    names = ['Fe1', 'Fe2']

    structure = orm.StructureData(cell=cell)
    for position, symbol, name in zip(positions, symbols, names):
        if fractional:
            position = np.dot(position, cell).tolist()
        structure.append_atom(position=position, symbols=symbol, name=name)

    return structure


def generate_potential() -> LammpsPotentialData:
    """
    Generate the potential to be used in the calculation.

    Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

    :return: potential to do the calculation
    :rtype: LammpsPotentialData
    """

    potential_parameters = {
        'species': ['Fe'],
        'atom_style': 'atomic',
        'pair_style': 'eam',
        'units': 'metal',
        'extra_tags': {
            'content_origin':
            'NIST IPRP: https: // www.ctcms.nist.gov/potentials/Fe.html',
            'content_other_locations':
            None,
            'data_method':
            'unknown',
            'description':
            "This Fe EAM potential parameter file is from the NIST repository, \"Fe_2.eam.fs\" as of the March 9, 2009 update. It is similar to \"Fe_mm.eam.fs\" in the LAMMPS distribution dated 2007-06-11, but gives different results for very small interatomic distances (The LAMMPS potential is in fact the deprecated potential referred to in the March 9, 2009 update on the NIST repository). The file header includes a note from the NIST contributor: \"The potential was taken from v9_4_bcc (in C:\\SIMULATION.MD\\Fe\\Results\\ab_initio+Interstitials)\"",
            'developer': ['Ronald E. Miller'],
            'disclaimer':
            'According to the developer Giovanni Bonny (as reported by the NIST IPRP), this potential was not stiffened and cannot be used in its present form for collision cascades.',
            'properties':
            None,
            'publication_year':
            2018,
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
            'title':
            'EAM potential (LAMMPS cubic hermite tabulation) for Fe developed by Mendelev et al. (2003) v000'
        }
    }

    potential = LammpsPotentialData.get_or_create(
        source='./Fe_2.eam.fs',
        filename='./Fe_2.eam.fs',
        **potential_parameters,
    )

    return potential
