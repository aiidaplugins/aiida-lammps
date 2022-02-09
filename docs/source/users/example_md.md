# Molecular dynamics simulations

``LAMMPS`` is widely used to perform molecular dynamics simulations, these can be codified in an ``aiida-lammps`` process by submitting a ``Calculation`` with the correct ``parameters``.

## Parameters setup
To run a MD simulation, one needs to define a dictionary named ``md`` inside the parameters dictionary (which controls the ``LAMMPS`` simulation). The ``md`` dictionary defines the options that control how the molecular dynamics simulations are performed, several entries are needed to fully control its behavior:
* ``max_number_steps``: maximum number of steps for the molecular dynamics simulation (default: 100)
* ``run_style``: type of molecular dynamics algorithm (default: ``verlet``).
* ``velocity``: set of variables needed to define the [velocity](https://docs.lammps.org/velocity.html) of the system.
* ``integration``: parameters relating to the integrators of the molecular dynamics simulation:
    - ``style``: Type of [integrator](https://docs.lammps.org/fixes.html) used for the molecular dynamics simulation. In this example the chosen integrator is [npt](https://docs.lammps.org/fix_nh.html#fix-npt-command), which requires that one at leasts sets the temperature of the simulation box, one can also setup the pressure which acts over the simulation box. These parameters called ``constraints`` are set in another entry of the ``integration`` dictionary.
    - ``constraints``: set of options for each integrator, the values depend on the type of integrator. This dictionary takes as keys the options available for the ``npt`` integrator, in this case the values ``temp`` for the temperature and ``iso`` for the barostat. The values for each key in the dictionary are lists which contain each one of the values that one would normally add besides these commands in the ``LAMMPS`` input.

```{code-block} python
from aiida import orm
from aiida.common.extendeddicts import AttributeDict

parameters = AttributeDict()
parameters.control = AttributeDict()
parameters.control.units = 'metal'
parameters.control.timestep = 1e-5
parameters.compute = {
    'pe/atom': [{
        'type': [{
            'keyword': ' ',
            'value': ' '
        }],
        'group': 'all'
    }],
    'ke/atom': [{
        'type': [{
            'keyword': ' ',
            'value': ' '
        }],
        'group': 'all'
    }],
    'stress/atom': [{
        'type': ['NULL'],
        'group': 'all'
    }],
    'pressure': [{
        'type': ['thermo_temp'],
        'group': 'all'
    }],
}
parameters.md = {
    'integration': {
        'style': 'npt',
        'constraints': {
            'temp': [300, 300, 100],
            'iso': [0.0, 0.0, 1000.0],
        }
    },
    'max_number_steps': 5000,
    'velocity': [{
        'create': {
            'temp': 300
        },
        'group': 'all'
    }]
}
parameters.structure = {'atom_style': 'atomic'}
parameters.thermo = {
    'printing_rate': 100,
    'thermo_printing': {
        'step': True,
        'pe': True,
        'ke': True,
        'press': True,
        'pxx': True,
        'pyy': True,
        'pzz': True,
    }
}
parameters.dump = {'dump_rate': 1000}

PARAMETERS = orm.Dict(dict=parameters)
```

## Structure setup
In this case the structure to be used is bcc Fe, with a basis of two atoms, one can use the ``kind_name`` property for each site in the ``orm.StructureData`` to differentiate the two Fe atoms, that will allow ``aiida-lammps`` to define two possible ``groups`` so that computes, fixes, etc. can be applied individually to each of the atoms.

```{code-block} python
import numpy as np
from aiida import orm

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
```


## Potential setup
When dealing with a new potential that one wishes to use for a simulation one can upload a potential by using the ``get_or_create`` method in the ``LammpsPotentialData``, this method will calculate the ``md5`` checksum of the file and check if exists in the database, if it does that database entry is used, otherwise the file it is uploaded into the database and the used in the simulation. To make the potential easy to find and reuse one can pass a series of optional tags based of the [OpenKIM schema](https://openkim.org/doc/schema/kimspec/), which will provide a systematic way of tagging and finding potentials.

```{code-block} python
from aiida_lammps.data.lammps_potential import LammpsPotentialData

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
        'pair_style': 'eam/fs',
        'units': 'metal',
        'extra_tags': {
            'content_origin':
            'NIST IPRP: https: // www.ctcms.nist.gov/potentials/Fe.html',
            'content_other_locations':
            None,
            'data_method':
            'unknown',
            'description': """
            This Fe EAM potential parameter file is from the NIST repository, \"Fe_2.eam.fs\" as of the March 9, 2009 update. It is similar to \"Fe_mm.eam.fs\" in the LAMMPS distribution dated 2007-06-11, but gives different results for very small interatomic distances (The LAMMPS potential is in fact the deprecated potential referred to in the March 9, 2009 update on the NIST repository). The file header includes a note from the NIST contributor: \"The potential was taken from v9_4_bcc (in C:\\SIMULATION.MD\\Fe\\Results\\ab_initio+Interstitials)\"
            """,
            'developer': ['Ronald E. Miller'],
            'disclaimer':"""
            According to the developer Giovanni Bonny (as reported by the NIST IPRP), this potential was not stiffened and cannot be used in its present form for collision cascades.
            """,
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
        source='Fe_2.eam.fs',
        **potential_parameters,
    )

    return potential
```
