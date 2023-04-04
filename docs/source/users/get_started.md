# Getting started

In a traditional ``LAMMPS`` calculation a user has an input file which is sequentially read by the executable, each line has a command specifying what actions will be taken, usually the potential is kept as a separate file which is then referred to in the input file.

``aiida-lammps`` generates the necessary inputs by taking three principal data types, namely the ``structure``, the ``parameters`` and the ``potential``.

## Potential

The potential is one of the most important pieces of data in a MD simulation, since it controls how the atoms interact with each other.
In ``aiida-lammps`` the potential file is stored in the `LammpsPotentialData` data type, which will store the entire potential file in the database, and add certain attributes so that the data node is easily queryable for later usage, these attributes have been chosen so that they resemble the [OpenKIM](https://openkim.org/doc/schema/kimspec/) standard as much as possible.

To demonstrate how this works one can [download](https://openkim.org/id/EAM_Dynamo_Mendelev_2003_Fe__MO_546673549085_000) a potential from the OpenKIM database, after the file has been downloaded one can generate a dictionary with the metadata of the potential to tag it in the AiiDA database.

```{code-block} python
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
Certain tags are required, and must be provided to be able to upload the potential to the database. This is because they identify which ``pair_style`` is associated with the potential, which atomic species can be treated with it, etc. The rest of the tags, in this example are filled so that they follow the OpenKIM standard as that is the place where the potential was obtained, if another database is used or if it is a homemade potential, these tags can be used to facilitate the querying of the potential.

Then the potential can be uploaded to the database
```{code-block} python
from aiida_lamps.data.lammps_potential import LammpsPotentialData

potential = LammpsPotentialData.get_or_create(
    source='Fe_2.eam.fs', # Relative path to the potential file
    **potential_parameters, # Parameters to tag the potential
)

```

The ``get_or_create`` method is based on the one by [aiida-pseudo](https://github.com/aiidateam/aiida-pseudo/blob/master/aiida_pseudo/data/pseudo/pseudo.py), which will calculate the md5 sum of the file and check the database for another file with the same md5 sum, if such entry is found, that potential is used instead. This avoids the unnecessary replication of potential data nodes whenever one tries to upload a previously uploaded potential.

### Potentials without files
In ``LAMMPS`` certain [pair_style](https://docs.lammps.org/pair_style.html) such as the Lenard-Johns potential do not read their parameters from an auxiliary file, if not they are directly written to the main input file. In ``aiida-lammps`` to standardize the potential storage in the database these kinds of potentials are expected to be also be stored as a file. The format expected for these kinds of potentials is simply the coefficients that one would normally write the in the ``LAMMPS`` input file. The input file generator will then generate the necessary lines for the input file depending on the potential ``pair_style``.

## Parameters
The behavior of the ``aiida-lammps`` calculation can be controlled by collecting ``LAMMPS`` simulation parameters in a dictionary

```{code-block} python
parameters = {
    'md': {
        'velocity': [{'group': 'all', 'create': {'temp': 300}}],
        'integration': {
            'style': 'npt',
            'constraints': {'iso': [0.0, 0.0, 1000.0], 'temp': [300, 300, 100]}
        },
        'max_number_steps': 5000
    },
    'dump': {'dump_rate': 1000},
    'thermo': {
        'printing_rate': 100,
        'thermo_printing': {
            'ke': True,
            'pe': True,
            'pxx': True,
            'pyy': True,
            'pzz': True,
            'step': True,
            'press': True
        }
    },
    'compute': {
        'ke/atom': [{'type': [{'value': ' ', 'keyword': ' '}], 'group': 'all'}],
        'pe/atom': [{'type': [{'value': ' ', 'keyword': ' '}], 'group': 'all'}],
        'pressure': [{'type': ['thermo_temp'], 'group': 'all'}],
        'stress/atom': [{'type': ['NULL'], 'group': 'all'}]
    },
    'control': {'units': 'metal', 'timestep': 1e-05},
    'structure': {'atom_style': 'atomic'}
}
```

The dictionary is separated into several nested dictionaries that control different behaviors of the ``LAMMPS`` simulation:
- ``control``: takes keywords specifying global simulation parameters:
    * ``units``: ``LAMMPS`` [units](https://docs.lammps.org/units.html) used in the calculation (default: ``si``).
    * ``timestep``: [time step](https://docs.lammps.org/timestep.html) used in the simulation, it depends on the units used (default: ``LAMMPS`` default dependent on units parameter).
    * ``newton``: it controls whether the Newton's third law is [turned on or off](https://docs.lammps.org/newton.html) for the calculation (default: ``on``).
    * ``processors``: specifies how [processors](https://docs.lammps.org/processors.html) are mapped to the simulation box (default: ignore the command).
- ``structure``: variables controlling structure options:
    * ``box_tilt``: determines how [skewed the cell](https://docs.lammps.org/box.html) is, of great importance for triclinic systems (default: ``small``).
    * ``groups``: list with the names of the groups to be added. The names of the possible groups are generated by the list of possible kind names generated by the structure (default: skip parameter).
    * ``atom_style``: how the [atoms](https://docs.lammps.org/atom_style.html) are treated by the ``LAMMPS`` simulation.
- ``potential``: parameters related to the potential describing the system:
    * ``potential_style_options``: extra parameters related to each of the possible pair styles (default: skip parameter).
    * ``neighbor``: sets the parameters affecting the construction of the [neighbor list](https://docs.lammps.org/neighbor.html) (default: skip parameter).
    * ``neighbor_modify``: set of options that [modify](https://docs.lammps.org/neigh_modify.html) the pairwise neighbor list generation (default: skip parameter).
- ``dump``: controls parameters regarding the printing of the site dependent quantities:
    * ``dump_rate``: how often are the site dependent quantities printed to file (default: 10).
- ``compute``: set of lists containing information about which ``LAMMPS`` [computes](https://docs.lammps.org/compute.html) should be calculated. For each ``LAMMPS`` command one passes a list of dictionaries, each dictionary has a ``type`` key containing the options of the compute and ``group`` a key specifying over which group the compute is acting on.
- ``fix``: set of list containing information about which ``LAMMPS`` [fixes](https://docs.lammps.org/fix.html) should be calculated.  For each ``LAMMPS`` command one passes a list of dictionaries, each dictionary has a ``type`` key containing the options of the fixes and ``group`` a key specifying over which group the fix is acting on.
- ``thermo``: set of variables indicating which global parameters (printed in the ``lammps.log``) should be printed:
    * ``printing_rate``: how often should the parameters be written to the ``lammps.log`` (default: 1000)
    * ``thermo_printing``: dictionary containing which ``LAMMPS`` internal [variables](https://docs.lammps.org/thermo_style.html) are printed to the ``lammps.log``. The keys are the names of ``LAMMPS`` parameters and the value is a boolean on whether to print it or not.
- ``md``: set of variables controlling a molecular dynamics simulation (exclusionary with ``minimize`` key word):
    * ``max_number_steps``: maximum number of steps for the molecular dynamics simulation (default: 100)
    * ``run_style``: type of molecular dynamics algorithm (default: ``verlet``).
    * ``velocity``: set of variables needed to define the [velocity](https://docs.lammps.org/velocity.html) of the system.
    * ``integration``: parameters relating to the integrators of the molecular dynamics simulation:
        - ``style``: Type of [integrator](https://docs.lammps.org/fixes.html) used for the molecular dynamics simulation.
        - ``constraints``: set of options for each integrator, the values depend on the type of integrator.
- ``minimize``: set of variables controlling a minimization simulation (exclusionary with ``md`` key word):
    * ``style``: [type of minimization](https://docs.lammps.org/min_style.html) algorithm (default: ``cg``).
    * ``energy_tolerance``: tolerance for the energy minimization (default: 1e-4).
    * ``force_tolerance``: tolerance for the force minimization (default: 1e-4).
    * ``max_iterations``: maximum number of iterations (default: 1000).
    * ``max_evaluations``: maximum number of evaluations (default: 1000).
- ``restart``: set of variables controlling the printing of the binary file to [restart](https://docs.lammps.org/Howto_restart.html)  a ``LAMMPS`` calculation.
    * ``print_final``: whether or not to print a restart file at the end of the calculation, equivalent to setting [write_restart](https://docs.lammps.org/write_restart.html) at the end of the calculation (default: ``False``).
    * ``print_intermediate``: whether or not to print restart files throughout the run at regular intervals, equivalent to the [restart](https://docs.lammps.org/restart.html) ``LAMMPS`` command (default: ``False``).
    * ``num_steps``: however often the restart file is written if ``print_intermediate`` is used (default: ``max_number_steps*0.1``).

```{note}
As the restart files can be very large, they are by default not printed, nor stored in the database. Even when one prints them with the ``print_final`` and/or ``print_intermediate`` they are not retrieved and are only kept in the remote folder. The storage of the restart file can be controlled via the ``store_restart=True``(``store_restart=False``) to store(not-store) option in the ``settings`` dictionary.
```
### Compute parameters
When asking ``aiida-lammps`` to calculate a certain ``compute`` its ``LAMMPS`` name will be automatically generated following the pattern ``compute_name_group_name_aiida`` where ``compute_name`` is the ``LAMMPS`` name of the compute, e.g. ``pe/atom`` with the difference than the ``/`` character is replaced by ``_`` and ``group_name`` is the name of the group to which the compute is applied. All global computes are printed to the ``lammps.log`` and all site dependent quantities are printed to the trajectory file. These computes can then be accessed as outputs of the simulation.

### Input validation
``LAMMPS`` has a quite large amount of possible parameters which can be passed into it to control its behavior. Many of these options are incompatible which can cause the ``LAMMPS`` simulation to fail. To try to catch as many as possible of these possible conflicts the ``aiida-lammps`` input is validated against a [JSON schema](https://json-schema.org/understanding-json-schema/index.html), that checks that the provided input parameters fulfill this schema as much as possible, e.g. it checks that only ``LAMMPS`` computes can be passed to the ``compute`` block, etc. Due to the large amount and variety of options for each compute/fixes these options are not thoroughly checked, only the name of the compute itself is checked.
