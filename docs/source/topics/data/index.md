# Data

```{toctree}
:maxdepth: 1

parameters
potential
trajectory

```

`aiida-lammps` has two specific data types, the [`LammpsPotentialData`](potential.md) handling the different types of interatomic potentials and the [`LammpsTrajectory`](trajectory.md) dealing with the atomic positions and the time dependent site dependent calculated properties of the system.

Another data set of interest, is the [parameters](parameters.md), this is a dictionary that contains the instructions on how to generate the LAMMPS input file. It abstracts, as much as possible, the generation of a single stage LAMMPS calculation input file into a python dictionary.
