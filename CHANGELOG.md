# Changelog

## v1.0.2 2024-04-26

- Fixing an issue in which some LAMMPS vectorial properties were not working properly.
- Changed how the formatting of the dump command is done to prevent issues from vectorial quantities with undetermined size.
- Added the capability to override the dimension and boundary commands in LAMMPS.
- Pinning the version of jsonschema to avoid issues with python>=3.9.

## v1.0.1 2023-11-28

Minor internal improvements to the code base

## v1.0.0 2023-11-28

⬆️ Support for aiida-core >= 2.0.0

- drop support for python<3.8
- fix deprecation warnings

♻️ Refactoring of the plugin
- Removed the old Calculation interfaces and replaced them by a more flexible instances, either by passing a set of parameters that describe a single stage `LAMMPS` run (`LammpsBaseCalculation`) or by passing the input script directly (`LammpsRawCalculation`).
- Removed the old potential dataclasses, changed them by the `LammpsPotential` class where a potential file can be passed and tagged with a set of attributes to improve qurying.
- Improved the parsing to better handle errors, custom global and site dependent computes.
- Documentation style chaned to MysT
- Coding style changed to black.


✨ Added `LammpsRawCalculation`
This is a `CalcJob` that can handle calculations in which the `LAMMPS` input script and necessary files are explicitly given.

✨ Added `LammpsBaseCalculation`
This is a `CalcJob` that takes a set of parameters and constructs the `LAMMPS` input file for a single stage calculation.

✨ Added `LammpsBaseWorkChain`
A `WorkChain` wrapper for the `LammpsBaseCalculation` to harness the `BaseRestartWorkchain` from `aiida-core` and allow error correction and automatic restarting of the calculation.

✨ Added `LammpsMDWorkChain`
A `WorkChain` that deals specifically with MD runs in `LAMMPS`.

✨ Added `LammpsRelaxWorkChain`
A `WOrkChain` that deals with structural optimization via the `minimize` method in `LAMMPS`.


## v0.8.0  2020-09-29

✨ Support for aiida-core >= 1.4.0

- drop support for python<3.5
- remove `six` dependency
- fix deprecation warnings
- improve pre-commit code style
- add Github CI, tox.ini and docker-compose.yml for improved test infrastructure

♻️ Refactor potential plugins:
Potential plugins are now structured as a class, which inherit from an abstract base class.

✨ Add `MdMultiCalculation`:
This is a generalisation of MdCalculation, which can sequentially run 1 or more integration 'stages', that may have different integration styles and dump rates

✨ implement `LammpsTrajectory`: Instead of using `aiida.orm.TrajectoryData` use a bespoke data object, that directly stores the trajectory file. The trajectory file is also read/stored in step chunks, so will not over exert the memory buffer, and is stored in a compressed zip file. In testing, this halved the memory footprint.

## v0.4.1b3  2019-06-26

Support for aiida-core v1.0.0b3
