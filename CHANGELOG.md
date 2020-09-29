# Changelog

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
