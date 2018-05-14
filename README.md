
AiiDA LAMMPS plugin
====================

This a LAMMPS plugin for AiiDA. 
This plugin contains 4 code types:

- lammps.forces: Atomic forces calculation
- lammps.md: Molecular dynamics calculation
- lammps.optimize: Crystal structure optimization
- lammps.combinate: DynaPhoPy calculation using LAMMPS MD trajectory


Note: lammps.combinate requires aiida-phonopy (https://github.com/abelcarreras/aiida-phonopy) 
plugin to work.


Supported Potentials
--------------------
 - EAM
 - Lennad Jones
 - Tersoff
 
 
Examples
--------
Some test calculations are found in the folder **/examples**
