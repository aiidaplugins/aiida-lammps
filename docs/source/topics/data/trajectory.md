---
myst:
  substitutions:
    aiida_lammps: '`aiida-lammps`'
    LAMMPS: '[LAMMPS](https://lammps.org)'
    AiiDA: '[AiiDA](https://www.aiida.net/)'
---

# ``LammpsTrajectory``

During the course of a {{ LAMMPS }} simulation large trajectory files are generated. However, these are text files that can be compressed greatly reducing the used space. Hence, storing them directly in the {{ AiiDA }} repository is not efficient. The {class}`~aiida_lammps.data.trajectory.LammpsTrajectory` data class takes care of this by storing each trajectory step is as a separate file, within a compressed zip folder. Thus reducing storage space, and allowing for fast access to each step.

It is possible to access the data for each step using several methods:
- {meth}`~aiida_lammps.data.trajectory.LammpsTrajectory.get_step_string`: which returns the raw step string from the {{ LAMMPS }} output for a given step.
- {meth}`~aiida_lammps.data.trajectory.LammpsTrajectory.get_step_data`: which returns the parsed data for a given step.
- {meth}`~aiida_lammps.data.trajectory.LammpsTrajectory.get_step_structure`: which returns a {class}`~aiida.orm.nodes.data.structure.StructureData` instance for the crystal structure for a given step.
