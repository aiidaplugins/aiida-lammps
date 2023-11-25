---
myst:
  substitutions:
    aiida_lammps: '`aiida-lammps`'
    LAMMPS: '[LAMMPS](https://lammps.org)'
    AiiDA: '[AiiDA](https://www.aiida.net/)'
    LammpsBaseCalculation: '{class}`~aiida_lammps.calculations.base.LammpsBaseCalculation`'
    LammpsRawCalculation: '{class}`~aiida_lammps.calculations.raw.LammpsRawCalculation`'
---


(topics-calculations)=

# Calculations

```{toctree}
:maxdepth: 1

base
raw

```

{{ aiida_lammps }} has two different types of calculations {{ LammpsBaseCalculation }} and {{ LammpsRawCalculation }}. The {{ LammpsBaseCalculation }} generates a {{ LAMMPS }} input script for a single stage calculation from a set of parameters passed as as dictionary to the calculation. However, to give more flexibility for cases not being tractable with a single stage calculation, or that require options not covered by the {{ LammpsBaseCalculation }} the {{ LammpsRawCalculation }} is designed to be able to accept any {{ LAMMPS }} script as an input.
