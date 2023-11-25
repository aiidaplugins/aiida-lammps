---
myst:
  substitutions:
    aiida_lammps: '`aiida-lammps`'
    LAMMPS: '[LAMMPS](https://lammps.org)'
    AiiDA: '[AiiDA](https://www.aiida.net/)'
---

(topics-workflows)=

# Workflows

```{toctree}
:maxdepth: 1

base
relax
md
```

The workflows act as abstractions of the {class}`~aiida_lammps.calculations.base.LammpsBaseCalculation` focusing on being able to perform error correction as well as specific tasks. The idea is that these are used as building blocks for more complex calculations, offloading what is normally done inside by scripting in the {{ LAMMPS }} input to {{ AiiDA }}. The advantage of this is that each step of the complex workflow will be stored in the {{ AiiDA }} provenance graph, and will be able to be used for other calculations, data analysis or just as a way to effectively monitor what happens at each stage. The drawback with this approach is that each stage is a {{ LAMMPS }} calculation, which implies some overhead, as well as effort into rewriting the {{ LAMMPS }} script into an {{ AiiDA }} compliant workflow.
