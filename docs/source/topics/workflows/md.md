---
myst:
  substitutions:
    aiida_lammps: '`aiida-lammps`'
    LAMMPS: '[LAMMPS](https://lammps.org)'
    AiiDA: '[AiiDA](https://www.aiida.net/)'
    LammpsBaseCalculation: '{class}`~aiida_lammps.calculations.base.LammpsBaseCalculation`'
    LammpsRawCalculation: '{class}`~aiida_lammps.calculations.raw.LammpsRawCalculation`'
    Int: '{class}`~aiida.orm.nodes.data.int.Int`'
    Str: '{class}`~aiida.orm.nodes.data.str.Str`'
    Dict: '{class}`~aiida.orm.nodes.data.dict.Dict`'
    List: '{class}`~aiida.orm.nodes.data.list.List`'
---

# ``LammpsMDWorkChain``

This is a subclass of the {class}`~aiida_lammps.workflows.base.LammpsBaseWorkChain` which focuses on MD simulations specifically. It overrides any set of parameters given in the md block (see [](#topics-data-parameters)) and instead directly exposes them to the user in the `md` input namespace.


## Inputs:

- **lammps.structure**, ({class}`~aiida.orm.nodes.data.structure.StructureData`) - Structure used in the ``LAMMPS`` calculation.
- **lammps.potential**, ({class}`~aiida_lammps.data.potential.LammpsPotentialData`) - Potential used in the ``LAMMPS`` calculation. See [](#topics-data-potential).
- **lammps.parameters**, ({class}`~aiida.orm.nodes.data.dict.Dict`) - Parameters that control the input script generated for the ``LAMMPS`` calculation. See [](#topics-data-parameters).
- **lammps.settings**, ({class}`~aiida.orm.nodes.data.dict.Dict`), *optional* - Additional settings that control the ``LAMMPS`` calculation.
- **lammps.input_restartfile** ({class}`~aiida.orm.nodes.data.singlefile.SinglefileData`), *optional* - Input restart file to continue from a previous ``LAMMPS`` calculation.
- **lammps.parent_folder**, ({class}`~aiida.orm.nodes.data.remote.base.RemoteData`), *optional* - An optional working directory of a previously completed calculation to restart from.
- **store_restart**, ({class}`~aiida.orm.nodes.data.bool.Bool`), *optional* - Whether to store the restart file in the repository. Defaults to `False`.
- **md.steps**, ({{ Int }}), *optional* - Number of steps in the MD simulation. Defaults to `1000`.
- **md.algo**, {{ Str }}, *optional* - Type of time integrator used for MD simulations in LAMMPS ([run_style](https://docs.lammps.org/run_style.html)). Defaults to verlet.
- **md.integrator**, ({{ Str }}), *optional* - Type of thermostat used for the MD simulation in LAMMPS, e.g. ``fix npt``. Defaults to `npt`.
- **md.integrator_constraints**, ({{ Dict }}), *optional* - Set of constraints that are applied to the thermostat. Defaults to `{"temp":[300,300,100], "iso":[0.0, 0.0, 1000]}`.
- **md.velocity**, ({{ List }}), *optional* - List with the information describing how to generate the velocities for the initialization of the MD run.
- **md.respa_options**, ({{ List }}), *optional* - List with the information needed to setup the respa options.

:::{note}
LAMMPS can produce binary restart files which contain all the atomic positions, forces and other computed variables until when the are asked to be printed. One can control this by passing a dictionary called `restart` to the `settings` input. The available options for the `restart` are:
- `print_final`, (`bool`) - whether to print a restart file at the end of the calculation. Defaults to `False`. See [`write_restart`](https://docs.lammps.org/write_restart.html).
- `print intermediate`, (`bool`) - whether to print restart files periodically throughout the calculation. Defaults to `False`. See [`restart`](https://docs.lammps.org/restart.html).
- `num_steps`, (`int`) - how often is the intermediate restart file printed. Defaults to 10% of the total number of steps.
:::

## Outputs:

- **results**, ({class}`~aiida.orm.nodes.data.dict.Dict`) - The parsed data extracted from the lammps output file.
- **trajectories** ({class}`~aiida_lammps.data.trajectory.LammpsTrajectory`) - The data extracted from the lammps trajectory file, includes the atomic trajectories and the site and time dependent calculation parameters.
- **time_dependent_computes**, ({class}`~aiida.orm.nodes.data.array.array.ArrayData`) - The data with the time dependent computes parsed from the lammps.out.
- **restartfile**, ({class}`~aiida.orm.nodes.data.singlefile.SinglefileData`), *optional* - The restart file of a ``LAMMPS`` calculation.
- **structure**, ({class}`~aiida.orm.nodes.data.structure.StructureData`), *optional* - The output structure of the calculation.
- **remote_folder**, ({class}`~aiida.orm.nodes.data.remote.base.RemoteData`) - Folder in the remote machine where the calculation was performed.
- **remote_stash**, ({class}`~aiida.orm.nodes.data.remote.stash.base.RemoteStashData`), *optional* – Contents of the stash.source_list option are stored in this remote folder after job completion.
- **retrieved**, ({class}`~aiida.orm.nodes.data.folder.FolderData`) - Files that are retrieved by the daemon will be stored in this node. By default the stdout and stderr of the scheduler will be added, but one can add more by specifying them in `settings["additional_retrieve_list"] = ["foo", "bar"]`.
