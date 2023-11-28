---
myst:
  substitutions:
    aiida_lammps: '`aiida-lammps`'
    LAMMPS: '[LAMMPS](https://lammps.org)'
    AiiDA: '[AiiDA](https://www.aiida.net/)'
    LammpsBaseCalculation: '{class}`~aiida_lammps.calculations.base.LammpsBaseCalculation`'
    LammpsRawCalculation: '{class}`~aiida_lammps.calculations.raw.LammpsRawCalculation`'
    Int: '{class}`~aiida.orm.nodes.data.int.Int`'
    Float: '{class}`~aiida.orm.nodes.data.float.Float`'
    Str: '{class}`~aiida.orm.nodes.data.str.Str`'
    Dict: '{class}`~aiida.orm.nodes.data.dict.Dict`'
    List: '{class}`~aiida.orm.nodes.data.list.List`'
    Bool: '{class}`~aiida.orm.nodes.data.bool.Bool`'
---

# ``LammpsRelaxWorkChain``

This is a subclass of the {class}`~aiida_lammps.workflows.base.LammpsBaseWorkChain` which focuses on minimization simulations specifically. It overrides any set of parameters given in the minimize block (see [](#topics-data-parameters)) and instead directly exposes them to the user in the `relax` input namespace.

## Inputs:

- **lammps.structure**, ({class}`~aiida.orm.nodes.data.structure.StructureData`) - Structure used in the ``LAMMPS`` calculation.
- **lammps.potential**, ({class}`~aiida_lammps.data.potential.LammpsPotentialData`) - Potential used in the ``LAMMPS`` calculation. See [](#topics-data-potential).
- **lammps.parameters**, ({class}`~aiida.orm.nodes.data.dict.Dict`) - Parameters that control the input script generated for the ``LAMMPS`` calculation. See [](#topics-data-parameters).
- **lammps.settings**, ({class}`~aiida.orm.nodes.data.dict.Dict`), *optional* - Additional settings that control the ``LAMMPS`` calculation. One can control if extra filess will be copied to the repository by specifying `settings["additional_retrieve_list"] = ["foo", "bar"]`. It is also possible to do pattern matching via [globs patterns](https://en.wikipedia.org/wiki/Glob_%28programming%29) by `settings["additional_retrieve_list"] = [('path/sub/*c.txt', '.', None)]`, for more information see the [pattern matching](https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/calculations/usage.html#pattern-matching) in the `aiida-core` documentation.
- **lammps.input_restartfile** ({class}`~aiida.orm.nodes.data.singlefile.SinglefileData`), *optional* - Input restart file to continue from a previous ``LAMMPS`` calculation.
- **lammps.parent_folder**, ({class}`~aiida.orm.nodes.data.remote.base.RemoteData`), *optional* - An optional working directory of a previously completed calculation to restart from.
- **store_restart**, ({{ Bool }}), *optional* - Whether to store the restart file in the repository. Defaults to `False`.
- **relax.algo**, ({{ Str }}), *optional* - The algorithm to be used during relaxation. Defaults to cg.
- **relax.volume**, ({{ Bool }}), *optional* -  Whether or not relaxation of the volume will be performed by using the [box/relax](https://docs.lammps.org/fix_box_relax.html) fix from LAMMPS. Defaults to `False`.
- **relax.shape**, ({{ Bool }}), *optional* - Whether or not the shape of the cell will be relaxed by using the [box/relax](https://docs.lammps.org/fix_box_relax.html) fix from LAMMPS. Defaults to `False`
- **relax.positions**, ({{ Bool }}), *optional* - Whether or not to allow the relaxation of the atomic positions. Defaults to `True`.
- **relax.steps**, ({{ Int }}), *optional* - Maximum number of steps during the relaxation. Defaults to `1000`.
- **relax.evaluations**, ({{ Int }}), *optional* - Maximum number of force/energy evaluations during the relaxation. Defaults to `10000`.
- **relax.energy_tolerance**, ({{ Float }}), *optional* - The tolerance that determined whether the relaxation procedure is stopped. In this case it stops when the relative change between outer iterations of the relaxation run is less than the given value. Defaults to 1e-4.
- **relax.force_tolerance**, ({{ Float }}), *optional* - The tolerance that determines whether the relaxation procedure is stopped. In this case it stops when the 2-norm of the global force vector is less than the given value. Defaults to `1e-4`.
- **relax.target_pressure**, ({{ Dict }}), *optional* - Dictionary containing the values for the target pressure tensor. See the [box/relax](https://docs.lammps.org/fix_box_relax.html) for more information.
- **relax.max_volume_change**, ({{ Float }}), *optional* - Maximum allowed change in one iteration (``vmax``).
- **relax.nreset**, ({{ Int }}), *optional* - Reset the reference cell every this many minimizer iterations.
- **relax.meta_convergence**, ({{ Bool }}), *optional* - If `True` the workchain will perform a meta-convergence on the cell volume. Defaults to `True`.
- **relax.max_meta_convergence_iterations**, ({{ Int}}), *optional* - The maximum number of variable cell relax iterations in the meta convergence cycle. Defaults to `5`.
- **relax.volume_convergence**, ({{ Float }}), *optional* - The volume difference threshold between two consecutive meta convergence iterations. Defaults to `0.1`.

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
- **retrieved**, ({class}`~aiida.orm.nodes.data.folder.FolderData`) - Files that are retrieved by the daemon will be stored in this node. By default the stdout and stderr of the scheduler will be added.
