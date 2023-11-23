---
myst:
  substitutions:
    aiida_lammps: '`aiida-lammps`'
    LAMMPS: '[LAMMPS](https://lammps.org)'
    OpenKIM: '[OpenKIM](https://openkim.org/)'
---

(tutorials-raw)=

# Raw LAMMPS calculation

Sometimes transforming a {{ LAMMPS }} script into a set of parameters that can be passed as a dictionary to {{ aiida_lammps }} can be very complicated or impossible. That is why the `LammpsRawCalculation` is included, as it gives a way to pass a functioning {{ LAMMPS }} script to {{ aiida_lammps }} and run it via AiiDA. This will store the calculation in the AiiDA provenance graph and perform some basic parsing functions. However, as a great deal of the information needed to be able to parse the data is not present (due to the lack of parameters passed to the calculation engine) many of the automatic parsing done in the `LammpsBaseCalculation` is not performed in this case.

:::{note}
The usage of the `LammpsRawCalculation` also introduces difficulties with regards to the querying of results. With the `LammpsBaseCalculation` one passes several nodes, parameters, structure and potential which can be used in the AiiDA query engine to get specific calculations. As these do not exist for the `LammpsRawCalculation` the querying can be severely limited.
:::

:::{tip}
The code shown in the snippets below can be {download}`downloaded as a script <include/scripts/run_raw_basic.py>`,
The script can be made executable and then run to execute the example calculation.
:::



First import the required classes and functions:

```python
from aiida.plugins import CalculationFactory
from aiida import engine
from aiida.orm import SinglefileData, load_code
```

Then, load the code that was setup in AiiDA for `lmp` and get an instance of the [process builder](https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/processes/usage.html#process-builder):

```python
# Load the code configured for ``lmp``. Make sure to replace
# this string with the label used in the code setup.
code = load_code('lammps@localhost')
builder = CalculationFactory("lammps.raw").get_builder()
builder.code = code
```

The process builder can be used to assign and automatically validate the inputs that will be used for the calculation.

For the raw calculation the most important piece is to pass the LAMMPS script that will be run. To be able to pass it to AiiDA one needs to store it as a `SinglefileData` node, which basically stores a file in the AiiDA provenance graph. When a `LammpsRawCalculation` is submitted this file will be copied **exactly** in the machine performing the calculation.
```python
import io
import textwrap

script = SinglefileData(
    io.StringIO(
        textwrap.dedent(
            """
            # Rhodopsin model

            units           real
            neigh_modify    delay 5 every 1

            atom_style      full
            bond_style      harmonic
            angle_style     charmm
            dihedral_style  charmm
            improper_style  harmonic
            pair_style      lj/charmm/coul/long 8.0 10.0
            pair_modify     mix arithmetic
            kspace_style    pppm 1e-4

            read_data       data.rhodo

            fix             1 all shake 0.0001 5 0 m 1.0 a 232
            fix             2 all npt temp 300.0 300.0 100.0 &
                    z 0.0 0.0 1000.0 mtk no pchain 0 tchain 1

            special_bonds   charmm

            thermo          50
            thermo_style    multi
            timestep        2.0

            run     100
            """
        )
    )
)
builder.script = script
```

As one can notice the script wants to read a file named `data.rhodo` via the [`read_data`](https://docs.lammps.org/read_data.html) command. One can pass any set of files that the script might need, in this case a file stored in the lammps repository that is downloaded using the [requests library](https://docs.python-requests.org/en/latest/index.html), by first storing them as `SinglefileData` nodes and the passing them to the builder as follows:

```python
import requests
request = requests.get("https://raw.githubusercontent.com/lammps/lammps/develop/bench/data.rhodo")
data = SinglefileData(io.StringIO(request.text))
builder.files = {"data": data}
builder.filenames = {"data": "data.rhodo"}
```

:::{important}
Notice that one first passes the files in a dictionary with a key called `data`, the filename dictionary specifies the name that will be given to the file stored under the key `data` in the machine performing the calculation. One needs to ensure that this name, `data.rhodo` in this case, matches the expected name by the script.
:::

Lastly one needs to define the computational resources needed to perform the calculation
```python
# Run the calculation on 1 CPU and kill it if it runs longer than 1800 seconds.
# Set ``withmpi`` to ``False`` if ``pw.x`` was compiled without MPI support.
builder.metadata.options = {
    'resources': {
        'num_machines': 1,
    },
    'max_wallclock_seconds': 1800,
    'withmpi': False,
}
```

Now as all the needed parameters have been defined the calculation can bse launched using the process builder:

```python
outputs, node = engine.get_node(builder)
```

Once the calculation is finished `run.get_node` will return the outputs produced and the calculation node, `outputs` and `node` respectively.

The `node` is the entry that contains the information pertaining the calculation.
It is possible to check if the calculation finished successfully (processes that return `0` are considered to be successful) by looking at its exit status:

```python
node.exit_status
```

If the result is different from zero it means that a problem was encountered in the calculation. This might indicate that some output is not present, that the calculation failed due to a transitory issue, an input problem, etc.

The `outputs` is a dictionary containing the output nodes produced by the calculation:

```python
print(outputs)
{
    'remote_folder': <RemoteData: uuid: 70b075de-1597-4997-a4c1-7a86af790dfb (pk: 77529)>,
    'retrieved': <FolderData: uuid: 83b32034-7eef-4f0b-b567-f312a46cc2d3 (pk: 77530)>,
    'results': <Dict: uuid: c0fc582e-16b3-464f-8627-3023baebc459 (pk: 77531)>
}
```

The `results` node is a dictionary that will contain some basic parsed information from the data written to the stdout


```python
print(outputs['results'].get_dict())
{
    'compute_variables': {
        'bin': 'standard',
        'bins': [10, 13, 13],
        'errors': [],
        'binsize': 6,
        'warnings': [],
        'units_style': 'real',
        'total_wall_time': '0:00:20',
        'steps_per_second': 5.046,
        'ghost_atom_cutoff': 12,
        'max_neighbors_atom': 2000,
        'total_wall_time_seconds': 20,
        'master_list_distance_cutoff': 12
    }
}
```

The complete output that was written by {{ LAMMPS }} to stdout, can be retrieved as follows:

```python
results['retrieved'].base.repository.get_object_content('lammps.out')
```
