[![Build Status](https://travis-ci.org/abelcarreras/aiida-lammps.svg?branch=master)](https://travis-ci.org/abelcarreras/aiida-lammps)

# AiiDA LAMMPS plugin

This a LAMMPS plugin for [AiiDA](http://aiida-core.readthedocs.io/).
This plugin contains 4 code types:

- `lammps.forces`: Atomic single-point forces calculation
- `lammps.optimize`: Crystal structure optimization
- `lammps.md`: Molecular dynamics calculation
- `lammps.combinate`: DynaPhoPy calculation using LAMMPS MD trajectory (currently untested)

Note: `lammps.combinate` requires `aiida-phonopy` (https://github.com/abelcarreras/aiida-phonopy)
plugin to work, DynaPhoPy can be found in: https://github.com/abelcarreras/aiida-phonopy

- [AiiDA LAMMPS plugin](#AiiDA-LAMMPS-plugin)
  - [Built-in Potential Support](#Built-in-Potential-Support)
  - [Examples](#Examples)
    - [Code Setup](#Code-Setup)
    - [Structure Setup](#Structure-Setup)
    - [Potential Setup](#Potential-Setup)
    - [Force Calculation](#Force-Calculation)
    - [Optimisation Calculation](#Optimisation-Calculation)
    - [MD Calculation](#MD-Calculation)

## Built-in Potential Support

- EAM
- Lennad Jones
- Tersoff
- ReaxFF

## Examples

More example calculations are found in the folder **/examples**,
and there are many test examples in **/aiida_lammps/tests/test_calculations**.

### Code Setup

```python
from aiida_lammps.tests.utils import (
    get_or_create_local_computer, get_or_create_code)
from aiida_lammps.tests.utils import lammps_version

computer_local = get_or_create_local_computer('work_directory', 'localhost')
code_lammps_force = get_or_create_code('lammps.force', computer_local, 'lammps')
code_lammps_opt = get_or_create_code('lammps.optimize', computer_local, 'lammps')
code_lammps_md = get_or_create_code('lammps.md', computer_local, 'lammps')

meta_options = {
    "resources": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1}
}
```

### Structure Setup

```python
from aiida.plugins import DataFactory
import numpy as np

cell = [[3.1900000572, 0, 0],
        [-1.5950000286, 2.762621076, 0],
        [0.0, 0, 5.1890001297]]

positions = [(0.6666669, 0.3333334, 0.0000000),
             (0.3333331, 0.6666663, 0.5000000),
             (0.6666669, 0.3333334, 0.3750000),
             (0.3333331, 0.6666663, 0.8750000)]

symbols = names = ['Ga', 'Ga', 'N', 'N']

structure = DataFactory('structure')(cell=cell)
for position, symbol, name in zip(positions, symbols, names):
    position = np.dot(position, cell).tolist()
    structure.append_atom(
        position=position, symbols=symbol, name=name)

structure
```

```console
<StructureData: uuid: 96f9c02b-77c7-4889-9de2-dbda27bb03fa (unstored)>
```

### Potential Setup

```python
pair_style = 'tersoff'
potential_dict = {
    'Ga Ga Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199',
    'N  N  N': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77',
    'Ga Ga N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
    'Ga N  N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
    'N  Ga Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
    'N  Ga N ': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000',
    'N  N  Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
    'Ga N  Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000'}
potential = DataFactory("lammps.potential")(
    structure=structure, type=pair_style, data=potential_dict
)
potential.attributes
```

```python
{'kind_elements': ['Ga', 'N'],
 'potential_type': 'tersoff',
 'atom_style': 'atomic',
 'default_units': 'metal',
 'potential_md5': 'b3b7d45ae7b92eba05ed99ffe69810d0',
 'input_lines_md5': '3145644a408a6d464e80866b833115a2'}
```

### Force Calculation

```python
from aiida.engine import run_get_node
parameters = DataFactory('dict')(dict={
    'lammps_version': lammps_version(),
    'output_variables': ["temp", "etotal", "pe", "ke"],
    'thermo_keywords': []
})
builder = code_lammps_force.get_builder()
builder.metadata.options = meta_options
builder.structure = structure
builder.potential = potential
builder.parameters = parameters
result, calc_node = run_get_node(builder)
```

```console
$ verdi process list -D desc -a -l 1
  PK  Created    Process label     Process State    Process status
----  ---------  ----------------  ---------------  ----------------
2480  32s ago    ForceCalculation  Finished [0]

Total results: 1

Info: last time an entry changed state: 28s ago (at 02:02:36 on 2019-06-21)

$ verdi process show 2480
Property       Value
-------------  ------------------------------------
type           CalcJobNode
pk             2480
uuid           c754f044-b190-4505-b121-776b79d2d1c8
label
description
ctime          2019-06-21 02:02:32.894858+00:00
mtime          2019-06-21 02:02:33.297377+00:00
process state  Finished
exit status    0
computer       [2] localhost

Inputs        PK  Type
----------  ----  ------------------
code        1351  Code
parameters  2479  Dict
potential   2478  EmpiricalPotential
structure   2477  StructureData

Outputs          PK  Type
-------------  ----  ----------
arrays         2483  ArrayData
remote_folder  2481  RemoteData
results        2484  Dict
retrieved      2482  FolderData
```

```python
calc_node.outputs.results.attributes
```

```python
{'parser_version': '0.4.0b3',
 'parser_class': 'ForceParser',
 'errors': [],
 'warnings': '',
 'distance_units': 'Angstroms',
 'force_units': 'eV/Angstrom',
 'energy_units': 'eV',
 'energy': -18.1098859130104,
 'final_variables': {'ke': 0.0,
  'pe': -18.1098859130104,
  'etotal': -18.1098859130104,
  'temp': 0.0},
 'units_style': 'metal'}
```

```python
calc_node.outputs.arrays.attributes
```

```python
{'array|forces': [1, 4, 3]}
```

### Optimisation Calculation

```python
from aiida.engine import run_get_node
parameters = DataFactory('dict')(dict={
    'lammps_version': lammps_version(),
    'output_variables': ["temp", "etotal", "pe", "ke"],
    'thermo_keywords': [],
    'units': 'metal',
    'relax': {
        'type': 'iso',
        'pressure': 0.0,
        'vmax': 0.001,
    },
    "minimize": {
        'style': 'cg',
        'energy_tolerance': 1.0e-25,
        'force_tolerance': 1.0e-25,
        'max_evaluations': 100000,
        'max_iterations': 50000}
})
builder = code_lammps_opt.get_builder()
builder.metadata.options = meta_options
builder.structure = structure
builder.potential = potential
builder.parameters = parameters
result, calc_node = run_get_node(builder)
```

```console
$ verdi process list -D desc -a -l 1
  PK  Created    Process label        Process State    Process status
----  ---------  -------------------  ---------------  ----------------
2486  1m ago     OptimizeCalculation  ⏹ Finished [0]

Total results: 1

Info: last time an entry changed state: 1m ago (at 02:09:54 on 2019-06-21)

$ verdi process show 2486
Property       Value
-------------  ------------------------------------
type           CalcJobNode
pk             2486
uuid           5c64433d-6337-4352-a0a8-0acb4083a0c3
label
description
ctime          2019-06-21 02:09:50.872336+00:00
mtime          2019-06-21 02:09:51.128639+00:00
process state  Finished
exit status    0
computer       [2] localhost

Inputs        PK  Type
----------  ----  ------------------
code        1344  Code
parameters  2485  Dict
potential   2478  EmpiricalPotential
structure   2477  StructureData

Outputs          PK  Type
-------------  ----  -------------
arrays         2490  ArrayData
remote_folder  2487  RemoteData
results        2491  Dict
retrieved      2488  FolderData
structure      2489  StructureData
```

```python
calc_node.outputs.results.attributes
```

```python
{'parser_version': '0.4.0b3',
 'parser_class': 'OptimizeParser',
 'errors': [],
 'warnings': '',
 'stress_units': 'bars',
 'distance_units': 'Angstroms',
 'force_units': 'eV/Angstrom',
 'energy_units': 'eV',
 'energy': -18.1108516231423,
 'final_variables': {'ke': 0.0,
  'pe': -18.1108516231423,
  'etotal': -18.1108516231423,
  'temp': 0.0},
 'units_style': 'metal'}
```

```python
calc_node.outputs.arrays.attributes
```

```python
{'array|positions': [56, 4, 3],
 'array|stress': [3, 3],
 'array|forces': [56, 4, 3]}
```

### MD Calculation

```python
from aiida.engine import submit
parameters = DataFactory('dict')(dict={
    'lammps_version': lammps_version(),
    'output_variables': ["temp", "etotal", "pe", "ke"],
    'thermo_keywords': [],
    'units': 'metal',
    'timestep': 0.001,
    'integration': {
        'style': 'nvt',
        'constraints': {
            'temp': [300, 300, 0.5]
        }
    },
    "neighbor": [0.3, "bin"],
    "neigh_modify": {"every": 1, "delay": 0, "check": False},
    'equilibrium_steps': 100,
    'total_steps': 1000,
    'dump_rate': 10,
    'restart': 100
})
builder = code_lammps_md.get_builder()
builder.metadata.options = meta_options
builder.structure = structure
builder.potential = potential
builder.parameters = parameters
result, calc_node = run_get_node(builder)
```

```console
$ verdi process list -D desc -a -l 1
  PK  Created    Process label    Process State    Process status
----  ---------  ---------------  ---------------  ----------------
2493  12s ago    MdCalculation    ⏹ Finished [0]

Total results: 1

Info: last time an entry changed state: 4s ago (at 02:15:02 on 2019-06-21)

$ verdi process show 2493
Property       Value
-------------  ------------------------------------
type           CalcJobNode
pk             2493
uuid           351b4721-10ff-406c-8f1c-951317091524
label
description
ctime          2019-06-21 02:14:54.986384+00:00
mtime          2019-06-21 02:14:55.282272+00:00
process state  Finished
exit status    0
computer       [2] localhost

Inputs        PK  Type
----------  ----  ------------------
code        1540  Code
parameters  2492  Dict
potential   2478  EmpiricalPotential
structure   2477  StructureData

Outputs            PK  Type
---------------  ----  --------------
remote_folder    2494  RemoteData
results          2496  Dict
retrieved        2495  FolderData
system_data      2498  ArrayData
trajectory_data  2497  TrajectoryData
```

```python
calc_node.outputs.results.attributes
```

```python
{'parser_version': '0.4.0b3',
 'parser_class': 'MdParser',
 'errors': [],
 'warnings': '',
 'time_units': 'picoseconds',
 'distance_units': 'Angstroms',
 'energy': -17.8464193488116,
 'units_style': 'metal'}
```

```python
calc_node.outputs.system_data.attributes
```

```python
{'units_style': 'metal',
 'array|step': [100],
 'array|ke': [100],
 'array|pe': [100],
 'array|etotal': [100],
 'array|temp': [100]}
```

```python
calc_node.outputs.trajectory_data.attributes
```

```python
{'array|times': [101],
 'array|cells': [101, 3, 3],
 'array|steps': [101],
 'array|positions': [101, 4, 3],
 'symbols': ['Ga', 'Ga', 'N', 'N']}
```
