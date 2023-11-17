---
myst:
  substitutions:
    pip: '[`pip`](https://pip.pypa.io/en/stable/index.html)'
    PyPI: '[PyPI](https://pypi.org/)'
---

# Get started

(getting_started-requirements)=

## Requirements

To use `aiida-lammps` one should have already:

- installed [aiida-core](https://github.com/aiidateam/aiida-core)
- configured an AiiDA profile.

Information about how to perform these steps can be found in the `aiida-core` [documentation](https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html)

(getting_started-installation)=

## Installation

The package can be installed either from the Python Package Index {{ PyPI }} or from the source.

::::{tab-set}

:::{tab-item} PyPI
Installing via {{ PyPI }} can be done making use of the Python package manager {{ pip }}:

```shell
pip install aiida-lammps
```

This will install the latest release version available in {{ PyPI }}. Other releases can be installed vy explicitly passing the release version number in the installation command.
:::

:::{tab-item} Source
To install from source one needs to clone the repository and then install the package making use of {{ pip }}:

```shell
git clone https://github.com/aiidaplugins/aiida-lammps.git
pip install -e aiida-lammps
```

Notice that the flag ``-e`` means that the package is installed in editable mode, meaning that any changes to the source code will be automatically picked up.
:::

::::

:::{note}
Installing from source in the editable mode is recommended when developing as one would not need to re-install the package every time that a change has been made.
:::


(getting_started-setup)=

## Setup

Setting up `aiida-lammps` to run a [LAMMPS](https://www.lammps.org/#gsc.tab=0) job is done in a similar way as any other [AiiDA plugin](https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/plugins.html). That is a [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html#how-to-set-up-a-computer) and a [Code](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html#how-to-create-a-code) need to be setup for the current profile.

(getting_started-setup-computer)=

### Computer

A [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.html#aiida.orm.computers.Computer) is an aiida data type that contains the information about a compute resource, this could be a local machine, a remote personal computer, an HPC center, etc. Basically it is the machine where [LAMMPS](https://www.lammps.org/#gsc.tab=0) is installed and it will be run.

For the following it will be considered that [LAMMPS](https://www.lammps.org/#gsc.tab=0) is installed in the same machine where the [AiiDA](https://aiida.net/) instance is running, i.e. `localhost`.

The computer setup can be done either via the [AiiDA](https://aiida.net/) command line interface (CLI), [verdi](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/command_line.html), or the Python application programming interface (API).

::::{tab-set}

:::{tab-item} bash
To define the computer via the CLI one needs to run the following command:

```console
verdi computer setup -n --label localhost --hostname localhost --transport core.local --scheduler core.direct --work-dir /home/my_username/aiida_workspace
```

This command will create a [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.html#aiida.orm.computers.Computer) named `localhost`, with the `localhost` hostname, no scheduler is considered and no special transport plugin is needed. All the calculations performed using this computer will be stored in the directory `/home/my_username/aiida_workspace`.

After this its is still necessary to configure the `core.local` transport:

```console
verdi computer configure core.local localhost -n --safe-interval 0 --user my_email@provider.com
```
This configures that the computer will wait 0 seconds between connections attempts (one can increase this to avoid putting pressure in a network) for the user with email `my_email@provider.com`
:::

:::{tab-item} API

Using the Python API one can create a [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.html#aiida.orm.computers.Computer) by running the following code either in the `verdi shell` or using `verdi run` in a script that contains the code

```python
from aiida.orm import Computer
from pathlib import Path

# Create the node with the computer
computer = Computer(
    label='localhost',
    hostname='localhost',
    transport_type='core.local',
    scheduler_type='core.direct',
    workdir=Path('/home/my_username/aiida_workspace').resolve()
)
# Store the node in the database
computer.store()
# Configure the core.local transport
computer.configure()
```
:::
::::

For more detailed information, please refer to the documentation [on setting up compute resources](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html#how-to-set-up-a-computer).

(getting_started-setup-code)=

### Code

Since LAMMPS is a piece of software that is installed in a machine we have to create a [InstalledCode](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.nodes.data.code.html#aiida.orm.nodes.data.code.installed.InstalledCode) node in the database which contains the information about this program. As with the [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.html#aiida.orm.computers.Computer) it is possible to define this both via the [AiiDA](https://aiida.net/) command line interface (CLI), [verdi](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/command_line.html), or the Python application programming interface (API).

In the following it will be assumed that [LAMMPS](https://www.lammps.org/#gsc.tab=0) is installed in the same machine that is running [AiiDA](https://aiida.net/), `localhost`, that the executable is called `lmp` and that there is already a [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.html#aiida.orm.computers.Computer) named `localhost` setup in the database.

::::{tab-set}

:::{tab-item} CLI

To define the [InstalledCode](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.nodes.data.code.html#aiida.orm.nodes.data.code.installed.InstalledCode) which refers to the [LAMMPS](https://www.lammps.org/#gsc.tab=0) installation via the CLI one runs the command:

```console
verdi code create core.code.installed --label lammps  --computer localhost --default-calc-job-plugin lammps.base --filepath-executable /path/to/lammps/lmp
```

This will create an [InstalledCode](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.nodes.data.code.html#aiida.orm.nodes.data.code.installed.InstalledCode) with the name `lammps` associated to the [Computer](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.html#aiida.orm.computers.Computer) named `localhost` whose executable absolute path is `/path/to/lammps/lmp`
:::

:::{tab-item} API
To define the [InstalledCode](https://aiida.readthedocs.io/projects/aiida-core/en/latest/reference/apidoc/aiida.orm.nodes.data.code.html#aiida.orm.nodes.data.code.installed.InstalledCode) which refers to the [LAMMPS](https://www.lammps.org/#gsc.tab=0) installation via the Python API one can use the following code in either the `verdi shell` or by running a script which contains it via `verdi run`:

```python
from aiida.orm import InstalledCode

# Load the computer resource where LAMMPS is installed
computer = load_computer('localhost')

# Define the code node
code = InstalledCode(
    label='lammps',
    computer=computer,
    filepath_executable='/path/to/lammps/lmp',
    default_calc_job_plugin='lammps.base'
)

# Store the code node in the database
code.store()
```
:::

::::

For more detailed information, please refer to the documentation [on setting up codes](https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/run_codes.html#how-to-setup-a-code).
