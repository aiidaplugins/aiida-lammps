[![CI Status](https://github.com/aiidaplugins/aiida-lammps/workflows/CI/badge.svg)](https://github.com/aiidaplugins/aiida-lammps)
[![Coverage Status](https://codecov.io/gh/aiidaplugins/aiida-lammps/branch/master/graph/badge.svg)](https://codecov.io/gh/aiidaplugins/aiida-lammps)
[![PyPI](https://img.shields.io/pypi/v/aiida-lammps.svg)](https://pypi.python.org/pypi/aiida-lammps/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Docs status](https://readthedocs.org/projects/aiida-lammps/badge)](http://aiida-lammps.readthedocs.io/)

# AiiDA LAMMPS plugin

An [AiiDA](http://aiida-core.readthedocs.io/) plugin for the classical molecular dynamics code [LAMMPS](https://www.lammps.org).

This plugin contains 2 types of calculations:

- `lammps.base`: Calculation making use of parameter based input generation for single stage LAMMPS calculations.
- `lammps.raw`: Calculation making use of a pre-made LAMMPS input file.

The `lammps.base` is also used to handle three workflows:

- `lammps.base`: A workflow that can be used to submit any single stage LAMMPS calculation.
- `lammps.relax`: A workflow to submit a structural relaxation using LAMMPS.
- `lammps.md`: A workflow to submit a molecular dynamics calculation using LAMMPS.

- [AiiDA LAMMPS plugin](#aiida-lammps-plugin)
  - [Installation](#installation)
  - [Built-in Potential Support](#built-in-potential-support)
  - [Examples](#examples)
    - [Code Setup](#code-setup)
    - [Structure Setup](#structure-setup)
    - [Potential Setup](#potential-setup)
    - [Force Calculation](#force-calculation)
    - [Optimisation Calculation](#optimisation-calculation)
    - [MD Calculation](#md-calculation)
  - [Development](#development)
    - [Coding Style Requirements](#coding-style-requirements)
    - [Testing](#testing)

## Installation

To install a stable version from pypi:

```shell
pip install aiida-lammps
```

To install from source:

```shell
git clone https://github.com/aiidaplugins/aiida-lammps.git
pip install -e aiida-lammps
```

## Built-in Potential Support

The `lammps.base` calculation and associated workflows make use of the ``LammpsPotentialData`` data structure which is created by passing a potential file, plus some labelling parameters to it.

This data structure can be used to handle the following potential types:

- Single file potentials: Any potential that can be stored in a single file, e.g. [EAM](https://docs.lammps.org/pair_eam.html), [MEAM](https://docs.lammps.org/pair_meam.html), [Tersoff](https://docs.lammps.org/pair_tersoff.html) and [ReaxFF](https://docs.lammps.org/pair_reaxff.html).
- Directly parametrized potentials: Potentials whose parameters are directly given via ``pair_coeff`` in the input file, e.g [Born](https://docs.lammps.org/pair_born_gauss.html), [Lennard-Jones](https://docs.lammps.org/pair_line_lj.html) and [Yukawa](https://docs.lammps.org/pair_yukawa.html). These parameters should be written into a file that is then stored into a ``LammpsPotentialData`` node.



## Examples

More example calculations are found in the folder **/examples** as well as in the documentation. The examples touch some common cases for the usage of LAMMPS for a single stage calculation.

## Development

### Running tests

The test suite can be run in an isolated, virtual environment using `tox` (see `tox.ini` in the repo):

```shell
pip install tox
tox -e 3.9-aiida_lammps -- tests/
```

or directly:

```shell
pip install .[testing]
pytest -v
```

The tests require that both PostgreSQL and RabbitMQ are running.
If you wish to run an isolated RabbitMQ instance, see the `docker-compose.yml` file in the repo.

Some tests require that a `lammps` executable be present.

The easiest way to achieve this is to use Conda:

```shell
conda install lammps==2019.06.05
# this will install lmp_serial and lmp_mpi
```

You can specify a different executable name for LAMMPS with:

```shell
tox -e 3.9-aiida_lammps -- --lammps-exec lmp_exec
```

To output the results of calcjob executions to a specific directory:

```shell
pytest --lammps-workdir "test_workdir"
```

### Pre-commit

The code is formatted and linted using [pre-commit](https://pre-commit.com/), so that the code conform to the standard:

```shell
cd aiida-lammps
pre-commit run --all
```
or to automate runs, triggered before each commit:

```shell
pre-commit install
```

## License

The `aiida-lammps` plugin package is released under the MIT license. See the `LICENSE` file for more details.
