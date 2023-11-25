# Developer Guide

This is a guide for internal development of `aiida-lammps`

## Coding Style Requirements

The code is formatted and linted using [pre-commit](https://pre-commit.com/), which runs in an isolated, virtual environment:

```console
pip install pre-commit
pre-commit run --all
```

or to automate runs, triggered before each commit:

```console
pre-commit install
```

To avoid problems arising from different configurations in virtual environments, one can also use [tox](https://tox.wiki/en/latest/index.html) to run the pre-commit command inside a clean virtual environment. This can be done in the following manner

```console
pip install tox
pip install -e .[pre-commit]
tox -e pre-commit
```


## Testing

The test suite can be run in an isolated, virtual environment using `tox` (see `[tool.tox]` in `pyproject.toml`):

```console
pip install tox
tox -e 3.8-aiida_lammps
```

or directly:

```console
pip install .[tests]
pytest -v
```

The tests require that both PostgreSQL and RabbitMQ are running.
If you wish to run an isolated RabbitMQ instance, see the `docker-compose.yml` file in the repo.

Some tests require that a `lammps` executable be present.

The easiest way to achieve this is to use Conda:

```console
conda install lammps==2019.06.05
# this will install lmp_serial and lmp_mpi
```

You can specify a different executable name for LAMMPS with:

```console
tox -e 3.8-aiida_lammps -- --lammps-exec lmp_exec
```

To output the results of calcjob executions to a specific directory:

```console
pytest --lammps-workdir "test_workdir"
```

## Documentation

To run a full docs build:

```console
tox -e docs-clean
```

or to re-build from the current documentation:

```console
$ tox -e docs-update
```
