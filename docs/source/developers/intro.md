# Developer Guide

This is a guide for internal develoment of `aiida-lammps`

## Coding Style Requirements

The code is formatted and linted using [pre-commit](https://pre-commit.com/), which runs in an isolated, virtual environment:

```shell
>> pip install pre-commit
>> pre-commit run --all
```

or to automate runs, triggered before each commit:

```shell
>> pre-commit install
```

## Testing

The test suite can be run in an isolated, virtual environment using `tox` (see `tox.ini` in the repo):

```shell
>> pip install tox
>> tox -e py37
```

or directly:

```shell
>> pip install -e .[testing]
>> reentry scan -r aiida
>> pytest -v
```

The tests require that both PostgreSQL and RabbitMQ are running.
If you wish to run an isolated RabbitMQ instance, see the `docker-compose.yml` file in the repo.

Some tests require that a `lammps` executable be present.

The easiest way to achieve this is to use Conda:

```shell
>> conda install lammps==2019.06.05
# this will install lmp_serial and lmp_mpi
```

You can specify a different executable name for LAMMPS with:

```shell
>> tox -e py37 -- --lammps-exec lmp_exec
```

To output the results of calcjob executions to a specific directory:

```shell
>> pytest --lammps-workdir "test_workdir"
```
