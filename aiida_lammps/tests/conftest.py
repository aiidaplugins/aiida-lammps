"""
initialise a text database and profile
"""
from collections import namedtuple
import io
import os
import shutil
import tempfile

from aiida.manage.fixtures import fixture_manager
from aiida.plugins import DataFactory
import numpy as np
import pytest

from aiida_lammps.tests.utils import AiidaTestApp, TEST_DIR


@pytest.fixture(scope='session')
def aiida_environment():
    """setup a test profile for the duration of the tests"""
    # TODO this is required locally for click
    # (see https://click.palletsprojects.com/en/7.x/python3/)
    os.environ["LC_ALL"] = "en_US.UTF-8"
    with fixture_manager() as fixture_mgr:
        yield fixture_mgr


@pytest.fixture(scope='function')
def db_test_app(aiida_environment):
    """clear the database after each test"""

    executables = {
        'lammps.md': 'lammps',
        'lammps.optimize': 'lammps',
        'lammps.force': 'lammps',
        'lammps.combinate': 'lammps'
    }

    work_directory = tempfile.mkdtemp()
    yield AiidaTestApp(
        work_directory, executables, environment=aiida_environment)
    aiida_environment.reset_db()
    shutil.rmtree(work_directory)


potential_data = namedtuple(
    'PotentialTestData', ['potential', 'structure', 'units', 'input_files', 'output'])


@pytest.fixture(scope='function')
def get_potential_data():
    def _get_potential_data(pkey):
        """ return data to create a potential,
        and accompanying structure data and expected output data to test it with
        """
        if pkey == "eam":
            pair_style = 'eam'
            with io.open(os.path.join(TEST_DIR, 'input_files', 'Fe_mm.eam.fs')) as handle:
                potential_dict = {
                    'type': 'fs',
                    'file_contents': handle.readlines()
                }
            cell = [[2.848116, 0.000000, 0.000000],
                    [0.000000, 2.848116, 0.000000],
                    [0.000000, 0.000000, 2.848116]]

            scaled_positions = [(0.0000000, 0.0000000, 0.0000000),
                                (0.5000000, 0.5000000, 0.5000000)]

            symbols = ['Fe', 'Fe']

            input_files = ['input.data', 'input.in', 'potential.pot']

            output_dict = {
                "energy": -8.2448702,
                "warnings": ''
            }

        elif pkey == "lennard-jones":

            cell = [[3.987594, 0.000000, 0.000000],
                    [-1.993797, 3.453358, 0.000000],
                    [0.000000, 0.000000, 6.538394]]

            symbols = ['Ar'] * 2
            scaled_positions = [(0.33333, 0.66666, 0.25000),
                                (0.66667, 0.33333, 0.75000)]

            # Example LJ parameters for Argon. These may not be accurate at all
            pair_style = 'lennard_jones'
            potential_dict = {
                '1  1': '0.01029   3.4    2.5',
                # '2  2':   '1.0      1.0    2.5',
                # '1  2':   '1.0      1.0    2.5'
            }

            input_files = ['input.data', 'input.in']

            output_dict = {
                "energy": 0.0,  # TODO should LJ energy be 0?
                "warnings": ''
            }

        elif pkey == "tersoff":

            cell = [[3.1900000572, 0, 0],
                    [-1.5950000286, 2.762621076, 0],
                    [0.0, 0, 5.1890001297]]

            scaled_positions = [(0.6666669, 0.3333334, 0.0000000),
                                (0.3333331, 0.6666663, 0.5000000),
                                (0.6666669, 0.3333334, 0.3750000),
                                (0.3333331, 0.6666663, 0.8750000)]

            symbols = ['Ga', 'Ga', 'N', 'N']

            potential_dict = {
                'Ga Ga Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 1.0 1.44970 410.132 2.87 0.15 1.60916 535.199',
                'N  N  N': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 1.0 2.38426 423.769 2.20 0.20 3.55779 1044.77',
                'Ga Ga N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
                'Ga N  N': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
                'N  Ga Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 1.0 2.63906 3864.27 2.90 0.20 2.93516 6136.44',
                'N  Ga N ': '1.0 0.766120 0.000 0.178493 0.20172 -0.045238 1.0 0.0 0.00000 0.00000 2.20 0.20 0.00000 0.00000',
                'N  N  Ga': '1.0 0.001632 0.000 65.20700 2.82100 -0.518000 1.0 0.0 0.00000 0.00000 2.90 0.20 0.00000 0.00000',
                'Ga N  Ga': '1.0 0.007874 1.846 1.918000 0.75000 -0.301300 1.0 0.0 0.00000 0.00000 2.87 0.15 0.00000 0.00000'}

            pair_style = 'tersoff'

            input_files = ['input.data', 'input.in', 'potential.pot']

            output_dict = {
                "energy": -18.110852,
                "warnings": ''
            }

        elif pkey == "reaxff":

            pair_style = 'reaxff'
            with io.open(os.path.join(TEST_DIR, 'input_files', 'FeCrOSCH.reaxff')) as handle:
                potential_dict = {
                    'file_contents': handle.readlines(),
                    # 'data': read_reaxff_file(reaxff_path)
                    'safezone': 1.6,
                }

            # pyrite
            cell = [[5.38, 0.000000, 0.000000],
                    [0.000000, 5.38, 0.000000],
                    [0.000000, 0.000000, 5.38]]

            scaled_positions = [[0.0, 0.0, 0.0],
                                [0.5, 0.0, 0.5],
                                [0.0, 0.5, 0.5],
                                [0.5, 0.5, 0.0],
                                [0.338, 0.338, 0.338],
                                [0.662, 0.662, 0.662],
                                [0.162, 0.662, 0.838],
                                [0.838, 0.338, 0.162],
                                [0.662, 0.838, 0.162],
                                [0.338, 0.162, 0.838],
                                [0.838, 0.162, 0.662],
                                [0.162, 0.838, 0.338]]

            symbols = ['Fe'] * 4 + ['S'] * 8

            input_files = ['input.data', 'input.in', 'potential.pot']

            output_dict = {
                "energy": -1030.3543,
                "units": "real",
                "warnings": 'Warning: changed valency_val to valency_boc for X'
            }

        else:
            raise ValueError('Unknown potential key: {}'.format(pkey))

        # create structure
        structure = DataFactory('structure')(cell=cell)
        for scaled_position, symbols in zip(scaled_positions, symbols):
            structure.append_atom(
                position=np.dot(scaled_position, cell).tolist(),
                symbols=symbols)

        # create potential from dict
        potential = DataFactory("lammps.potential")(
            structure=structure, type=pair_style, data=potential_dict
        )

        return potential_data(potential, structure, potential.default_units, input_files, output_dict)

    return _get_potential_data
