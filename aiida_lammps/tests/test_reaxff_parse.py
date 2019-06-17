import os
import six

from aiida_lammps.tests.utils import TEST_DIR
from aiida_lammps.common.reaxff_convert import read_reaxff_file, write_gulp, write_lammps


def test_read_reaxff_file(data_regression):

    fpath = os.path.join(TEST_DIR, 'input_files', 'FeCrOSCH.reaxff')
    data = read_reaxff_file(fpath)

    data_regression.check(data)


def test_write_gulp(file_regression):

    fpath = os.path.join(TEST_DIR, 'input_files', 'FeCrOSCH.reaxff')
    data = read_reaxff_file(fpath)
    contents = write_gulp(data, species_filter=['Fe', 'S'])
    file_regression.check(six.ensure_text(contents))


def test_write_lammps(file_regression):

    fpath = os.path.join(TEST_DIR, 'input_files', 'FeCrOSCH.reaxff')
    data = read_reaxff_file(fpath)
    contents = write_lammps(data)
    file_regression.check(six.ensure_text(contents))
