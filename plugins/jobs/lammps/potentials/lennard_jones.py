import numpy as np


def generate_LAMMPS_potential(data):
    return None


def get_input_potential_lines(data, names=None, potential_filename='potential.pot'):

    cut = np.max([float(i.split()[2]) for i in data.values()])

    lammps_input_text = 'pair_style  lj/cut {}\n'.format(cut)

    for key, value in data.iteritems():
        lammps_input_text += 'pair_coeff {}    {}\n'.format(key, value)
    return lammps_input_text
