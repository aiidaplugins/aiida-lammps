from aiida_lammps.common.reaxff_convert import write_lammps
from aiida_lammps.validation import validate_with_json


def generate_LAMMPS_potential(data):

    if 'file_contents' in data:
        potential_file = ''
        for line in data['file_contents']:
            potential_file += '{}'.format(line)
    else:
        validate_with_json(data, 'reaxff')
        potential_file = write_lammps(data)

    return potential_file


def get_input_potential_lines(data, names=None, potential_filename='potential.pot'):

    lammps_input_text = 'pair_style reax/c NULL '
    if 'safezone' in data:
        lammps_input_text += 'safezone {0} '.format(data['safezone'])
    lammps_input_text += "\n"
    lammps_input_text += 'pair_coeff      * * {} {}\n'.format(potential_filename, ' '.join(names))
    lammps_input_text += "fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
    lammps_input_text += "fix_modify qeq energy yes\n"

    return lammps_input_text


DEFAULT_UNITS = 'real'
ATOM_STYLE = 'charge'
