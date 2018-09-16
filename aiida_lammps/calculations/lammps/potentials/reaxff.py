
def generate_LAMMPS_potential(data):

    potential_file = ''
    for line in data['file_contents']:
        potential_file += '{}'.format(line)

    return potential_file


def get_input_potential_lines(data, names=None, potential_filename='potential.pot'):

    lammps_input_text = 'pair_style reax/c NULL '
    if 'safezone' in data:
        lammps_input_text += 'safezone {0} '.format(data['safezone'])
    lammps_input_text += "\n"
    lammps_input_text += 'pair_coeff      * * {} {}\n'.format(potential_filename, ' '.join(names))
    if "neighbor_bin" in data:
        lammps_input_text += "neighbor {0} bin\n".format(data["neighbor_bin"])
    if "neighbor_list" in data:
        lammps_input_text += "neigh_modify delay {} check yes\n".format(data["neighbor_list"])
    lammps_input_text += "fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
    lammps_input_text += "fix_modify qeq energy yes\n"

    return lammps_input_text


DEFAULT_UNITS = 'real'
ATOM_STYLE = 'charge'
