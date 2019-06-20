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

    # lammps_input_text += "compute reax all pair reax/c\n"
    # lammps_input_text += "variable reax_eb           equal c_reax[1]\n"
    # lammps_input_text += "variable reax_ea           equal c_reax[2]\n"
    # lammps_input_text += "variable reax_elp          equal c_reax[3]\n"
    # lammps_input_text += "variable reax_emol         equal c_reax[4]\n"
    # lammps_input_text += "variable reax_ev           equal c_reax[5]\n"
    # lammps_input_text += "variable reax_epen         equal c_reax[6]\n"
    # lammps_input_text += "variable reax_ecoa         equal c_reax[7]\n"
    # lammps_input_text += "variable reax_ehb          equal c_reax[8]\n"
    # lammps_input_text += "variable reax_et           equal c_reax[9]\n"
    # lammps_input_text += "variable reax_eco          equal c_reax[10]\n"
    # lammps_input_text += "variable reax_ew           equal c_reax[11]\n"
    # lammps_input_text += "variable reax_ep           equal c_reax[12]\n"
    # lammps_input_text += "variable reax_efi          equal c_reax[13]\n"
    # lammps_input_text += "variable reax_eqeq         equal c_reax[14]\n"
    
    # TODO to access these variables, the compute must be triggered,
    # for example by adding c_reax[1] to the thermo_style
    # but how to do this in a generalised manner?

    return lammps_input_text


DEFAULT_UNITS = 'real'
ATOM_STYLE = 'charge'
