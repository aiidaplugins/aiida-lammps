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


def get_input_potential_lines(data, kind_elements=None, potential_filename='potential.pot'):

    lammps_input_text = 'pair_style reax/c NULL '
    if 'safezone' in data:
        lammps_input_text += 'safezone {0} '.format(data['safezone'])
    lammps_input_text += "\n"
    lammps_input_text += 'pair_coeff      * * {} {}\n'.format(potential_filename, ' '.join(kind_elements))
    lammps_input_text += "fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
    lammps_input_text += "fix_modify qeq energy yes\n"
    lammps_input_text += "compute reax all pair reax/c\n"

    # NOTE: to use c_reax[1] - c_reax[14],
    # c_reax[1] must be added to `thermo_style custom`, to trigger the compute

    # The array values correspond to:
    # eb = bond energy
    # ea = atom energy
    # elp = lone-pair energy
    # emol = molecule energy (always 0.0)
    # ev = valence angle energy
    # epen = double-bond valence angle penalty
    # ecoa = valence angle conjugation energy
    # ehb = hydrogen bond energy
    # et = torsion energy
    # eco = conjugation energy
    # ew = van der Waals energy
    # ep = Coulomb energy
    # efi = electric field energy (always 0.0)
    # eqeq = charge equilibration energy

    return lammps_input_text


DEFAULT_UNITS = 'real'
ATOM_STYLE = 'charge'
