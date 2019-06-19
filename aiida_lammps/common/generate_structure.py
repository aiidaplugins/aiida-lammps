""" creation of the structure file content

The code here is largely adapted from https://github.com/andeplane/cif2cell-lammps
ESPInterfaces.LAMMPSFile
"""
from math import cos, sin, atan2
import ase
import numpy as np


def get_vector(value):
    return [float(v) for v in value]


def get_matrix(value):
    return [[float(v) for v in vec] for vec in value]


def mvmult3(mat, vec):
    """ matrix-vector multiplication """
    w = [0., 0., 0.]
    for i in range(3):
        t = 0
        for j in range(3):
            t = t + mat[j][i] * vec[j]
        w[i] = t
    return w


def cartesian_to_frac(lattice, ccoords):
    """convert cartesian coordinate to fractional

    Parameters
    ----------
    lattice: list
        3x3 array of lattice vectors
    ccoord: list
        Nx3 cartesian coordinates

    Returns
    -------
    list:
        Nx3 array of fractional coordinate

    """
    det3 = np.linalg.det
    latt_tr = np.transpose(lattice)
    det_latt_tr = np.linalg.det(latt_tr)

    fcoords = []
    for ccoord in ccoords:
        a = (det3([[ccoord[0], latt_tr[0][1], latt_tr[0][2]],
                   [ccoord[1], latt_tr[1][1], latt_tr[1][2]],
                   [ccoord[2], latt_tr[2][1], latt_tr[2][2]]])) / det_latt_tr
        b = (det3([[latt_tr[0][0], ccoord[0], latt_tr[0][2]],
                   [latt_tr[1][0], ccoord[1], latt_tr[1][2]],
                   [latt_tr[2][0], ccoord[2], latt_tr[2][2]]])) / det_latt_tr
        c = (det3([[latt_tr[0][0], latt_tr[0][1], ccoord[0]],
                   [latt_tr[1][0], latt_tr[1][1], ccoord[1]],
                   [latt_tr[2][0], latt_tr[2][1], ccoord[2]]])) / det_latt_tr

        fcoords.append([a, b, c])

    return fcoords


def is_not_zero(value):
    return not np.isclose(value, 0)


def round_by(value, round_dp):
    if round_dp is None:
        return value
    return round(value, round_dp)


class AtomSite(object):
    def __init__(self, kind_name, cartesian, fractional=None):
        self.kind_name = kind_name
        self.cartesian = cartesian
        self.fractional = fractional


def generate_lammps_structure(structure,
                              atom_style='atomic', charge_dict=None,
                              round_dp=None,
                              docstring="generated by aiida_lammps"):
    """create lammps input structure file content

    Parameters
    ----------
    structure: StructureData
        the structure to use
    atom_style: str
        the atomic style
    charge_dict: dict
        mapping of atom kind_name to charge
    round_dp: None or int
        round output values to a number of decimal places (used for testing)
    docstring : str
        docstring to put at top of file

    """
    if atom_style not in ['atomic', 'charge']:
        raise ValueError("atom_style must be in ['atomic', 'charge']")
    if charge_dict is None:
        charge_dict = {}

    atom_sites = [AtomSite(site.kind_name, site.position)
                  for site in structure.sites]
    # mapping of atom kind_name to id number
    kind_name_id_map = {}
    for site in atom_sites:
        if site.kind_name not in kind_name_id_map:
            kind_name_id_map[site.kind_name] = len(kind_name_id_map) + 1
    # mapping of atom kind_name to mass
    kind_mass_dict = {kind.name: kind.mass for kind in structure.kinds}

    filestring = ""
    filestring += "# {}\n\n".format(docstring)
    filestring += "{0} atoms\n".format(len(atom_sites))
    filestring += "{0} atom types\n\n".format(len(kind_name_id_map))

    lattice = get_matrix(structure.cell)

    # As per https://lammps.sandia.gov/doc/Howto_triclinic.html,
    # if the lattice does not conform to a regular parallelpiped
    # then it must first be rotated

    if is_not_zero(lattice[0][1]) or is_not_zero(lattice[0][2]) or is_not_zero(lattice[1][2]):
        rotated_cell = True
        for site in atom_sites:
            site.fractional = cartesian_to_frac(lattice, [site.cartesian])[0]
        # creating the cell from its lengths and angles,
        # generally ensures that it is in a compatible orientation
        atoms = ase.Atoms(cell=structure.cell_lengths + structure.cell_angles)
        lattice = get_matrix(atoms.cell)
    else:
        rotated_cell = False

    if is_not_zero(lattice[0][1]):
        theta = atan2(-lattice[0][1], lattice[0][0])
        rot_matrix = get_matrix([
            [cos(theta), sin(theta), 0],
            [-sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        lattice[0] = get_vector(mvmult3(rot_matrix, lattice[0]))
        lattice[1] = get_vector(mvmult3(rot_matrix, lattice[1]))
        lattice[2] = get_vector(mvmult3(rot_matrix, lattice[2]))

    if is_not_zero(lattice[0][2]):
        theta = atan2(-lattice[0][2], lattice[0][0])
        rot_matrix = get_matrix([
            [cos(theta), sin(theta), 0],
            [0, 1, 0],
            [-sin(theta), cos(theta), 0]
        ])
        lattice[0] = get_vector(mvmult3(rot_matrix, lattice[0]))
        lattice[1] = get_vector(mvmult3(rot_matrix, lattice[1]))
        lattice[2] = get_vector(mvmult3(rot_matrix, lattice[2]))

    if is_not_zero(lattice[1][2]):
        theta = atan2(-lattice[1][2], lattice[1][1])
        rot_matrix = get_matrix([
            [1, 0, 0],
            [0, cos(theta), sin(theta)],
            [0, -sin(theta), cos(theta)]
        ])
        lattice[0] = get_vector(mvmult3(rot_matrix, lattice[0]))
        lattice[1] = get_vector(mvmult3(rot_matrix, lattice[1]))
        lattice[2] = get_vector(mvmult3(rot_matrix, lattice[2]))

    if is_not_zero(lattice[0][1]) or is_not_zero(lattice[0][2]) or is_not_zero(lattice[1][2]) or lattice[0][0] < 1e-9 or lattice[1][1] < 1e-9 or lattice[2][2] < 1e-9:
        raise ValueError(
            "Error in triclinic box: {}\n"
            "Vectors should follow these rules: "
            "https://lammps.sandia.gov/doc/Howto_triclinic.html".format(lattice))

    a = round_by(lattice[0][0], round_dp)
    b = round_by(lattice[1][1], round_dp)
    c = round_by(lattice[2][2], round_dp)

    filestring += "0.0 {0:20.10f} xlo xhi\n".format(a)
    filestring += "0.0 {0:20.10f} ylo yhi\n".format(b)
    filestring += "0.0 {0:20.10f} zlo zhi\n".format(c)

    xy = round_by(lattice[1][0], round_dp)
    xz = round_by(lattice[2][0], round_dp)
    yz = round_by(lattice[2][1], round_dp)

    if is_not_zero(xy) or is_not_zero(xz) or is_not_zero(yz):
        filestring += "{0:20.10f} {1:20.10f} {2:20.10f} xy xz yz\n\n".format(
            xy, xz, yz)

    filestring += 'Masses\n\n'
    for kind_name, kind_id in kind_name_id_map.items():
        filestring += '{0} {1:20.10f} \n'.format(
            kind_id, kind_mass_dict[kind_name])
    filestring += "\n"

    filestring += "Atoms\n\n"

    for site_index, site in enumerate(atom_sites):
        if rotated_cell:
            pos = get_vector(mvmult3(lattice, site.fractional))
        else:
            pos = site.cartesian
        pos = [round_by(v, round_dp) for v in pos]

        kind_id = kind_name_id_map[site.kind_name]

        if atom_style == 'atomic':
            filestring += "{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n".format(
                site_index + 1, kind_id, pos[0], pos[1], pos[2])
        elif atom_style == 'charge':
            charge = charge_dict.get(site.kind_name, 0.0)
            filestring += "{0} {1} {2} {3:20.10f} {4:20.10f} {5:20.10f}\n".format(
                site_index + 1, kind_id, charge, pos[0], pos[1], pos[2])
        else:
            raise ValueError('atom_style')

    return filestring


def old_generate_lammps_structure(structure, atom_style):
    """ this is the deprecated method, used before 0.3.0b3,
    stored here for prosperity.

    This method can create erroneous structures for triclinic cells
    """
    import numpy as np

    types = [site.kind_name for site in structure.sites]

    type_index_unique = np.unique(types, return_index=True)[1]
    count_index_unique = np.diff(np.append(type_index_unique, [len(types)]))

    atom_index = []
    for i, index in enumerate(count_index_unique):
        atom_index += [i for j in range(index)]

    masses = [site.mass for site in structure.kinds]
    positions = [site.position for site in structure.sites]

    number_of_atoms = len(positions)

    lammps_data_file = 'Generated using dynaphopy\n\n'
    lammps_data_file += '{0} atoms\n\n'.format(number_of_atoms)
    lammps_data_file += '{0} atom types\n\n'.format(len(masses))

    cell = np.array(structure.cell)

    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])

    alpha = np.arccos(np.dot(cell[1], cell[2]) / (c * b))
    gamma = np.arccos(np.dot(cell[1], cell[0]) / (a * b))
    beta = np.arccos(np.dot(cell[2], cell[0]) / (a * c))

    xhi = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    yhi = np.sqrt(pow(b, 2) - pow(xy, 2))
    yz = (b * c * np.cos(alpha) - xy * xz) / yhi
    zhi = np.sqrt(pow(c, 2) - pow(xz, 2) - pow(yz, 2))

    xhi = xhi + max(0, 0, xy, xz, xy + xz)
    yhi = yhi + max(0, 0, yz)

    lammps_data_file += '\n{0:20.10f} {1:20.10f} xlo xhi\n'.format(0, xhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} ylo yhi\n'.format(0, yhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} zlo zhi\n'.format(0, zhi)
    lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f} xy xz yz\n\n'.format(
        xy, xz, yz)

    lammps_data_file += 'Masses\n\n'

    for i, mass in enumerate(masses):
        lammps_data_file += '{0} {1:20.10f} \n'.format(i + 1, mass)

    lammps_data_file += '\nAtoms\n\n'
    for i, row in enumerate(positions):
        if atom_style == 'charge':
            lammps_data_file += '{0} {1} 0.0 {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(
                i + 1, atom_index[i] + 1, row[0], row[1], row[2])
        else:
            lammps_data_file += '{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(
                i + 1, atom_index[i] + 1, row[0], row[1], row[2])

    return lammps_data_file
