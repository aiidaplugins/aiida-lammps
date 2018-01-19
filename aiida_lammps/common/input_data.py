import numpy as np


def get_trajectory_txt(trajectory):

    cell = trajectory.get_cells()[0]

    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])

    alpha = np.arccos(np.dot(cell[1], cell[2])/(c*b))
    gamma = np.arccos(np.dot(cell[1], cell[0])/(a*b))
    beta = np.arccos(np.dot(cell[2], cell[0])/(a*c))

    xhi = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    yhi = np.sqrt(pow(b,2)- pow(xy,2))
    yz = (b*c*np.cos(alpha)-xy * xz)/yhi
    zhi = np.sqrt(pow(c,2)-pow(xz,2)-pow(yz,2))

    xhi = xhi + max(0,0, xy, xz, xy+xz)
    yhi = yhi + max(0,0, yz)

    xlo_bound = np.min([0.0, xy, xz, xy+xz])
    xhi_bound = xhi + np.max([0.0, xy, xz, xy+xz])
    ylo_bound = np.min([0.0, yz])
    yhi_bound = yhi + np.max([0.0, yz])
    zlo_bound = 0
    zhi_bound = zhi

    ind = trajectory.get_array('steps')
    lammps_data_file = ''
    for i, position_step in enumerate(trajectory.get_positions()):
        lammps_data_file += 'ITEM: TIMESTEP\n'
        lammps_data_file += '{}\n'.format(ind[i])
        lammps_data_file += 'ITEM: NUMBER OF ATOMS\n'
        lammps_data_file += '{}\n'.format(len(position_step))
        lammps_data_file += 'ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(xlo_bound, xhi_bound, xy)
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(ylo_bound, yhi_bound, xz)
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(zlo_bound, zhi_bound, yz)
        lammps_data_file += ('ITEM: ATOMS x y z\n')
        for position in position_step:
            lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*position)
    return lammps_data_file


def parameters_to_input_file(parameters_object):

    parameters = parameters_object.get_dict()
    input_file = ('STRUCTURE FILE POSCAR\nPOSCAR\n\n')
    input_file += ('FORCE CONSTANTS\nFORCE_CONSTANTS\n\n')
    input_file += ('PRIMITIVE MATRIX\n')
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[2])
    input_file += ('\n')
    input_file += ('SUPERCELL MATRIX PHONOPY\n')
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[2])
    input_file += ('\n')

    return input_file
