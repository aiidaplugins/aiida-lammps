"""Set of functions to generate dynaphopy compatible input files."""
import numpy as np


def get_trajectory_txt(trajectory) -> str:
    """Generate a trajectory file to be given to dynaphopy.

    :param trajectory: lammps trajectories
    :type trajectory: LammpsTrajectory
    :return: trajectory string for dynaphopy
    :rtype: str
    """
    # pylint: disable=too-many-locals
    cell = trajectory.get_cells()[0]

    alat = np.linalg.norm(cell[0])
    blat = np.linalg.norm(cell[1])
    clat = np.linalg.norm(cell[2])

    alpha = np.arccos(np.dot(cell[1], cell[2]) / (clat * blat))
    gamma = np.arccos(np.dot(cell[1], cell[0]) / (alat * blat))
    beta = np.arccos(np.dot(cell[2], cell[0]) / (alat * clat))

    xhi = alat
    xy_box = blat * np.cos(gamma)
    xz_box = clat * np.cos(beta)
    yhi = np.sqrt(pow(blat, 2) - pow(xy_box, 2))
    yz_box = (blat * clat * np.cos(alpha) - xy_box * xz_box) / yhi
    zhi = np.sqrt(pow(clat, 2) - pow(xz_box, 2) - pow(yz_box, 2))

    xhi = xhi + max(0, 0, xy_box, xz_box, xy_box + xz_box)
    yhi = yhi + max(0, 0, yz_box)

    xlo_bound = np.min([0.0, xy_box, xz_box, xy_box + xz_box])
    xhi_bound = xhi + np.max([0.0, xy_box, xz_box, xy_box + xz_box])
    ylo_bound = np.min([0.0, yz_box])
    yhi_bound = yhi + np.max([0.0, yz_box])
    zlo_bound = 0
    zhi_bound = zhi

    ind = trajectory.get_array('steps')
    lammps_data_file = ''
    for i, position_step in enumerate(trajectory.get_positions()):
        lammps_data_file += 'ITEM: TIMESTEP\n'
        lammps_data_file += f'{ind[i]}\n'
        lammps_data_file += 'ITEM: NUMBER OF ATOMS\n'
        lammps_data_file += f'{len(position_step)}\n'
        lammps_data_file += 'ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
        lammps_data_file += f'{xlo_bound:20.10f} {xhi_bound:20.10f} {xy_box:20.10f}\n'
        lammps_data_file += f'{ylo_bound:20.10f} {yhi_bound:20.10f} {yz_box:20.10f}\n'
        lammps_data_file += f'{zlo_bound:20.10f} {zhi_bound:20.10f} {yz_box:20.10f}\n'
        lammps_data_file += 'ITEM: ATOMS x y z\n'
        for position in position_step:
            lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(
                *position)
    return lammps_data_file


def parameters_to_input_file(parameters_object: dict) -> str:
    """Generate the input file for dynaphopy.

    :param parameters_object: dictionary with inputs for dynaphopy.
    :type parameters_object: dict
    :return: dynaphopy string input
    :rtype: str
    """
    parameters = parameters_object.get_dict()
    input_file = 'STRUCTURE FILE POSCAR\nPOSCAR\n\n'
    input_file += 'FORCE CONSTANTS\nFORCE_CONSTANTS\n\n'
    input_file += 'PRIMITIVE MATRIX\n'
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['primitive'])[2])
    input_file += '\n'
    input_file += 'SUPERCELL MATRIX PHONOPY\n'
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[0])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[1])
    input_file += ('{} {} {} \n').format(*np.array(parameters['supercell'])[2])
    input_file += '\n'

    return input_file
