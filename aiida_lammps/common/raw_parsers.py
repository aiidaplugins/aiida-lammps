import mmap
import re
import numpy as np
import six


def parse_quasiparticle_data(qp_file):
    import yaml

    with open(qp_file, "r") as handle:
        quasiparticle_data = yaml.load(handle)

    data_dict = {}
    for i, data in enumerate(quasiparticle_data):
        data_dict['q_point_{}'.format(i)] = data

    return data_dict


def parse_dynaphopy_output(file):

    thermal_properties = None

    with open(file, 'r') as handle:
        data_lines = handle.readlines()

    indices = []
    q_points = []
    for i, line in enumerate(data_lines):
        if 'Q-point' in line:
            #        print i, np.array(line.replace(']', '').replace('[', '').split()[4:8], dtype=float)
            indices.append(i)
            q_points.append(np.array(line.replace(
                ']', '').replace('[', '').split()[4:8], dtype=float))

    indices.append(len(data_lines))

    phonons = {}
    for i, index in enumerate(indices[:-1]):

        fragment = data_lines[indices[i]: indices[i + 1]]
        if 'kipped' in fragment:
            continue
        phonon_modes = {}
        for j, line in enumerate(fragment):
            if 'Peak' in line:
                number = line.split()[2]
                phonon_mode = {'width': float(fragment[j + 2].split()[1]),
                               'positions': float(fragment[j + 3].split()[1]),
                               'shift': float(fragment[j + 12].split()[2])}
                phonon_modes.update({number: phonon_mode})

            if 'Thermal' in line:
                free_energy = float(fragment[j + 4].split()[4])
                entropy = float(fragment[j + 5].split()[3])
                cv = float(fragment[j + 6].split()[3])
                total_energy = float(fragment[j + 7].split()[4])

                temperature = float(fragment[j].split()[5].replace('(', ''))

                thermal_properties = {'temperature': temperature,
                                      'free_energy': free_energy,
                                      'entropy': entropy,
                                      'cv': cv,
                                      'total_energy': total_energy}

        phonon_modes.update({'q_point': q_points[i].tolist()})

        phonons.update({'wave_vector_' + str(i): phonon_modes})

    return thermal_properties


def read_lammps_forces(file_name):

    # Time in picoseconds
    # Coordinates in Angstroms

    # Starting reading

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    cells = []

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        # Read time steps
        position_number = file_map.find('TIMESTEP')

        file_map.seek(position_number)
        file_map.readline()

        # Read number of atoms
        position_number = file_map.find('NUMBER OF ATOMS')
        file_map.seek(position_number)
        file_map.readline()
        number_of_atoms = int(file_map.readline())

        # Read cell
        position_number = file_map.find('ITEM: BOX')
        file_map.seek(position_number)
        file_map.readline()

        bounds = []
        for i in range(3):
            bounds.append(file_map.readline().split())

        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T, axis=1)

        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi - xlo, xy, xz],
                               [0, yhi - ylo, yz],
                               [0, 0, zhi - zlo]])

        cells.append(super_cell.T)

        position_number = file_map.find('ITEM: ATOMS')
        file_map.seek(position_number)
        file_map.readline()

        # Reading forces
        forces = []
        read_elements = []
        for i in range(number_of_atoms):
            line = file_map.readline().split()[0:number_of_dimensions + 1]
            forces.append(line[1:number_of_dimensions + 1])
            read_elements.append(line[0])

    file_map.close()

    forces = np.array([forces], dtype=float)

    return forces


def read_log_file(logdata_txt, compute_stress=False):
    """ read the log.lammps file """
    # Dimensionality of LAMMP calculation
    # number_of_dimensions = 3

    data = logdata_txt.splitlines()

    if not data:
        raise IOError('The logfile is empty')

    data_dict = {}
    cell_params = None
    stress_params = None
    for i, line in enumerate(data):
        if 'units' in line:
            data_dict['units_style'] = line.split()[1]
        if line.startswith("final_energy:"):
            data_dict['energy'] = float(line.split()[1])
        if line.startswith("final_variable:"):
            if 'final_variables' not in data_dict:
                data_dict['final_variables'] = {}
            data_dict['final_variables'][line.split()[1]] = float(
                line.split()[3])

        if line.startswith("final_cell:"):
            cell_params = [float(v) for v in line.split()[1:10]]
        if line.startswith("final_stress:"):
            stress_params = [float(v) for v in line.split()[1:7]]

    if not compute_stress:
        return {"data": data_dict}

    if cell_params is None:
        raise IOError("'final_cell' could not be found")
    if stress_params is None:
        raise IOError("'final_stress' could not be found")

    xlo, xhi, xy, ylo, yhi, xz, zlo, zhi, yz = cell_params
    super_cell = np.array([[xhi - xlo, xy, xz],
                           [0, yhi - ylo, yz],
                           [0, 0, zhi - zlo]])
    cell = super_cell.T
    if np.linalg.det(cell) < 0:
        cell = -1.0 * cell
    volume = np.linalg.det(cell)

    xx, yy, zz, xy, xz, yz = stress_params
    stress = np.array([[xx, xy, xz],
                       [xy, yy, yz],
                       [xz, yz, zz]], dtype=float)
    stress = -stress / volume  # to get stress in units of pressure

    return {"data": data_dict, "cell": cell, "stress": stress}


def read_lammps_trajectory_txt(data_txt,
                               limit_number_steps=100000000,
                               initial_cut=1,
                               timestep=1):

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    blocks = [m.start() for m in re.finditer('TIMESTEP', data_txt)]
    blocks = [(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)]

    blocks = blocks[initial_cut:initial_cut + limit_number_steps]

    step_ids = []
    position_list = []

    read_elements = None
    cells = []

    time_steps = []
    for ini, end in blocks:
        # Read number of atoms
        block_lines = data_txt[ini:end].split('\n')
        id = block_lines.index('TIMESTEP')
        time_steps.append(block_lines[id + 1])

        id = get_index('NUMBER OF ATOMS', block_lines)
        number_of_atoms = int(block_lines[id + 1])

        id = get_index('ITEM: BOX', block_lines)
        bounds = [line.split() for line in block_lines[id + 1:id + 4]]
        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T, axis=1)

        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi - xlo, xy, xz],
                               [0, yhi - ylo, yz],
                               [0, 0, zhi - zlo]])
        cell = super_cell.T

        # id = [i for i, s in enumerate(block_lines) if 'ITEM: BOX BOUNDS' in s][0]

        # Reading positions
        id = get_index('ITEM: ATOMS', block_lines)

        positions = []
        read_elements = []
        for i in range(number_of_atoms):
            line = block_lines[id + i + 1].split()
            positions.append(line[1:number_of_dimensions + 1])
            read_elements.append(line[0])

        position_list.append(positions)
        cells.append(cell)

    positions = np.array(position_list, dtype=float)
    step_ids = np.array(time_steps, dtype=int)
    cells = np.array(cells)
    elements = np.array(read_elements)
    time = np.array(step_ids) * float(timestep)

    return positions, step_ids, cells, elements, time


def read_lammps_trajectory(file_name,
                           limit_number_steps=100000000,
                           initial_cut=1, end_cut=None,
                           timestep=1, log_warning_func=None):
    """ should be used with:
    `dump name all custom n element x y z q`, where q is optional
    """
    if log_warning_func is None:
        log_warning_func = six.print_
    # Time in picoseconds
    # Coordinates in Angstroms

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    step_ids = []
    positions = []
    charges = []
    cells = []
    read_elements = []
    counter = 0
    bounds = None
    number_of_atoms = None

    field_names = False

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        while True:

            counter += 1

            # Read time steps
            position_number = file_map.find(b'TIMESTEP')
            if position_number < 0:
                break

            file_map.seek(position_number)
            file_map.readline()
            step_ids.append(float(file_map.readline()))

            if number_of_atoms is None:
                # Read number of atoms
                position_number = file_map.find(b'NUMBER OF ATOMS')
                file_map.seek(position_number)
                file_map.readline()
                number_of_atoms = int(file_map.readline())

            if True:
                # Read cell
                position_number = file_map.find(b'ITEM: BOX')
                file_map.seek(position_number)
                file_map.readline()

                bounds = []
                for i in range(3):
                    bounds.append(file_map.readline().split())

                bounds = np.array(bounds, dtype=float)
                if bounds.shape[1] == 2:
                    bounds = np.append(bounds, np.array(
                        [0, 0, 0])[None].T, axis=1)

                xy = bounds[0, 2]
                xz = bounds[1, 2]
                yz = bounds[2, 2]

                xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
                xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
                ylo = bounds[1, 0] - np.min([0.0, yz])
                yhi = bounds[1, 1] - np.max([0.0, yz])
                zlo = bounds[2, 0]
                zhi = bounds[2, 1]

                super_cell = np.array([[xhi - xlo, xy, xz],
                                       [0, yhi - ylo, yz],
                                       [0, 0, zhi - zlo]])
                cells.append(super_cell.T)

            position_number = file_map.find(b'ITEM: ATOMS')
            file_map.seek(position_number)
            field_names = six.ensure_str(file_map.readline()).split()[2:]
            has_charge = field_names[-1] == 'q'

            # Initial cut control
            if initial_cut > counter:
                continue

            # Reading coordinates
            read_coordinates = []
            read_elements = []
            read_charges = []
            for i in range(number_of_atoms):
                fields = file_map.readline().split()
                read_coordinates.append(fields[1:number_of_dimensions + 1])
                read_elements.append(fields[0])
                read_charges.append(fields[-1])
            try:
                positions.append(np.array(read_coordinates, dtype=float))
            except ValueError:
                log_warning_func("Error reading step {0}".format(counter))
                break
            if has_charge:
                charges.append(read_charges)
            # security routine to limit maximum of steps to read and put in memory
            if limit_number_steps + initial_cut < counter:
                log_warning_func(
                    "Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break

    file_map.close()

    positions = np.array(positions)
    if charges:
        charges = np.array(charges, dtype=float)
    else:
        charges = None
    step_ids = np.array(step_ids, dtype=int)
    cells = np.array(cells)
    elements = np.array(read_elements, dtype='str')

    time = np.array(step_ids) * timestep
    return positions, charges, step_ids, cells, elements, time


def read_lammps_positions_and_forces(file_name):

    # Time in picoseconds
    # Coordinates in Angstroms

    # Starting reading

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        # Read time steps
        while True:
            position_number = file_map.find('TIMESTEP')
            try:
                file_map.seek(position_number)
                file_map.readline()
            except ValueError:
                break

        # Read number of atoms
        position_number = file_map.find('NUMBER OF ATOMS')
        file_map.seek(position_number)
        file_map.readline()
        number_of_atoms = int(file_map.readline())

        # Read cell
        position_number = file_map.find('ITEM: BOX')
        file_map.seek(position_number)
        file_map.readline()

        bounds = []
        for i in range(3):
            bounds.append(file_map.readline().split())

        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T, axis=1)

        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi - xlo, xy, xz],
                               [0, yhi - ylo, yz],
                               [0, 0, zhi - zlo]])

        cell = super_cell.T

        position_number = file_map.find('ITEM: ATOMS')
        file_map.seek(position_number)
        file_map.readline()

        # Reading positions
        positions = []
        forces = []
        read_elements = []
        for i in range(number_of_atoms):
            line = file_map.readline().split()[0:number_of_dimensions * 2 + 1]
            positions.append(line[1:number_of_dimensions + 1])
            forces.append(
                line[1 + number_of_dimensions:number_of_dimensions * 2 + 1])
            read_elements.append(line[0])

    file_map.close()

    positions = np.array([positions])
    forces = np.array([forces], dtype=float)

    return positions, forces, read_elements, cell


def get_index(string, lines):
    for i, item in enumerate(lines):
        if string in item:
            return i


def read_lammps_positions_and_forces_txt(data_txt):
    """ should be used with:
    `dump name all custom n element x y z fx fy fz q`,
    where q is optional
    """

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    block_start = [m.start() for m in re.finditer('TIMESTEP', data_txt)]
    blocks = [(block_start[i], block_start[i + 1]) for i in range(len(block_start) - 1)]
    # add last block
    blocks.append((block_start[-1], len(data_txt)))

    position_list = []
    forces_list = []
    charge_list = []

    read_elements = None
    cell = None

    time_steps = []
    for ini, end in blocks:

        # Read number of atoms
        block_lines = data_txt[ini:end].split('\n')

        id = block_lines.index('TIMESTEP')
        time_steps.append(block_lines[id + 1])

        id = get_index('NUMBER OF ATOMS', block_lines)
        number_of_atoms = int(block_lines[id + 1])

        id = get_index('ITEM: BOX', block_lines)
        bounds = [line.split() for line in block_lines[id + 1:id + 4]]
        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T, axis=1)

        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi - xlo, xy, xz],
                               [0, yhi - ylo, yz],
                               [0, 0, zhi - zlo]])
        cell = super_cell.T

        # id = [i for i, s in enumerate(block_lines) if 'ITEM: BOX BOUNDS' in s][0]

        # Reading positions
        id = get_index('ITEM: ATOMS', block_lines)
        field_names = block_lines[id].split()[2:]  # noqa: F841

        positions = []
        forces = []
        charges = []
        read_elements = []
        for i in range(number_of_atoms):
            fields = block_lines[id + i + 1].split()
            positions.append(fields[1:number_of_dimensions + 1])
            forces.append(
                fields[1 + number_of_dimensions:number_of_dimensions * 2 + 1])
            if field_names[-1] == "q":
                charges.append(fields[-1])
            read_elements.append(fields[0])

        position_list.append(positions)
        forces_list.append(forces)
        if field_names[-1] == "q":
            charge_list.append(charges)

    positions = np.array(position_list, dtype=float)
    forces = np.array(forces_list, dtype=float)
    if charge_list:
        charges = np.array(charge_list, dtype=float)
    else:
        charges = None

    return positions, forces, charges, read_elements, cell


def get_units_dict(style, quantities):
    """

    :param style: the unit style set in the lammps input
    :type style: str
    :param quantities: the quantities to get units for
    :type quantities: list of str
    :return:
    """

    units_dict = {
        'real':
        {
            'mass': 'grams/mole',
            'distance': 'Angstroms',
            'time': 'femtoseconds',
            'energy': 'Kcal/mole',
            'velocity': 'Angstroms/femtosecond',
            'force': 'Kcal/mole-Angstrom',
            'torque': 'Kcal/mole',
            'temperature': 'Kelvin',
            'pressure': 'atmospheres',
            'dynamic_viscosity': 'Poise',
            'charge': 'e',  # multiple of electron charge (1.0 is a proton)
            'dipole': 'charge*Angstroms',
            'electric field': 'volts/Angstrom',
            'density': 'gram/cm^dim',
        },
        'metal': {

            'mass': 'grams/mole',
            'distance': 'Angstroms',
            'time': 'picoseconds',
            'energy': 'eV',
            'velocity': 'Angstroms/picosecond',
            'force': 'eV/Angstrom',
            'torque': 'eV',
            'temperature': 'Kelvin',
            'pressure': 'bars',
            'dynamic_viscosity': 'Poise',
            'charge': 'e',  # multiple of electron charge (1.0 is a proton)
            'dipole': 'charge*Angstroms',
            'electric field': 'volts/Angstrom',
            'density': 'gram/cm^dim',
        },
        'si': {
            'mass': 'kilograms',
            'distance': 'meters',
            'time': 'seconds',
            'energy': 'Joules',
            'velocity': 'meters/second',
            'force': 'Newtons',
            'torque': 'Newton-meters',
            'temperature': 'Kelvin',
            'pressure': 'Pascals',
            'dynamic_viscosity': 'Pascal*second',
            'charge': 'Coulombs',  # (1.6021765e-19 is a proton)
            'dipole': 'Coulombs*meters',
            'electric field': 'volts/meter',
            'density': 'kilograms/meter^dim',
        },
        'cgs': {

            'mass': 'grams',
            'distance': 'centimeters',
            'time': 'seconds',
            'energy': 'ergs',
            'velocity': 'centimeters/second',
            'force': 'dynes',
            'torque': 'dyne-centimeters',
            'temperature': 'Kelvin',
            'pressure': 'dyne/cm^2',  # or barye': '1.0e-6 bars
            'dynamic_viscosity': 'Poise',
            'charge': 'statcoulombs',  # or esu (4.8032044e-10 is a proton)
            'dipole': 'statcoul-cm',  #: '10^18 debye
            'electric_field': 'statvolt/cm',  # or dyne/esu
            'density': 'grams/cm^dim',
        },
        'electron': {

            'mass': 'amu',
            'distance': 'Bohr',
            'time': 'femtoseconds',
            'energy': 'Hartrees',
            'velocity': 'Bohr/atu',  # [1.03275e-15 seconds]
            'force': 'Hartrees/Bohr',
            'temperature': 'Kelvin',
            'pressure': 'Pascals',
            'charge': 'e',  # multiple of electron charge (1.0 is a proton)
            'dipole_moment': 'Debye',
            'electric_field': 'volts/cm',
        },
        'micro': {

            'mass': 'picograms',
            'distance': 'micrometers',
            'time': 'microseconds',
            'energy': 'picogram-micrometer^2/microsecond^2',
            'velocity': 'micrometers/microsecond',
            'force': 'picogram-micrometer/microsecond^2',
            'torque': 'picogram-micrometer^2/microsecond^2',
            'temperature': 'Kelvin',
            'pressure': 'picogram/(micrometer-microsecond^2)',
            'dynamic_viscosity': 'picogram/(micrometer-microsecond)',
            'charge': 'picocoulombs',  # (1.6021765e-7 is a proton)
            'dipole': 'picocoulomb-micrometer',
            'electric field': 'volt/micrometer',
            'density': 'picograms/micrometer^dim',
        },

        'nano': {

            'mass': 'attograms',
            'distance': 'nanometers',
            'time': 'nanoseconds',
            'energy': 'attogram-nanometer^2/nanosecond^2',
            'velocity': 'nanometers/nanosecond',
            'force': 'attogram-nanometer/nanosecond^2',
            'torque': 'attogram-nanometer^2/nanosecond^2',
            'temperature': 'Kelvin',
            'pressure': 'attogram/(nanometer-nanosecond^2)',
            'dynamic_viscosity': 'attogram/(nanometer-nanosecond)',
            'charge': 'e',  # multiple of electron charge (1.0 is a proton)
            'dipole': 'charge-nanometer',
            'electric_field': 'volt/nanometer',
            'density': 'attograms/nanometer^dim'
        }
    }
    out_dict = {}
    for quantity in quantities:
        out_dict[quantity + "_units"] = units_dict[style][quantity]
    return out_dict
