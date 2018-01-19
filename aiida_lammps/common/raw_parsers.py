import numpy as np


def parse_quasiparticle_data(qp_file):
    import yaml

    f = open(qp_file, "r")
    quasiparticle_data = yaml.load(f)
    f.close()
    data_dict = {}
    for i, data in enumerate(quasiparticle_data):
        data_dict['q_point_{}'.format(i)] = data

    return data_dict


def parse_dynaphopy_output(file):

    thermal_properties = None
    f = open(file, 'r')
    data_lines = f.readlines()

    indices = []
    q_points = []
    for i, line in enumerate(data_lines):
        if 'Q-point' in line:
    #        print i, np.array(line.replace(']', '').replace('[', '').split()[4:8], dtype=float)
            indices.append(i)
            q_points.append(np.array(line.replace(']', '').replace('[', '').split()[4:8],dtype=float))

    indices.append(len(data_lines))

    phonons = {}
    for i, index in enumerate(indices[:-1]):

        fragment = data_lines[indices[i]: indices[i+1]]
        if 'kipped' in fragment:
            continue
        phonon_modes = {}
        for j, line in enumerate(fragment):
            if 'Peak' in line:
                number = line.split()[2]
                phonon_mode = {'width':     float(fragment[j+2].split()[1]),
                               'positions': float(fragment[j+3].split()[1]),
                               'shift':     float(fragment[j+12].split()[2])}
                phonon_modes.update({number: phonon_mode})

            if 'Thermal' in line:
                free_energy = float(fragment[j+4].split()[4])
                entropy = float(fragment[j+5].split()[3])
                cv = float(fragment[j+6].split()[3])
                total_energy = float(fragment[j+7].split()[4])

                temperature = float(fragment[j].split()[5].replace('(',''))

                thermal_properties = {'temperature': temperature,
                                      'free_energy': free_energy,
                                      'entropy': entropy,
                                      'cv': cv,
                                      'total_energy': total_energy}


        phonon_modes.update({'q_point': q_points[i].tolist()})

        phonons.update({'wave_vector_'+str(i): phonon_modes})

        f.close()

    return thermal_properties


def read_lammps_forces(file_name):

    import mmap
    # Time in picoseconds
    # Coordinates in Angstroms

    # Starting reading

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    cells = []

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        # Read time steps
        position_number=file_map.find('TIMESTEP')

        file_map.seek(position_number)
        file_map.readline()


        #Read number of atoms
        position_number=file_map.find('NUMBER OF ATOMS')
        file_map.seek(position_number)
        file_map.readline()
        number_of_atoms = int(file_map.readline())


        #Read cell
        position_number=file_map.find('ITEM: BOX')
        file_map.seek(position_number)
        file_map.readline()


        bounds = []
        for i in range(3):
            bounds.append(file_map.readline().split())

        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T ,axis=1)


        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy+xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy+xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi-xlo, xy,  xz],
                               [0,  yhi-ylo,  yz],
                               [0,   0,  zhi-zlo]])

        cells.append(super_cell.T)

        position_number = file_map.find('ITEM: ATOMS')
        file_map.seek(position_number)
        file_map.readline()

        #Reading forces
        forces = []
        read_elements = []
        for i in range (number_of_atoms):
            line = file_map.readline().split()[0:number_of_dimensions+1]
            forces.append(line[1:number_of_dimensions+1])
            read_elements.append(line[0])

    file_map.close()

    forces = np.array([forces], dtype=float)

    return forces

import numpy as np

def read_log_file(logfile):

    f = open(logfile, 'r')
    data = f.readlines()

    data_dict = {}
    for i, line in enumerate(data):
        if 'Loop time' in line:
            energy = float(data[i-1].split()[4])
            data_dict['energy'] = energy

    return data_dict


def read_lammps_trajectory(file_name,
                           limit_number_steps=100000000,
                           initial_cut=1,
                           end_cut=None,
                           timestep=1):

    import mmap
    # Time in picoseconds
    # Coordinates in Angstroms

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    step_ids = []
    data = []
    cells = []
    counter = 0
    bounds = None
    number_of_atoms = None

    lammps_labels = False

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        while True:

            counter += 1

            # Read time steps
            position_number=file_map.find('TIMESTEP')
            if position_number < 0: break

            file_map.seek(position_number)
            file_map.readline()
            step_ids.append(float(file_map.readline()))

            if number_of_atoms is None:
                # Read number of atoms
                position_number=file_map.find('NUMBER OF ATOMS')
                file_map.seek(position_number)
                file_map.readline()
                number_of_atoms = int(file_map.readline())

            if True:
                # Read cell
                position_number=file_map.find('ITEM: BOX')
                file_map.seek(position_number)
                file_map.readline()


                bounds = []
                for i in range(3):
                    bounds.append(file_map.readline().split())

                bounds = np.array(bounds, dtype=float)
                if bounds.shape[1] == 2:
                    bounds = np.append(bounds, np.array([0, 0, 0])[None].T ,axis=1)

                xy = bounds[0, 2]
                xz = bounds[1, 2]
                yz = bounds[2, 2]

                xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy+xz])
                xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy+xz])
                ylo = bounds[1, 0] - np.min([0.0, yz])
                yhi = bounds[1, 1] - np.max([0.0, yz])
                zlo = bounds[2, 0]
                zhi = bounds[2, 1]

                super_cell = np.array([[xhi-xlo, xy,  xz],
                                       [0,  yhi-ylo,  yz],
                                       [0,   0,  zhi-zlo]])
                cells.append(super_cell.T)

            position_number = file_map.find('ITEM: ATOMS')
            file_map.seek(position_number)
            lammps_labels=file_map.readline() # lammps_labels not used now but call is necessary!

            # Initial cut control
            if initial_cut > counter:
                continue

            # Reading coordinates
            read_coordinates = []
            read_elements = []
            for i in range (number_of_atoms):
                line = file_map.readline().split()[0:number_of_dimensions+1]
                read_coordinates.append(line[1:number_of_dimensions+1])
                read_elements.append(line[0])
            try:
                data.append(np.array(read_coordinates, dtype=float)) #in angstroms
            except ValueError:
                print("Error reading step {0}".format(counter))
                break
            # security routine to limit maximum of steps to read and put in memory
            if limit_number_steps+initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break

    file_map.close()

    data = np.array(data)
    step_ids = np.array(step_ids, dtype=int)
    cells = np.array(cells)
    elements = np.array(read_elements)
    time = np.array(step_ids)*timestep
    return data, step_ids, cells, elements, time


def read_log_file2(logfile):

    f = open(logfile, 'r')
    data = f.readlines()

    data_dict = {}
    for i, line in enumerate(data):
        if 'Loop time' in line:
            energy = float(data[i-1].split()[4])
            data_dict['energy'] = energy
            xx, yy, zz, xy, xz, yz = data[i-1].split()[5:11]
            stress = np.array([[xx, xy, xz],
                               [xy, yy, yz],
                               [xz, yz, zz]], dtype=float)


        if '$(xlo)' in line:
            a = data[i+1].split()
        if '$(ylo)' in line:
            b = data[i+1].split()
        if '$(zlo)' in line:
            c = data[i+1].split()

    bounds = np.array([a, b, c], dtype=float)

 #   lammps_input_file += 'print           "$(xlo) $(xhi) $(xy)"\n'
 #   lammps_input_file += 'print           "$(ylo) $(yhi) $(xz)"\n'
 #   lammps_input_file += 'print           "$(zlo) $(zhi) $(yz)"\n'


    xy = bounds[0, 2]
    xz = bounds[1, 2]
    yz = bounds[2, 2]

    xlo = bounds[0, 0]
    xhi = bounds[0, 1]
    ylo = bounds[1, 0]
    yhi = bounds[1, 1]
    zlo = bounds[2, 0]
    zhi = bounds[2, 1]

    super_cell = np.array([[xhi-xlo, xy,  xz],
                           [0,  yhi-ylo,  yz],
                           [0,   0,  zhi-zlo]])

    cell=super_cell.T

    if np.linalg.det(cell) < 0:
        cell = -1.0*cell

    volume = np.linalg.det(cell)
    stress = -stress/volume * 1.e-3  # bar*A^3 -> kbar

    return data_dict, cell, stress

def read_lammps_positions_and_forces(file_name):

    import mmap
    # Time in picoseconds
    # Coordinates in Angstroms

    # Starting reading

    # Dimensionality of LAMMP calculation
    number_of_dimensions = 3


    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        # Read time steps
        while True:
            position_number=file_map.find('TIMESTEP')
            try:
                file_map.seek(position_number)
                file_map.readline()
            except ValueError:
                break

        #Read number of atoms
        position_number=file_map.find('NUMBER OF ATOMS')
        file_map.seek(position_number)
        file_map.readline()
        number_of_atoms = int(file_map.readline())


       #Read cell
        position_number=file_map.find('ITEM: BOX')
        file_map.seek(position_number)
        file_map.readline()

        bounds = []
        for i in range(3):
            bounds.append(file_map.readline().split())

        bounds = np.array(bounds, dtype=float)
        if bounds.shape[1] == 2:
            bounds = np.append(bounds, np.array([0, 0, 0])[None].T ,axis=1)

        xy = bounds[0, 2]
        xz = bounds[1, 2]
        yz = bounds[2, 2]

        xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy+xz])
        xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy+xz])
        ylo = bounds[1, 0] - np.min([0.0, yz])
        yhi = bounds[1, 1] - np.max([0.0, yz])
        zlo = bounds[2, 0]
        zhi = bounds[2, 1]

        super_cell = np.array([[xhi-xlo, xy,  xz],
                               [0,  yhi-ylo,  yz],
                               [0,   0,  zhi-zlo]])

        cell=super_cell.T


        position_number = file_map.find('ITEM: ATOMS')
        file_map.seek(position_number)
        file_map.readline()

        #Reading positions
        positions = []
        forces = []
        read_elements = []
        for i in range (number_of_atoms):
            line = file_map.readline().split()[0:number_of_dimensions*2+1]
            positions.append(line[1:number_of_dimensions+1])
            forces.append(line[1+number_of_dimensions:number_of_dimensions*2+1])
            read_elements.append(line[0])

    file_map.close()

    positions = np.array([positions])
    forces = np.array([forces], dtype=float)

    return positions, forces, read_elements, cell
