from collections import namedtuple
import re
import numpy as np


def read_log_file(logdata_txt, compute_stress=False):
    """Read the log.lammps file."""
    data = logdata_txt.splitlines()

    if not data:
        raise IOError("The logfile is empty")

    data_dict = {}
    cell_params = None
    stress_params = None
    for i, line in enumerate(data):
        if "units" in line:
            data_dict["units_style"] = line.split()[1]
        if line.startswith("final_energy:"):
            data_dict["energy"] = float(line.split()[1])
        if line.startswith("final_variable:"):
            if "final_variables" not in data_dict:
                data_dict["final_variables"] = {}
            data_dict["final_variables"][line.split()[1]] = float(line.split()[3])

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
    super_cell = np.array([[xhi - xlo, xy, xz], [0, yhi - ylo, yz], [0, 0, zhi - zlo]])
    cell = super_cell.T
    if np.linalg.det(cell) < 0:
        cell = -1.0 * cell
    volume = np.linalg.det(cell)

    xx, yy, zz, xy, xz, yz = stress_params
    stress = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
    stress = -stress / volume  # to get stress in units of pressure

    return {"data": data_dict, "cell": cell, "stress": stress}


TRAJ_BLOCK = namedtuple(
    "TRAJ_BLOCK", ["timestep", "natoms", "cell", "field_names", "fields"]
)


def get_line(string, lines):
    for i, item in enumerate(lines):
        if string in item:
            return i


def iter_lammps_trajectories(file_obj):
    """Parse a LAMMPS Trajectory file, yielding data for each time step."""
    # TODO allow parsing of file chunks
    content = file_obj.read()

    # find all timestep blocks
    block_start = [m.start() for m in re.finditer("TIMESTEP", content)]
    blocks = [(block_start[i], block_start[i + 1]) for i in range(len(block_start) - 1)]
    if block_start:
        blocks.append((block_start[-1], len(content)))

    for row_start, row_end in blocks:
        block_lines = content[row_start:row_end].split("\n")

        time_step = int(block_lines[get_line("TIMESTEP", block_lines) + 1])
        number_of_atoms = int(block_lines[get_line("NUMBER OF ATOMS", block_lines) + 1])

        idx = get_line("ITEM: BOX", block_lines)
        # TODO check periodic dimensions
        bounds = [line.split() for line in block_lines[idx + 1 : idx + 4]]
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

        super_cell = np.array(
            [[xhi - xlo, xy, xz], [0, yhi - ylo, yz], [0, 0, zhi - zlo]]
        )
        cell = super_cell.T

        atom_line = get_line("ITEM: ATOMS", block_lines)
        field_names = block_lines[atom_line].split()[2:]

        atom_fields = []
        for i in range(number_of_atoms):
            atom_fields.append(block_lines[atom_line + i + 1].split())

        yield TRAJ_BLOCK(time_step, number_of_atoms, cell, field_names, atom_fields)


def get_units_dict(style, quantities, suffix="_units"):
    """Return a mapping of the unit name to the units, for a particular style.

    :param style: the unit style set in the lammps input
    :type style: str
    :param quantities: the quantities to get units for
    :type quantities: list of str
    :rtype: dict

    """
    units_dict = {
        "real": {
            "mass": "grams/mole",
            "distance": "Angstroms",
            "time": "femtoseconds",
            "energy": "Kcal/mole",
            "velocity": "Angstroms/femtosecond",
            "force": "Kcal/mole-Angstrom",
            "torque": "Kcal/mole",
            "temperature": "Kelvin",
            "pressure": "atmospheres",
            "dynamic_viscosity": "Poise",
            "charge": "e",  # multiple of electron charge (1.0 is a proton)
            "dipole": "charge*Angstroms",
            "electric field": "volts/Angstrom",
            "density": "gram/cm^dim",
        },
        "metal": {
            "mass": "grams/mole",
            "distance": "Angstroms",
            "time": "picoseconds",
            "energy": "eV",
            "velocity": "Angstroms/picosecond",
            "force": "eV/Angstrom",
            "torque": "eV",
            "temperature": "Kelvin",
            "pressure": "bars",
            "dynamic_viscosity": "Poise",
            "charge": "e",  # multiple of electron charge (1.0 is a proton)
            "dipole": "charge*Angstroms",
            "electric field": "volts/Angstrom",
            "density": "gram/cm^dim",
        },
        "si": {
            "mass": "kilograms",
            "distance": "meters",
            "time": "seconds",
            "energy": "Joules",
            "velocity": "meters/second",
            "force": "Newtons",
            "torque": "Newton-meters",
            "temperature": "Kelvin",
            "pressure": "Pascals",
            "dynamic_viscosity": "Pascal*second",
            "charge": "Coulombs",  # (1.6021765e-19 is a proton)
            "dipole": "Coulombs*meters",
            "electric field": "volts/meter",
            "density": "kilograms/meter^dim",
        },
        "cgs": {
            "mass": "grams",
            "distance": "centimeters",
            "time": "seconds",
            "energy": "ergs",
            "velocity": "centimeters/second",
            "force": "dynes",
            "torque": "dyne-centimeters",
            "temperature": "Kelvin",
            "pressure": "dyne/cm^2",  # or barye': '1.0e-6 bars
            "dynamic_viscosity": "Poise",
            "charge": "statcoulombs",  # or esu (4.8032044e-10 is a proton)
            "dipole": "statcoul-cm",  #: '10^18 debye
            "electric_field": "statvolt/cm",  # or dyne/esu
            "density": "grams/cm^dim",
        },
        "electron": {
            "mass": "amu",
            "distance": "Bohr",
            "time": "femtoseconds",
            "energy": "Hartrees",
            "velocity": "Bohr/atu",  # [1.03275e-15 seconds]
            "force": "Hartrees/Bohr",
            "temperature": "Kelvin",
            "pressure": "Pascals",
            "charge": "e",  # multiple of electron charge (1.0 is a proton)
            "dipole_moment": "Debye",
            "electric_field": "volts/cm",
        },
        "micro": {
            "mass": "picograms",
            "distance": "micrometers",
            "time": "microseconds",
            "energy": "picogram-micrometer^2/microsecond^2",
            "velocity": "micrometers/microsecond",
            "force": "picogram-micrometer/microsecond^2",
            "torque": "picogram-micrometer^2/microsecond^2",
            "temperature": "Kelvin",
            "pressure": "picogram/(micrometer-microsecond^2)",
            "dynamic_viscosity": "picogram/(micrometer-microsecond)",
            "charge": "picocoulombs",  # (1.6021765e-7 is a proton)
            "dipole": "picocoulomb-micrometer",
            "electric field": "volt/micrometer",
            "density": "picograms/micrometer^dim",
        },
        "nano": {
            "mass": "attograms",
            "distance": "nanometers",
            "time": "nanoseconds",
            "energy": "attogram-nanometer^2/nanosecond^2",
            "velocity": "nanometers/nanosecond",
            "force": "attogram-nanometer/nanosecond^2",
            "torque": "attogram-nanometer^2/nanosecond^2",
            "temperature": "Kelvin",
            "pressure": "attogram/(nanometer-nanosecond^2)",
            "dynamic_viscosity": "attogram/(nanometer-nanosecond)",
            "charge": "e",  # multiple of electron charge (1.0 is a proton)
            "dipole": "charge-nanometer",
            "electric_field": "volt/nanometer",
            "density": "attograms/nanometer^dim",
        },
    }
    out_dict = {}
    for quantity in quantities:
        out_dict[quantity + suffix] = units_dict[style][quantity]
    return out_dict


def convert_units(value, style, unit_type, out_units):
    conversion = {
        "seconds": 1,
        "milliseconds": 1e-3,
        "microseconds": 1e-6,
        "nanoseconds": 1e-9,
        "picoseconds": 1e-12,
        "femtoseconds": 1e-15,
    }
    if unit_type != "time" or out_units not in conversion:
        # TODO use https://pint.readthedocs.io
        raise NotImplementedError
    in_units = get_units_dict(style, [unit_type], "")[unit_type]
    return value * conversion[in_units] * (1.0 / conversion[out_units])


def parse_quasiparticle_data(qp_file):
    import yaml

    with open(qp_file, "r") as handle:
        quasiparticle_data = yaml.load(handle)

    data_dict = {}
    for i, data in enumerate(quasiparticle_data):
        data_dict["q_point_{}".format(i)] = data

    return data_dict


def parse_dynaphopy_output(file):

    thermal_properties = None

    with open(file, "r") as handle:
        data_lines = handle.readlines()

    indices = []
    q_points = []
    for i, line in enumerate(data_lines):
        if "Q-point" in line:
            #        print i, np.array(line.replace(']', '').replace('[', '').split()[4:8], dtype=float)
            indices.append(i)
            q_points.append(
                np.array(
                    line.replace("]", "").replace("[", "").split()[4:8], dtype=float
                )
            )

    indices.append(len(data_lines))

    phonons = {}
    for i, index in enumerate(indices[:-1]):

        fragment = data_lines[indices[i] : indices[i + 1]]
        if "kipped" in fragment:
            continue
        phonon_modes = {}
        for j, line in enumerate(fragment):
            if "Peak" in line:
                number = line.split()[2]
                phonon_mode = {
                    "width": float(fragment[j + 2].split()[1]),
                    "positions": float(fragment[j + 3].split()[1]),
                    "shift": float(fragment[j + 12].split()[2]),
                }
                phonon_modes.update({number: phonon_mode})

            if "Thermal" in line:
                free_energy = float(fragment[j + 4].split()[4])
                entropy = float(fragment[j + 5].split()[3])
                cv = float(fragment[j + 6].split()[3])
                total_energy = float(fragment[j + 7].split()[4])

                temperature = float(fragment[j].split()[5].replace("(", ""))

                thermal_properties = {
                    "temperature": temperature,
                    "free_energy": free_energy,
                    "entropy": entropy,
                    "cv": cv,
                    "total_energy": total_energy,
                }

        phonon_modes.update({"q_point": q_points[i].tolist()})

        phonons.update({"wave_vector_" + str(i): phonon_modes})

    return thermal_properties
