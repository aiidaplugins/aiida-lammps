import re

import numpy as np


def read_log_file(logdata_txt, compute_stress=False):
    """Read the log.lammps file."""
    data = logdata_txt.splitlines()

    if not data:
        raise IOError("The logfile is empty")

    perf_regex = re.compile(
        r"Performance\:\s(.+)\sns\/day,\s(.+)\shours\/ns\,\s(.+)\stimesteps\/s\s*"
    )

    data_dict = {}
    cell_params = None
    stress_params = None
    found_end = False
    for i, line in enumerate(data):
        line = line.strip()
        if "END_OF_COMP" in line:
            found_end = True
        elif "Total wall time:" in line:
            data_dict["total_wall_time"] = line.split()[-1]
        # These are handled in LAMMPSBaseParser.add_warnings_and_errors
        # if line.strip().startswith("WARNING"):
        #     data_dict.setdefault("warnings", []).append(line.strip())
        # if line.strip().startswith("ERROR"):
        #     data_dict.setdefault("errors", []).append(line.strip())
        elif perf_regex.match(line):
            ns_day, hr_ns, step_sec = perf_regex.match(line).groups()
            data_dict.setdefault("steps_per_second", []).append(float(step_sec))
        elif "units" in line:
            data_dict["units_style"] = line.split()[1]
        elif line.startswith("final_energy:"):
            data_dict["energy"] = float(line.split()[1])
        elif line.startswith("final_variable:"):
            if "final_variables" not in data_dict:
                data_dict["final_variables"] = {}
            data_dict["final_variables"][line.split()[1]] = float(line.split()[3])

        elif line.startswith("final_cell:"):
            cell_params = [float(v) for v in line.split()[1:10]]
        elif line.startswith("final_stress:"):
            stress_params = [float(v) for v in line.split()[1:7]]

    if not compute_stress:
        return {"data": data_dict, "found_end": found_end}

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

    return {"data": data_dict, "cell": cell, "stress": stress, "found_end": found_end}


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
