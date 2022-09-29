"""Fixtures for the calculations tests."""
from aiida.common import AttributeDict
import numpy as np


def minimize_parameters() -> AttributeDict:
    """
    Set of parameters for a minimization calculation

    :return: parameters for a minimization calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
    }

    parameters.minimize = {
        "style": "cg",
        "energy_tolerance": 1e-5,
        "force_tolerance": 1e-5,
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 100,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


def minimize_reference_data() -> AttributeDict:
    """
    Set of reference values for a minimization calculation

    :return: reference values for a minimization calculation
    :rtype: AttributeDict
    """
    reference_data = AttributeDict()
    _results = {
        "final_ke": 0,
        "final_pe": -8.2441284132802,
        "final_pxx": 13557.642534563,
        "final_pyy": 13557.642534563,
        "final_pzz": 13557.642534563,
        "final_step": 1,
        "final_press": 13557.642534563,
        "final_etotal": -8.2441284132802,
        "compute_variables": {
            "bin": "standard",
            "bins": [1, 1, 1],
            "binsize": 3.65,
            "units_style": "metal",
            "total_wall_time": "0:00:00",
            "ghost_atom_cutoff": 7.3,
            "max_neighbors_atom": 2000,
            "master_list_distance_cutoff": 7.3,
        },
        "final_c_pressure_all_aiida": 13557.642534563,
        "final_c_pressure_all_aiida__1__": 13557.642534563,
        "final_c_pressure_all_aiida__2__": 13557.642534563,
        "final_c_pressure_all_aiida__3__": 13557.642534563,
        "final_c_pressure_all_aiida__4__": 7.7954752747919e-11,
        "final_c_pressure_all_aiida__5__": -7.1699124440987e-11,
        "final_c_pressure_all_aiida__6__": -8.2285572345026e-11,
    }

    _time_dependent_computes = {
        "Pxx": np.asarray([13557.643, 13557.643]),
        "Pyy": np.asarray([13557.643, 13557.643]),
        "Pzz": np.asarray([13557.643, 13557.643]),
        "Step": np.asarray([0.0, 1.0]),
        "Press": np.asarray([13557.643, 13557.643]),
        "KinEng": np.asarray([0.0, 0.0]),
        "PotEng": np.asarray([-8.2441284, -8.2441284]),
        "TotEng": np.asarray([-8.2441284, -8.2441284]),
        "c_pressure_all_aiida": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__1__": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__2__": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__3__": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__4__": np.asarray([-1.0821034e-10, 7.7954753e-11]),
        "c_pressure_all_aiida__5__": np.asarray([-6.8330709e-11, -7.1699124e-11]),
        "c_pressure_all_aiida__6__": np.asarray([-8.2766775e-11, -8.2285572e-11]),
    }

    _trajectories = {
        "attributes": {
            "aliases": None,
            "elements": ["Fe"],
            "zip_prefix": "step-",
            "field_names": [
                "c_ke_atom_all_aiida",
                "c_pe_atom_all_aiida",
                "c_stress_atom_all_aiida__1__",
                "c_stress_atom_all_aiida__2__",
                "c_stress_atom_all_aiida__3__",
                "c_stress_atom_all_aiida__4__",
                "c_stress_atom_all_aiida__5__",
                "c_stress_atom_all_aiida__6__",
                "element",
                "id",
                "type",
                "x",
                "y",
                "z",
            ],
            "number_atoms": 2,
            "number_steps": 2,
            "timestep_filename": "timesteps.txt",
            "compression_method": 8,
            "trajectory_filename": "trajectory.zip",
        },
        "step_data": {
            0: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["0.0000000000", "1.4240580000"],
                "y": ["0.0000000000", "1.4240580000"],
                "z": ["0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113650",
                    "-156612.7819113652",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000009"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            1: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113636",
                    "-156612.7819113671",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113637",
                    "-156612.7819113671",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113644",
                    "-156612.7819113675",
                ],
                "c_stress_atom_all_aiida__4__": ["-0.0000000012", "-0.0000000011"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__6__": ["0.0000000009", "0.0000000009"],
            },
        },
    }

    reference_data.results = _results
    reference_data.time_dependent_computes = _time_dependent_computes
    reference_data.trajectories = _trajectories

    return reference_data


def minimize_parameters_groups() -> AttributeDict:
    """
    Set of parameters for a minimization calculation using groups

    :return: parameters for a minimization calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        "ke": [{"type": [{"keyword": " ", "value": " "}], "group": "test"}],
    }

    parameters.minimize = {
        "style": "cg",
        "energy_tolerance": 1e-5,
        "force_tolerance": 1e-5,
    }

    parameters.structure = {
        "atom_style": "atomic",
        "groups": [{"name": "test", "args": ["type", 1]}],
    }
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 100,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


def minimize_groups_reference_data() -> AttributeDict:
    reference_data = AttributeDict()

    _results = {
        "final_ke": 0,
        "final_pe": -8.2441284132802,
        "final_pxx": 13557.642534563,
        "final_pyy": 13557.642534563,
        "final_pzz": 13557.642534563,
        "final_step": 1,
        "final_press": 13557.642534563,
        "final_etotal": -8.2441284132802,
        "compute_variables": {
            "bin": "standard",
            "bins": [1, 1, 1],
            "binsize": 3.65,
            "units_style": "metal",
            "total_wall_time": "0:00:00",
            "ghost_atom_cutoff": 7.3,
            "max_neighbors_atom": 2000,
            "master_list_distance_cutoff": 7.3,
        },
        "final_c_ke_test_aiida": 0,
        "final_c_pressure_all_aiida": 13557.642534563,
        "final_c_pressure_all_aiida__1__": 13557.642534563,
        "final_c_pressure_all_aiida__2__": 13557.642534563,
        "final_c_pressure_all_aiida__3__": 13557.642534563,
        "final_c_pressure_all_aiida__4__": 7.7954752747919e-11,
        "final_c_pressure_all_aiida__5__": -7.1699124440987e-11,
        "final_c_pressure_all_aiida__6__": -8.2285572345026e-11,
    }

    _time_dependent_computes = {
        "Pxx": np.asarray([13557.643, 13557.643]),
        "Pyy": np.asarray([13557.643, 13557.643]),
        "Pzz": np.asarray([13557.643, 13557.643]),
        "Step": np.asarray([0.0, 1.0]),
        "Press": np.asarray([13557.643, 13557.643]),
        "KinEng": np.asarray([0.0, 0.0]),
        "PotEng": np.asarray([-8.2441284, -8.2441284]),
        "TotEng": np.asarray([-8.2441284, -8.2441284]),
        "c_ke_test_aiida": np.asarray([0.0, 0.0]),
        "c_pressure_all_aiida": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__1__": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__2__": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__3__": np.asarray([13557.643, 13557.643]),
        "c_pressure_all_aiida__4__": np.asarray([-1.0821034e-10, 7.7954753e-11]),
        "c_pressure_all_aiida__5__": np.asarray([-6.8330709e-11, -7.1699124e-11]),
        "c_pressure_all_aiida__6__": np.asarray([-8.2766775e-11, -8.2285572e-11]),
    }

    _trajectories = {
        "attributes": {
            "aliases": None,
            "elements": ["Fe"],
            "zip_prefix": "step-",
            "field_names": [
                "c_ke_atom_all_aiida",
                "c_pe_atom_all_aiida",
                "c_stress_atom_all_aiida__1__",
                "c_stress_atom_all_aiida__2__",
                "c_stress_atom_all_aiida__3__",
                "c_stress_atom_all_aiida__4__",
                "c_stress_atom_all_aiida__5__",
                "c_stress_atom_all_aiida__6__",
                "element",
                "id",
                "type",
                "x",
                "y",
                "z",
            ],
            "number_atoms": 2,
            "number_steps": 2,
            "timestep_filename": "timesteps.txt",
            "compression_method": 8,
            "trajectory_filename": "trajectory.zip",
        },
        "step_data": {
            0: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["0.0000000000", "1.4240580000"],
                "y": ["0.0000000000", "1.4240580000"],
                "z": ["0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113650",
                    "-156612.7819113652",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000009"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            1: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113636",
                    "-156612.7819113671",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113637",
                    "-156612.7819113671",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113644",
                    "-156612.7819113675",
                ],
                "c_stress_atom_all_aiida__4__": ["-0.0000000012", "-0.0000000011"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__6__": ["0.0000000009", "0.0000000009"],
            },
        },
    }

    reference_data.results = _results
    reference_data.time_dependent_computes = _time_dependent_computes
    reference_data.trajectories = _trajectories

    return reference_data


def md_parameters_nve() -> AttributeDict:
    """
    Set of parameters for a md calculation using the nve integration

    :return: parameters for a md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
    }
    parameters.md = {
        "integration": {
            "style": "nve",
        },
        "max_number_steps": 5000,
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 1000,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


def md_reference_data_nve():
    reference_data = AttributeDict()

    _results = {
        "final_ke": 1.3911828389436e-30,
        "final_pe": -8.2441284132802,
        "final_pxx": 13557.642534563,
        "final_pyy": 13557.642534563,
        "final_pzz": 13557.642534563,
        "final_step": 5000,
        "final_press": 13557.642534563,
        "final_etotal": -8.2441284132802,
        "compute_variables": {
            "bin": "standard",
            "bins": [1, 1, 1],
            "binsize": 3.65,
            "units_style": "metal",
            "total_wall_time": "0:00:00",
            "steps_per_second": 159046.5,
            "ghost_atom_cutoff": 7.3,
            "max_neighbors_atom": 2000,
            "master_list_distance_cutoff": 7.3,
        },
        "final_c_pressure_all_aiida": 13557.642534563,
        "final_c_pressure_all_aiida__1__": 13557.642534563,
        "final_c_pressure_all_aiida__2__": 13557.642534563,
        "final_c_pressure_all_aiida__3__": 13557.642534563,
        "final_c_pressure_all_aiida__4__": -1.0688703366748e-10,
        "final_c_pressure_all_aiida__5__": -9.2150216982879e-11,
        "final_c_pressure_all_aiida__6__": -1.0634568121784e-10,
    }

    _time_dependent_computes = {
        "Pxx": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "Pyy": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "Pzz": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "Step": np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
        "Press": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "KinEng": np.array(
            [
                0.0000000e00,
                2.4085709e-31,
                9.6342174e-31,
                2.0657100e-30,
                2.5806065e-30,
                1.3911828e-30,
            ]
        ),
        "PotEng": np.array(
            [-8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284]
        ),
        "TotEng": np.array(
            [-8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284]
        ),
        "c_pressure_all_aiida": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__1__": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__2__": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__3__": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__4__": np.array(
            [
                -1.0821034e-10,
                -1.0821034e-10,
                -1.0821034e-10,
                -7.8977307e-11,
                -1.4375915e-11,
                -1.0688703e-10,
            ]
        ),
        "c_pressure_all_aiida__5__": np.array(
            [
                -6.8330709e-11,
                -8.3729179e-11,
                -8.3729179e-11,
                -8.1804370e-11,
                -2.2857103e-11,
                -9.2150217e-11,
            ]
        ),
        "c_pressure_all_aiida__6__": np.array(
            [
                -8.2766775e-11,
                -8.2766775e-11,
                -8.2766775e-11,
                -8.0841966e-11,
                -1.0105246e-11,
                -1.0634568e-10,
            ]
        ),
    }

    _trajectories = {
        "attributes": {
            "aliases": None,
            "elements": ["Fe"],
            "zip_prefix": "step-",
            "field_names": [
                "c_ke_atom_all_aiida",
                "c_pe_atom_all_aiida",
                "c_stress_atom_all_aiida__1__",
                "c_stress_atom_all_aiida__2__",
                "c_stress_atom_all_aiida__3__",
                "c_stress_atom_all_aiida__4__",
                "c_stress_atom_all_aiida__5__",
                "c_stress_atom_all_aiida__6__",
                "element",
                "id",
                "type",
                "x",
                "y",
                "z",
            ],
            "number_atoms": 2,
            "number_steps": 6,
            "timestep_filename": "timesteps.txt",
            "compression_method": 8,
            "trajectory_filename": "trajectory.zip",
        },
        "step_data": {
            0: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["0.0000000000", "1.4240580000"],
                "y": ["0.0000000000", "1.4240580000"],
                "z": ["0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113650",
                    "-156612.7819113652",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000009"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            1: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113653",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000010"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            2: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113653",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000010"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            3: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113652",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113657",
                    "-156612.7819113656",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000009", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000009"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000008"],
            },
            4: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113708",
                    "-156612.7819113710",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113708",
                    "-156612.7819113711",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113710",
                    "-156612.7819113708",
                ],
                "c_stress_atom_all_aiida__4__": ["-0.0000000000", "-0.0000000001"],
                "c_stress_atom_all_aiida__5__": ["-0.0000000000", "0.0000000002"],
                "c_stress_atom_all_aiida__6__": ["0.0000000001", "0.0000000001"],
            },
            5: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113654",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113654",
                    "-156612.7819113656",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113661",
                    "-156612.7819113657",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000010"],
                "c_stress_atom_all_aiida__6__": ["0.0000000009", "0.0000000010"],
            },
        },
    }

    reference_data.results = _results
    reference_data.time_dependent_computes = _time_dependent_computes
    reference_data.trajectories = _trajectories

    return reference_data


def md_parameters_nvt() -> AttributeDict:
    """
    Set of parameters for a md calculation using the nvt integration

    :return: parameters for a md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
    }
    parameters.md = {
        "integration": {
            "style": "nvt",
            "constraints": {
                "temp": [400, 400, 100],
            },
        },
        "max_number_steps": 5000,
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 1000,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


def md_reference_data_nvt() -> AttributeDict:

    reference_data = AttributeDict()

    _results = {
        "final_ke": 1.3911832212068e-30,
        "final_pe": -8.2441284132802,
        "final_pxx": 13557.642534563,
        "final_pyy": 13557.642534563,
        "final_pzz": 13557.642534563,
        "final_step": 5000,
        "final_press": 13557.642534563,
        "final_etotal": -8.2441284132802,
        "compute_variables": {
            "bin": "standard",
            "bins": [1, 1, 1],
            "binsize": 3.65,
            "units_style": "metal",
            "total_wall_time": "0:00:00",
            "steps_per_second": 142157.953,
            "ghost_atom_cutoff": 7.3,
            "max_neighbors_atom": 2000,
            "master_list_distance_cutoff": 7.3,
        },
        "final_c_pressure_all_aiida": 13557.642534563,
        "final_c_pressure_all_aiida__1__": 13557.642534563,
        "final_c_pressure_all_aiida__2__": 13557.642534563,
        "final_c_pressure_all_aiida__3__": 13557.642534563,
        "final_c_pressure_all_aiida__4__": -1.0688703366748e-10,
        "final_c_pressure_all_aiida__5__": -9.2150216982879e-11,
        "final_c_pressure_all_aiida__6__": -1.0634568121784e-10,
    }

    _time_dependent_computes = {
        "Pxx": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "Pyy": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "Pzz": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "Step": np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
        "Press": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "KinEng": np.array(
            [
                0.0000000e00,
                2.4085710e-31,
                9.6342177e-31,
                2.0657102e-30,
                2.5806068e-30,
                1.3911832e-30,
            ]
        ),
        "PotEng": np.array(
            [-8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284]
        ),
        "TotEng": np.array(
            [-8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284, -8.2441284]
        ),
        "c_pressure_all_aiida": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__1__": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__2__": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__3__": np.array(
            [13557.643, 13557.643, 13557.643, 13557.643, 13557.643, 13557.643]
        ),
        "c_pressure_all_aiida__4__": np.array(
            [
                -1.0821034e-10,
                -1.0821034e-10,
                -1.0821034e-10,
                -7.8977307e-11,
                -1.4375915e-11,
                -1.0688703e-10,
            ]
        ),
        "c_pressure_all_aiida__5__": np.array(
            [
                -6.8330709e-11,
                -8.3729179e-11,
                -8.3729179e-11,
                -8.1804370e-11,
                -2.2857103e-11,
                -9.2150217e-11,
            ]
        ),
        "c_pressure_all_aiida__6__": np.array(
            [
                -8.2766775e-11,
                -8.2766775e-11,
                -8.2766775e-11,
                -8.0841966e-11,
                -1.0105246e-11,
                -1.0634568e-10,
            ]
        ),
    }

    _trajectories = {
        "attributes": {
            "aliases": None,
            "elements": ["Fe"],
            "zip_prefix": "step-",
            "field_names": [
                "c_ke_atom_all_aiida",
                "c_pe_atom_all_aiida",
                "c_stress_atom_all_aiida__1__",
                "c_stress_atom_all_aiida__2__",
                "c_stress_atom_all_aiida__3__",
                "c_stress_atom_all_aiida__4__",
                "c_stress_atom_all_aiida__5__",
                "c_stress_atom_all_aiida__6__",
                "element",
                "id",
                "type",
                "x",
                "y",
                "z",
            ],
            "number_atoms": 2,
            "number_steps": 6,
            "timestep_filename": "timesteps.txt",
            "compression_method": 8,
            "trajectory_filename": "trajectory.zip",
        },
        "step_data": {
            0: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["0.0000000000", "1.4240580000"],
                "y": ["0.0000000000", "1.4240580000"],
                "z": ["0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113650",
                    "-156612.7819113652",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000009"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            1: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113653",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000010"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            2: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113653",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113659",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000010"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000009"],
            },
            3: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113652",
                    "-156612.7819113653",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113651",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113657",
                    "-156612.7819113656",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000009", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000009"],
                "c_stress_atom_all_aiida__6__": ["0.0000000008", "0.0000000008"],
            },
            4: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113708",
                    "-156612.7819113710",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113708",
                    "-156612.7819113711",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113710",
                    "-156612.7819113708",
                ],
                "c_stress_atom_all_aiida__4__": ["-0.0000000000", "-0.0000000001"],
                "c_stress_atom_all_aiida__5__": ["-0.0000000000", "0.0000000002"],
                "c_stress_atom_all_aiida__6__": ["0.0000000001", "0.0000000001"],
            },
            5: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0000000000", "1.4240580000"],
                "y": ["-0.0000000000", "1.4240580000"],
                "z": ["-0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0000000000", "0.0000000000"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-156612.7819113654",
                    "-156612.7819113655",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-156612.7819113654",
                    "-156612.7819113656",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-156612.7819113661",
                    "-156612.7819113657",
                ],
                "c_stress_atom_all_aiida__4__": ["0.0000000008", "0.0000000008"],
                "c_stress_atom_all_aiida__5__": ["0.0000000008", "0.0000000010"],
                "c_stress_atom_all_aiida__6__": ["0.0000000009", "0.0000000010"],
            },
        },
    }

    reference_data.results = _results
    reference_data.time_dependent_computes = _time_dependent_computes
    reference_data.trajectories = _trajectories

    return reference_data


def md_parameters_npt() -> AttributeDict:
    """
    Set of parameters for a md calculation using the npt integration

    :return: parameters for a md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.control = AttributeDict()
    parameters.control.units = "metal"
    parameters.control.timestep = 1e-5
    parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
    }
    parameters.md = {
        "integration": {
            "style": "npt",
            "constraints": {
                "temp": [400, 400, 100],
                "iso": [0.0, 0.0, 1000.0],
            },
        },
        "max_number_steps": 5000,
        "velocity": [{"create": {"temp": 300, "seed": 1}, "group": "all"}],
    }

    parameters.structure = {"atom_style": "atomic"}
    parameters.potential = {}
    parameters.thermo = {
        "printing_rate": 1000,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    parameters.dump = {"dump_rate": 1000}

    return parameters


def md_reference_data_npt() -> AttributeDict:
    reference_data = AttributeDict()

    _results = {
        "final_ke": 0.024351081166091,
        "final_pe": -8.2297014499191,
        "final_pxx": 13762.351231266,
        "final_pyy": 13821.633441769,
        "final_pzz": 13702.34933049,
        "final_step": 5000,
        "final_press": 13762.111334508,
        "final_etotal": -8.205350368753,
        "compute_variables": {
            "bin": "standard",
            "bins": [1, 1, 1],
            "binsize": 3.65,
            "units_style": "metal",
            "total_wall_time": "0:00:00",
            "steps_per_second": 116607.779,
            "ghost_atom_cutoff": 7.3,
            "max_neighbors_atom": 2000,
            "master_list_distance_cutoff": 7.3,
        },
        "final_c_pressure_all_aiida": 13762.111334508,
        "final_c_pressure_all_aiida__1__": 13762.351231266,
        "final_c_pressure_all_aiida__2__": 13821.633441769,
        "final_c_pressure_all_aiida__3__": 13702.34933049,
        "final_c_pressure_all_aiida__4__": 939.85608325067,
        "final_c_pressure_all_aiida__5__": -1088.2253349127,
        "final_c_pressure_all_aiida__6__": -1015.7429644287,
    }

    _time_dependent_computes = {
        "Pxx": np.array(
            [15273.577, 14111.15, 13483.591, 13818.478, 13560.962, 13762.351]
        ),
        "Pyy": np.array(
            [14870.864, 14026.656, 14044.07, 14731.341, 14246.554, 13821.633]
        ),
        "Pzz": np.array(
            [15906.888, 14242.92, 12595.777, 12363.512, 12498.808, 13702.349]
        ),
        "Step": np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
        "Press": np.array(
            [15350.443, 14126.909, 13374.479, 13637.777, 13435.441, 13762.111]
        ),
        "KinEng": np.array(
            [0.03877804, 0.02939861, 0.01080534, 0.00017879, 0.00641852, 0.02435108]
        ),
        "PotEng": np.array(
            [-8.2441284, -8.234749, -8.2161557, -8.2055292, -8.2117689, -8.2297014]
        ),
        "TotEng": np.array(
            [-8.2053504, -8.2053504, -8.2053504, -8.2053504, -8.2053504, -8.2053504]
        ),
        "c_pressure_all_aiida": np.array(
            [15350.443, 14126.909, 13374.479, 13637.777, 13435.441, 13762.111]
        ),
        "c_pressure_all_aiida__1__": np.array(
            [15273.577, 14111.15, 13483.591, 13818.478, 13560.962, 13762.351]
        ),
        "c_pressure_all_aiida__2__": np.array(
            [14870.864, 14026.656, 14044.07, 14731.341, 14246.554, 13821.633]
        ),
        "c_pressure_all_aiida__3__": np.array(
            [15906.888, 14242.92, 12595.777, 12363.512, 12498.808, 13702.349]
        ),
        "c_pressure_all_aiida__4__": np.array(
            [1501.1336, 948.41393, 1604.4957, 2530.7533, 1949.1248, 939.85608]
        ),
        "c_pressure_all_aiida__5__": np.array(
            [-2007.7727, -1202.3279, -1377.2294, -2198.9156, -1680.1421, -1088.2253]
        ),
        "c_pressure_all_aiida__6__": np.array(
            [-1756.4396, -1076.7042, -1476.3543, -2275.9598, -1774.1419, -1015.743]
        ),
    }

    _trajectories = {
        "attributes": {
            "aliases": None,
            "elements": ["Fe"],
            "zip_prefix": "step-",
            "field_names": [
                "c_ke_atom_all_aiida",
                "c_pe_atom_all_aiida",
                "c_stress_atom_all_aiida__1__",
                "c_stress_atom_all_aiida__2__",
                "c_stress_atom_all_aiida__3__",
                "c_stress_atom_all_aiida__4__",
                "c_stress_atom_all_aiida__5__",
                "c_stress_atom_all_aiida__6__",
                "element",
                "id",
                "type",
                "x",
                "y",
                "z",
            ],
            "number_atoms": 2,
            "number_steps": 6,
            "timestep_filename": "timesteps.txt",
            "compression_method": 8,
            "trajectory_filename": "trajectory.zip",
        },
        "step_data": {
            0: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["0.0000000000", "1.4240580000"],
                "y": ["0.0000000000", "1.4240580000"],
                "z": ["0.0000000000", "1.4240580000"],
                "c_ke_atom_all_aiida": ["0.0193890218", "0.0193890218"],
                "c_pe_atom_all_aiida": ["-4.1220642066", "-4.1220642066"],
                "c_stress_atom_all_aiida__1__": [
                    "-176434.6085209166",
                    "-176434.6085209168",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-171782.6237327596",
                    "-171782.6237327599",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-183750.3834920975",
                    "-183750.3834920971",
                ],
                "c_stress_atom_all_aiida__4__": [
                    "-17340.5298153770",
                    "-17340.5298153771",
                ],
                "c_stress_atom_all_aiida__5__": [
                    "23193.0341510633",
                    "23193.0341510635",
                ],
                "c_stress_atom_all_aiida__6__": [
                    "20289.7295051402",
                    "20289.7295051403",
                ],
            },
            1: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0139829256", "1.4380409254"],
                "y": ["-0.0122325735", "1.4362905733"],
                "z": ["0.0163610454", "1.4076969544"],
                "c_ke_atom_all_aiida": ["0.0146993037", "0.0146993037"],
                "c_pe_atom_all_aiida": ["-4.1173744882", "-4.1173744882"],
                "c_stress_atom_all_aiida__1__": [
                    "-163006.6971724575",
                    "-163006.6971724574",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-162030.6516712176",
                    "-162030.6516712145",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-164528.8447729704",
                    "-164528.8447729735",
                ],
                "c_stress_atom_all_aiida__4__": [
                    "-10955.7206759443",
                    "-10955.7206759443",
                ],
                "c_stress_atom_all_aiida__5__": [
                    "13888.8391453118",
                    "13888.8391453117",
                ],
                "c_stress_atom_all_aiida__6__": [
                    "12437.6819741621",
                    "12437.6819741621",
                ],
            },
            2: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0244113953", "1.4484693947"],
                "y": ["-0.0213634887", "1.4454214881"],
                "z": ["0.0285525864", "1.3955054130"],
                "c_ke_atom_all_aiida": ["0.0054026702", "0.0054026702"],
                "c_pe_atom_all_aiida": ["-4.1080778541", "-4.1080778541"],
                "c_stress_atom_all_aiida__1__": [
                    "-155757.3668355147",
                    "-155757.3668355122",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-162231.8036250739",
                    "-162231.8036250749",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-145501.6770453191",
                    "-145501.6770453168",
                ],
                "c_stress_atom_all_aiida__4__": [
                    "-18534.5306563211",
                    "-18534.5306563211",
                ],
                "c_stress_atom_all_aiida__5__": [
                    "15909.2355504872",
                    "15909.2355504871",
                ],
                "c_stress_atom_all_aiida__6__": [
                    "17054.2890123824",
                    "17054.2890123825",
                ],
            },
            3: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0288588304", "1.4529168291"],
                "y": ["-0.0252908210", "1.4493488196"],
                "z": ["0.0336963371", "1.3903616615"],
                "c_ke_atom_all_aiida": ["0.0000893972", "0.0000893972"],
                "c_pe_atom_all_aiida": ["-4.1027645810", "-4.1027645810"],
                "c_stress_atom_all_aiida__1__": [
                    "-159625.8546568954",
                    "-159625.8546568929",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-170170.8993737693",
                    "-170170.8993737668",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-142818.6320433146",
                    "-142818.6320433151",
                ],
                "c_stress_atom_all_aiida__4__": [
                    "-29234.3092923942",
                    "-29234.3092923942",
                ],
                "c_stress_atom_all_aiida__5__": [
                    "25401.0454787973",
                    "25401.0454787973",
                ],
                "c_stress_atom_all_aiida__6__": [
                    "26291.0312476375",
                    "26291.0312476377",
                ],
            },
            4: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0263638127", "1.4504218104"],
                "y": ["-0.0231663781", "1.4472243757"],
                "z": ["0.0306562511", "1.3934017466"],
                "c_ke_atom_all_aiida": ["0.0032092610", "0.0032092610"],
                "c_pe_atom_all_aiida": ["-4.1058844450", "-4.1058844450"],
                "c_stress_atom_all_aiida__1__": [
                    "-156651.1289884508",
                    "-156651.1289884503",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-164570.8251617675",
                    "-164570.8251617675",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-144381.5220458810",
                    "-144381.5220458809",
                ],
                "c_stress_atom_all_aiida__4__": [
                    "-22515.5559017788",
                    "-22515.5559017789",
                ],
                "c_stress_atom_all_aiida__5__": [
                    "19408.3687962359",
                    "19408.3687962358",
                ],
                "c_stress_atom_all_aiida__6__": [
                    "20494.2188914648",
                    "20494.2188914648",
                ],
            },
            5: {
                "id": ["1", "2"],
                "type": ["1", "2"],
                "element": ["Fe", "Fe"],
                "x": ["-0.0174596488", "1.4415176452"],
                "y": ["-0.0154424555", "1.4395004519"],
                "z": ["0.0201027919", "1.4039552045"],
                "c_ke_atom_all_aiida": ["0.0121755406", "0.0121755406"],
                "c_pe_atom_all_aiida": ["-4.1148507250", "-4.1148507250"],
                "c_stress_atom_all_aiida__1__": [
                    "-158977.5009067586",
                    "-158977.5009067596",
                ],
                "c_stress_atom_all_aiida__2__": [
                    "-159662.3066870800",
                    "-159662.3066870797",
                ],
                "c_stress_atom_all_aiida__3__": [
                    "-158284.3815353101",
                    "-158284.3815353100",
                ],
                "c_stress_atom_all_aiida__4__": [
                    "-10856.8636867633",
                    "-10856.8636867633",
                ],
                "c_stress_atom_all_aiida__5__": [
                    "12570.7694318106",
                    "12570.7694318104",
                ],
                "c_stress_atom_all_aiida__6__": [
                    "11733.4803722818",
                    "11733.4803722818",
                ],
            },
        },
    }

    reference_data.results = _results
    reference_data.time_dependent_computes = _time_dependent_computes
    reference_data.trajectories = _trajectories

    return reference_data
