"""
Set of functions for the generation of the LAMMPS input file

These functions will be called depending on what options are present in the
calculation parameters. The input file is generated via 'blocks', each of these
blocks will be responsible for printing and/or generate the data that needs to
be set in the LAMMPS file in accordance to the user defined parameters.

Certain blocks are conditionally called, e.g. if no fixes are specified the
fixes block is never called, on the other hand the control block is always
called since it is necessary for the functioning of LAMMPS.
"""

from builtins import ValueError
import json
import os
import re
from typing import Any, Optional, Union

from aiida import orm
import numpy as np

from aiida_lammps.data.potential import LammpsPotentialData
from aiida_lammps.parsers.utils import flatten, generate_header


def generate_input_file(
    parameters: dict[str, Any],
    potential: LammpsPotentialData,
    structure: orm.StructureData,
    trajectory_filename: str = "aiida_lammps.trajectory.dump",
    restart_filename: str = "lammps.restart",
    potential_filename: str = "potential.dat",
    structure_filename: str = "structure.dat",
    variables_filename: str = "aiida_lammps.yaml",
    read_restart_filename: Optional[str] = None,
) -> str:
    """
    Generate the text for the lammps input file.

    It takes the parameters and other inputs needed to generate the lammps input
    file. All the required input blocks are always written, whilst some such as
    the compute and fixes block are only written if required.

    .. note:: If the ``read_restart_filename`` is provided the ``read_data``
        command is used to overwrite the structure and set the velocity and
        other parameters from a previous calculation.

    :param parameters: calculation parameters used to control the LAMMPS calculation
    :type parameters: dict
    :param potential: potential used during the LAMMPS calculation
    :type potential: LammpsPotentialData
    :param structure: structure used during the LAMMPS calculation
    :type structure: orm.StructureData
    :param trajectory_filename: filename used to write the trajectory information,
        defaults to 'aiida_lammps.trajectory.dump'
    :type trajectory_filename: str, optional
    :param restart_filename: filename used to write the restart information,
        defaults to 'restart.aiida'
    :type restart_filename: str, optional
    :param potential_filename: filename used to read the potential,
        defaults to 'potential.dat'
    :type potential_filename: str, optional
    :param structure_filename: filename used to read the structure,
        defaults to 'structure.dat'
    :type structure_filename: str, optional
    :param variables_filename: filename used to store the final variables,
        defaults to 'aiida_lammps.yaml'
    :type variables_filename: str, optional
    :param read_restart_filename: filename used to read the restart information,
        defaults to None
    :type read_restart_filename: str, optional
    :return: contents of the lammps input file.
    :rtype: str
    """
    # pylint: disable=too-many-locals, too-many-arguments

    # Set the maximum number of steps
    max_number_steps = 0

    # Generate the control input block
    control_block = write_control_block(
        parameters_control=parameters.get("control", {})
    )

    # Generate the thermo input block
    thermo_block, fixed_thermo = write_thermo_block(
        parameters_thermo=parameters.get("thermo", {}),
        parameters_compute=parameters.get("compute", {}),
    )
    # Generate the minimization input block
    if "minimize" in parameters:
        run_block = write_minimize_block(
            parameters_minimize=parameters.get("minimize", {})
        )
        max_number_steps = parameters.get("minimize", {}).get("max_iterations", 1000)
    # Generate the md input block
    if "md" in parameters:
        run_block = write_md_block(parameters_md=parameters.get("md", {}))
        max_number_steps = parameters.get("md", {}).get("max_number_steps", 100)
    # Generate the structure input block
    structure_block, group_lists = write_structure_block(
        parameters_structure=parameters.get("structure", {}),
        structure=structure,
        structure_filename=structure_filename,
    )
    # Append the read restart to the structure block
    if read_restart_filename is not None:
        read_restart_block = write_read_restart_block(
            restart_filename=read_restart_filename
        )
        structure_block = ""
    else:
        read_restart_block = ""
    # Generate the fix input block
    if "fix" in parameters:
        fix_block = write_fix_block(
            parameters_fix=parameters.get("fix", {}),
            group_names=group_lists,
        )
    else:
        fix_block = ""
    # Generate the compute input block
    if "compute" in parameters:
        compute_block = write_compute_block(
            parameters_compute=parameters.get("compute", {}), group_names=group_lists
        )
    else:
        compute_block = ""
    # Generate the potential input block
    potential_block = write_potential_block(
        parameters_potential=parameters.get("potential", {}),
        potential_file=potential_filename,
        potential=potential,
        structure=structure,
    )
    # Generate the dump input block
    dump_block = write_dump_block(
        parameters_dump=parameters.get("dump", {}),
        parameters_compute=parameters.get("compute", {}),
        trajectory_filename=trajectory_filename,
        atom_style=potential.atom_style,
        kind_symbols=[kind.symbol for kind in structure.kinds],
    )
    # Generate the restart input block
    if "restart" in parameters:
        restart_block = write_restart_block(
            parameters_restart=parameters.get("restart", {}),
            restart_filename=restart_filename,
            max_number_steps=max_number_steps,
        )
    else:
        restart_block = {"final": "", "intermediate": ""}
    # Generate the final variables input block
    final_block = write_final_variables_block(
        fixed_thermo=fixed_thermo,
        final_file=variables_filename,
    )
    # Printing the potential
    input_file = (
        control_block
        + read_restart_block
        + structure_block
        + potential_block
        + fix_block
        + compute_block
        + thermo_block
        + dump_block
        + restart_block["intermediate"]
        + run_block
        + final_block
        + restart_block["final"]
    )
    return input_file


def write_control_block(parameters_control: dict[str, Any]) -> str:
    """
    Generate the input block with global control options.

    This takes the general options that affect the entire simulation, these
    are then used (or their default values) to generate the control block.

    :param parameters_control: dictionary with the basic control parameters
    :type parameters_control: dict
    :return: control block with general parameters of the simulation.
    :rtype: str
    """

    default_timestep = {
        "si": 1.0e-8,
        "lj": 5.0e-3,
        "real": 1.0,
        "metal": 1.0e-3,
        "cgs": 1.0e-8,
        "electron": 1.0e-3,
        "micro": 2.0,
        "nano": 4.5e-4,
    }

    _time = default_timestep[parameters_control.get("units", "si")]
    control_block = generate_header("Start of the Control information")
    control_block += "clear\n"
    control_block += f'units {parameters_control.get("units", "si")}\n'
    control_block += f'newton {parameters_control.get("newton", "on")}\n'
    if "processors" in parameters_control:
        control_block += (
            f'processors {join_keywords(parameters_control["processors"])}\n'
        )
    control_block += f'timestep {parameters_control.get("timestep", _time)}\n'
    control_block += generate_header("End of the Control information")
    return control_block


def write_potential_block(
    potential: LammpsPotentialData,
    structure: orm.StructureData,
    parameters_potential: dict[str, Any],
    potential_file: str,
) -> str:
    """
    Generate the input block with potential options.

    This will take into consideration the type of potential, as well as other
    parameters which affect the usage of the potential (such as neighbor information)
    and generate a block that is written in the LAMMPS input file.

    :param potential: md-potential which will be used in the calculation
    :type potential: LammpsPotentialData,
    :param structure: structure used for the calculation
    :type structure: orm.StructureData
    :param parameters_potential: parameters which have to deal with the potential
    :type parameters_potential: dict
    :param potential_file: filename for the potential to be used.
    :type str:
    :return: block with the information needed to setup the potential part of
        the LAMMPS calculation.
    :rtype: str
    """

    default_potential = LammpsPotentialData.default_potential_info

    kind_symbols = [kind.symbol for kind in structure.kinds]

    potential_block = generate_header("Start of Potential information")
    potential_block += f"pair_style {potential.pair_style}"
    potential_block += (
        f' {" ".join(parameters_potential.get("potential_style_options", [""]))}\n'
    )

    if default_potential[potential.pair_style].get("read_from_file"):
        potential_block += f'pair_coeff * * {potential_file} {" ".join(kind_symbols)}\n'
    if not default_potential[potential.pair_style].get("read_from_file"):
        data = [
            line
            for line in potential.get_content().split("\n")
            if not line.startswith("#") and line
        ]
        potential_block += f'pair_coeff * * {" ".join(data)}\n'

    if "neighbor" in parameters_potential:
        potential_block += (
            f'neighbor {join_keywords(parameters_potential["neighbor"])}\n'
        )
    if "neighbor_modify" in parameters_potential:
        potential_block += "neigh_modify"
        potential_block += (
            f' {join_keywords(parameters_potential["neighbor_modify"])}\n'
        )
    potential_block += generate_header("End of Potential information")
    return potential_block


def write_structure_block(
    parameters_structure: dict[str, Any],
    structure: orm.StructureData,
    structure_filename: str,
) -> tuple[str, list[str]]:
    """
    Generate the input block with the structure options.

    Takes the AiiDA StructureData as well as as a series of user defined
    parameters to generate the structure related input block.
    This is also responsible of defining the distinct groups that can then
    be used for different compute and/or fixes operations.

    :param parameters_structure: set of user defined parameters relating to the
        structure.
    :type parameters_structure: Dict[str, Any]
    :param structure: structure that will be studied
    :type structure: orm.StructureData
    :param structure_filename: name of the file where the structure will be
        written so that LAMMPS can read it
    :type structure_filename: str
    :return: block with the structural information and list of groups present
    :rtype: Tuple[str, List[str]]
    """

    group_names: list[str] = []

    kind_name_id_map: dict[str, int] = {}
    for site in structure.sites:
        if site.kind_name not in kind_name_id_map:
            kind_name_id_map[site.kind_name] = len(kind_name_id_map) + 1

    structure_block = generate_header("Start of the Structure information")
    structure_block += f'box tilt {parameters_structure.get("box_tilt", "small")}\n'

    # Set the dimensions of the structure
    if "dimension" in parameters_structure:
        structure_block += f"dimension {parameters_structure['dimension']}\n"
    else:
        structure_block += f"dimension {structure.get_dimensionality()['dim']}\n"

    # Set the boundary conditions of the structure
    if "boundary" in parameters_structure:
        structure_block += f"boundary {' '.join(parameters_structure['boundary'])} \n"
    else:
        structure_block += f"boundary {' '.join(['p' if entry else 'f' for entry in structure.pbc])} \n"

    # Set the atom style for the structure
    structure_block += f'atom_style {parameters_structure["atom_style"]}\n'
    # Write the command to read the structure from a file
    structure_block += f"read_data {structure_filename}\n"
    # Set the groups which will be used for the calculations
    if "groups" in parameters_structure:
        for _group in parameters_structure["groups"]:
            # Check if the given type name corresponds to the ones assigned to the atom types
            if "type" in _group["args"]:
                _subset = _group["args"][_group["args"].index("type") + 1 :]

                if not all(kind in kind_name_id_map.values() for kind in _subset):
                    raise ValueError("atom type not defined")
            # Set the current group
            structure_block += (
                f'group {_group["name"]} {join_keywords(_group["args"])}\n'
            )
            # Store the name of the group for later usage
            group_names.append(_group["name"])
    structure_block += generate_header("End of the Structure information")
    return structure_block, group_names


def write_minimize_block(parameters_minimize: dict[str, Union[str, float, int]]) -> str:
    """
    Generate the input block with the minimization options.

    If the user wishes to do a minimization calculation the parameters will be passed
    to this routine and the necessary block for the input file will be generated.

    .. note:: this mode is mutually exclusive with the md mode.

    :param parameters_minimize: user defined parameters for the minimization
    :type parameters_minimize: dict
    :return: block with the minimization options.
    :rtype: str
    """

    minimize_block = generate_header("Start of the Minimization information")
    minimize_block += f'min_style {parameters_minimize.get("style", "cg")}\n'
    minimize_block += f'minimize {parameters_minimize.get("energy_tolerance", 1e-4)}'
    minimize_block += f' {parameters_minimize.get("force_tolerance", 1e-4)}'
    minimize_block += f' {parameters_minimize.get("max_iterations", 1000)}'
    minimize_block += f' {parameters_minimize.get("max_evaluations", 1000)}\n'
    minimize_block += generate_header("End of the Minimization information")

    return minimize_block


def write_md_block(parameters_md: dict[str, Any]) -> str:
    """
    Generate the input block with the MD options.

    If the user wishes to perform an MD run this will take the user defined
    parameters and set them in a LAMMPS compliant form.

    .. note:: For MD to function an integrator must be provided, this is done
        by providing a fix in the fix part of the input. The existence of at least one
        integrator is checked by the schema.

    .. note:: this mode is mutually exclusive with the minimize mode.

    :param parameters_md: user defined parameters for the MD run
    :type parameters_md: dict
    :return: block with the MD options.
    :rtype: str
    """

    integration_options = generate_integration_options(
        style=parameters_md["integration"].get("style", "nve"),
        integration_parameters=parameters_md["integration"].get("constraints"),
    )
    time_step = (
        join_keywords(parameters_md["reset_timestep"])
        if "reset_timestep" in parameters_md
        else 0
    )

    md_block = generate_header("Start of the MD information")
    _key = parameters_md["integration"].get("style", "nve")
    md_block += f'fix {generate_id_tag(_key, "all")} all {_key}{integration_options}\n'
    if "velocity" in parameters_md:
        md_block += (
            f'{generate_velocity_string(parameters_velocity=parameters_md["velocity"])}'
        )
    md_block += f"reset_timestep {time_step}\n"
    if parameters_md.get("run_style", "verlet") == "respa":
        md_block += f'run_style {parameters_md.get("run_style", "verlet")} '
        md_block += f'{join_keywords(parameters_md["respa_options"])}\n'
    else:
        md_block += f'run_style {parameters_md.get("run_style", "verlet")}\n'
    md_block += f'run {parameters_md.get("max_number_steps", 100)}\n'
    md_block += generate_header("End of the MD information")

    return md_block


def write_final_variables_block(
    fixed_thermo: list[str],
    final_file: str = "aiida_lammps.yaml",
) -> str:
    """
    Generate the block to print the final values of the compute variables.

    This takes all the global computes and other thermo values evaluated in
    the simulation and appends them to a final yaml file for recovery.

    :param fixed_thermo: list of thermo values (including computes)
    :type fixed_thermo: list
    :return: input block with the final variables of the calculation.
    :rtype: str
    """

    _variables = []

    variables_block = generate_header("Start of the Final Variables information")

    for _thermo in fixed_thermo:
        _variables.append(re.sub("[^a-zA-Z0-9_]", "__", _thermo))
        variables_block += f"variable final_{_variables[-1]} equal {_thermo}\n"
    variables_block += generate_header("End of the Final Variables information")

    variables_block += generate_header(
        "Start of the Printing Final Variables information"
    )
    variables_block += f'print "#Final results" file {final_file}\n'
    for variable in _variables:
        variables_block += (
            f'print "final_{variable}: ${{final_{variable}}}" append {final_file}\n'
        )
    variables_block += generate_header(
        "End of the Printing Final Variables information"
    )

    return variables_block


def generate_velocity_string(parameters_velocity: list[dict[str, Any]]) -> str:
    """
    Generate the velocity string for the MD block.

    This takes the different possible velocity settings and generate a string
    which is LAMMPS compatible.

    :param parameters_velocity: dictionary with the velocity parameters
    :type parameters_velocity: dict
    :return: string with the velocity options
    :rtype: str
    """
    options = ""
    for entry in parameters_velocity:
        _options = generate_velocity_options(entry)
        if "create" in entry:
            options += f'velocity {entry.get("group", "all")} create'
            options += f' {entry["create"].get("temp")}'
            options += (
                f' {entry["create"].get("seed", np.random.randint(10000))} {_options}\n'
            )
        if "set" in entry:
            options += f'velocity {entry.get("group", "all")} set'
            options += f' {entry["set"].get("vx", "NULL")}'
            options += f' {entry["set"].get("vy", "NULL")}'
            options += f' {entry["set"].get("vz", "NULL")} {_options}\n'
        if "scale" in entry:
            options += f'velocity {entry.get("group", "all")} scale'
            options += f' {entry["scale"]} {_options}\n'
        if "ramp" in entry:
            options += f'velocity {entry.get("group", "all")} ramp'
            options += f' {entry["ramp"].get("vdim")} {entry["ramp"].get("vlo")}'
            options += f' {entry["ramp"].get("vhi")} {entry["ramp"].get("dim")}'
            options += (
                f' {entry["ramp"].get("clo")} {entry["ramp"].get("chi")} {_options}\n'
            )
        if "zero" in entry:
            options += f'velocity {entry.get("group", "all")} zero'
            options += f' {entry["zero"]} {_options}\n'
    return options


def generate_velocity_options(options_velocity: dict[str, Any]) -> str:
    """
    Generate the options string for every velocity.

    Independent of the way in which one specifies the velocity there are several
    options that are global, this functions allows them to be setup.

    :param options_velocity: dictionary with the velocity parameters
    :type options_velocity: dict
    :return: string with the velocity options
    :rtype: str
    """
    _options = ["dist", "sum", "mom", "rot", "temp", "bias", "loop", "rigid", "units"]

    velocity_option = ""
    for _option in _options:
        if _option in options_velocity:
            velocity_option += f" {_option} {options_velocity[_option]} "
    return velocity_option


def generate_integration_options(
    style: str,
    integration_parameters: dict[str, Any],
) -> str:
    """
    Create a string with the integration options.

    This will check that the appropriate options are setup for each of the
    supported integrators. These will be appended to a string which is then
    passed to each of the integrators.

    :param style: Integration style performed in MD mode
    :type style: str
    :param integration_parameters: dictionary with the constraints for the integration
    :type integration_parameters: dict
    :return: string with the integration options.
    :rtype: str
    """

    temperature_dependent = [
        "nvt",
        "nvt/asphere",
        "nvt/body",
        "nvt/eff",
        "nvt/manifold/rattle",
        "nvt/sllod",
        "nvt/sllod/eff",
        "nvt/sphere",
        "nvt/uef",
        "nphug",
        "npt",
        "npt/asphere",
        "npt/body",
        "npt/cauchy",
        "npt/eff",
        "npt/sphere",
        "npt/uef",
    ]

    pressure_dependent = [
        "nph",
        "nph/asphere",
        "nph/body",
        "nph/eff",
        "nph/sphere",
        "nphug",
        "npt",
        "npt/asphere",
        "npt/body",
        "npt/cauchy",
        "npt/eff",
        "npt/sphere",
        "npt/uef",
    ]

    uef_dependent = ["npt/uef", "nvt/uef"]

    temperature_options = ["temp", "tchain", "tloop", "drag"]

    pressure_options = [
        "ani",
        "iso",
        "tri",
        "x",
        "y",
        "z",
        "xy",
        "xz",
        "yz",
        "couple",
        "pchain",
        "mtk",
        "ploop",
        "nreset",
        "drag",
        "dilate",
        "scaleyz",
        "scalexz",
        "scalexy",
        "flip",
        "fixedpoint",
        "update",
    ]

    uef_options = ["ext", "erotate"]

    options = ""

    # Set the options that depend on the temperature
    if style in temperature_dependent:
        for _option in temperature_options:
            if _option in integration_parameters:
                _value = integration_parameters.get(_option)
                if _value:
                    _value = [str(val) for val in _value]
                    options += f' {_option} {" ".join(_value) if isinstance(_value, list) else _value} '
    # Set the options that depend on the pressure
    if style in pressure_dependent:
        for _option in pressure_options:
            if _option in integration_parameters:
                _value = integration_parameters.get(_option)
                if _value:
                    _value = [str(val) for val in _value]
                    options += f' {_option} {" ".join(_value) if isinstance(_value, list) else _value} '
    # Set the options that depend on the 'uef' parameters
    if style in uef_dependent:
        for _option in uef_options:
            if _option in integration_parameters:
                _value = integration_parameters.get(_option)
                if _value:
                    _value = [str(val) for val in _value]
                    options += f' {_option} {" ".join(_value) if isinstance(_value, list) else _value} '
    # Set the options that depend on the 'nve/limit' parameters
    if style in ["nve/limit"]:
        options += f' {integration_parameters.get("xmax", 0.1)} '
    # Set the options that depend on the 'langevin' parameters
    if style in ["nve/dotc/langevin"]:
        options += f' {integration_parameters.get("temp")}'
        options += f' {integration_parameters.get("seed")}'
        options += f' angmom {integration_parameters.get("angmom")}'
    return options


def write_fix_block(
    parameters_fix: dict[str, Any],
    group_names: Optional[list[str]] = None,
) -> str:
    """
    Generate the input block with the fix options.

    This takes the user defined fixes and generates a block where each one of
    them is defined. They can be applied to different groups which can be
    selected by the user and are checked to exist with the previously defined groups
    in the structure setup.

    .. note:: fixes which are incompatible with the minimize option are checked by
        the validation schema.

    .. note:: the md mode required one of the integrators (nve, nvt, etc) to be defined
        their existence is checked by the schema.

    :param parameters_fix: fixes that will be applied to the calculation
    :type parameters_fix: dict
    :param group_names: list of groups names as defined during structure
        generation, defaults to None
    :type group_names: list, optional
    :return: block with the fixes information
    :rtype: str
    """

    if group_names is None:
        group_names = []

    fix_block = generate_header("Start of the Fix information")
    for key, value in parameters_fix.items():
        for entry in value:
            _group = entry.get("group", "all")
            if _group not in [*group_names, "all"]:
                raise ValueError(
                    f'group name "{_group}" is not the defined groups {[*group_names, "all"]}'
                )
            fix_block += f"fix {generate_id_tag(key, _group)} {_group} {key} "
            fix_block += f'{join_keywords(entry["type"])}\n'
    fix_block += generate_header("End of the Fix information")
    return fix_block


def write_compute_block(
    parameters_compute: dict[str, Any],
    group_names: Optional[list[str]] = None,
) -> str:
    """
    Generate the input block with the compute options.

    This takes the user defined computes and generates a block where each one of
    them is defined. They can be applied to different groups which can be
    selected by the user and are checked to exist with the previously defined groups
    in the structure setup.

    :param parameters_compute: computes that will be applied to the calculation
    :type parameters_compute: dict
    :param group_names: list of groups names as defined during structure
        generation, defaults to None
    :type group_names: list, optional
    :return: block with the computes information
    :rtype: str
    """

    if group_names is None:
        group_names = []

    compute_block = generate_header("Start of the Compute information")
    for key, value in parameters_compute.items():
        for entry in value:
            _group = entry.get("group", "all")
            if _group not in [*group_names, "all"]:
                raise ValueError(f'group name "{_group}" is not the defined groups')
            compute_block += f"compute {generate_id_tag(key, _group)} {_group} {key} "
            compute_block += f'{join_keywords(entry["type"])}\n'
    compute_block += generate_header("End of the Compute information")
    return compute_block


def write_dump_block(
    parameters_dump: dict[str, Any],
    trajectory_filename: str,
    atom_style: str,
    kind_symbols: list[str],
    parameters_compute: Optional[dict[str, Any]] = None,
) -> str:
    """Generate the block with dumps commands.

    This will check for any compute and/or fix that generates atom dependent data
    and will make sure that it is written to file in a controllable manner, so that
    they can be easily parsed afterwards.

    :param parameters_dump: set of user defined parameters for the writing of data
    :type parameters_dump: dict
    :param trajectory_filename: name of the file where the trajectory is written.
    :type trajectory_filename: str
    :param atom_style: which kind of LAMMPS atomic style is used for the calculation.
    :param parameters_compute: computes that will be applied to the calculation
    :type parameters_compute: dict
    :return: block with the dump options for the calculation
    :rtype: str
    """
    # pylint: disable=too-many-locals
    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "variables_types.json",
    )

    with open(_file) as handler:
        _compute_variables = json.load(handler)["computes"]

    computes_list = []

    if parameters_compute:
        for key, value in parameters_compute.items():
            for entry in value:
                _locality = _compute_variables[key]["locality"]
                _printable = _compute_variables[key]["printable"]

                if _locality == "local" and _printable:
                    computes_list.append(
                        generate_printing_string(
                            name=key,
                            group=entry["group"],
                            calculation_type="compute",
                        )
                    )

    num_double = len(list(flatten([compute.split() for compute in computes_list])))
    num_double += 3
    if atom_style == "charge":
        num_double += 1
    dump_block = generate_header("Start of the Dump information")
    dump_block += f'dump aiida all custom {parameters_dump.get("dump_rate", 10)} '
    dump_block += f"{trajectory_filename} id type element x y z "
    dump_block += f'{"q " if atom_style=="charge" else ""}'
    dump_block += f'{" ".join(computes_list)}\n'
    dump_block += "dump_modify aiida sort id\n"
    dump_block += f'dump_modify aiida element {" ".join(kind_symbols)}\n'
    dump_block += "dump_modify aiida format int ' %d ' \n"
    dump_block += "dump_modify aiida format float ' %16.10e ' \n"
    dump_block += generate_header("End of the Dump information")

    return dump_block


def write_thermo_block(
    parameters_thermo: dict[str, Any],
    parameters_compute: Optional[dict[str, Any]] = None,
) -> tuple[str, list[str]]:
    """Generate the block with the thermo command.

    This will take all the global computes which were generated during the calculation
    plus the 'common' thermodynamic parameters set by LAMMPS and set them so that
    they are printed to the LAMMPS output file.

    :param parameters_thermo: user defined parameters to control the output data.
    :type parameters_thermo: dict
    :param parameters_compute: computes that will be applied to the calculation
    :type parameters_compute: dict
    :return: block with the thermo options for the calculation, list of thermo printing variables
    :rtype: Union[str, list]
    """

    printing_variables = []

    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "variables_types.json",
    )

    with open(_file) as handler:
        _compute_variables = json.load(handler)["computes"]

    computes_list = []

    if parameters_compute:
        for key, value in parameters_compute.items():
            for entry in value:
                _locality = _compute_variables[key]["locality"]
                _printable = _compute_variables[key]["printable"]

                if _locality == "global" and _printable:
                    computes_list.append(
                        generate_printing_string(
                            name=key,
                            group=entry["group"],
                            calculation_type="compute",
                        )
                    )

    computes_printing = parameters_thermo.get("thermo_printing", None)

    if computes_printing is None or not computes_printing:
        fixed_thermo = ["step", "temp", "epair", "emol", "etotal", "press"]
    else:
        fixed_thermo = [key for key, value in computes_printing.items() if value]
        if "step" not in fixed_thermo:
            fixed_thermo = ["step", *fixed_thermo]
        if "etotal" not in fixed_thermo:
            fixed_thermo = [*fixed_thermo, "etotal"]

    if fixed_thermo.index("step") != 0:
        fixed_thermo.remove("step")
        fixed_thermo = ["step", *fixed_thermo]

    thermo_block = generate_header("Start of the Thermo information")
    thermo_block += (
        f'thermo_style custom {" ".join(fixed_thermo)} {" ".join(computes_list)}\n'
    )
    thermo_block += f'thermo {parameters_thermo.get("printing_rate", 1000)}\n'
    thermo_block += generate_header("End of the Thermo information")

    printing_variables = fixed_thermo + list(
        flatten([compute.split() for compute in computes_list])
    )

    return thermo_block, printing_variables


def write_restart_block(
    parameters_restart: dict[str, Any], restart_filename: str, max_number_steps: int
) -> dict[str, Any]:
    """Generate the block to write the restart file.

    :param parameters_restart: set of parameters controlling the printing of the restartfile
    :type parameters_restart: dict
    :param restart_filename: Name of the LAMMPS restart file
    :type restart_filename: str
    :param max_number_steps: maximum number of steps in the simulation
    :type max_number_steps: int
    :return: dictionary with the string block indicating the printing of the final restart file and intermediate files.
    :rtype: dict
    """

    restart_block = {"final": "", "intermediate": ""}

    if "print_final" in parameters_restart and parameters_restart["print_final"]:
        restart_block["final"] += generate_header(
            "Start of the write restart information"
        )
        restart_block["final"] += f"write_restart {restart_filename}\n"
        restart_block["final"] += generate_header(
            "End of the write restart information"
        )

    if (
        "print_intermediate" in parameters_restart
        and parameters_restart["print_intermediate"]
    ):
        restart_block["intermediate"] += generate_header(
            "Start of the intermediate write restart information"
        )
        restart_block[
            "intermediate"
        ] += f"restart {parameters_restart.get('num_steps', int(max_number_steps/10))} {restart_filename}\n"
        restart_block["intermediate"] += generate_header(
            "End of the intermediate write restart information"
        )
    return restart_block


def write_read_restart_block(restart_filename: str) -> str:
    """Generate the block to read the restart file.

    :param restart_filename: Name of the LAMMPS restart file
    :type restart_filename: str
    :return: string block indicating the reading of a restart file.
    :rtype: str
    """

    read_restart_block = generate_header("Start of the read restart information")
    read_restart_block += f"read_restart {restart_filename}\n"
    read_restart_block += generate_header("End of the read restart information")
    return read_restart_block


def generate_printing_string(
    name: str,
    group: str,
    calculation_type: str,
) -> str:
    """
    Generate string for the quantities that will be printed.

    The idea is to take the name as well as the group of the parameter
    that one wishes to print, then in conjunction with the information
    stored for each parameter one can generate a string for either the thermo
    or dump commands.

    :param name: Name of the compute/fix that one wishes to print
    :type name: str
    :param group: Name of the group where the compute/fix is calculated
    :type group: str
    :return: string for the compute/fix that will be used for printing
    :rtype: str
    """

    if calculation_type == "compute":
        prefactor = "c"
    if calculation_type == "fix":
        prefactor = "f"

    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "variables_types.json",
    )

    with open(_file) as handler:
        _compute_variables = json.load(handler)["computes"]

    _type = _compute_variables[name]["type"]
    _size = _compute_variables[name]["size"]

    _string = []

    if _type == "vector" and _size > 0:
        for index in range(1, _size + 1):
            _string.append(f"{prefactor}_{generate_id_tag(name, group)}[{index}]")
    elif _type == "vector" and _size == 0:
        _string.append(f"{prefactor}_{generate_id_tag(name, group)}[*]")

    if _type == "mixed" and _size > 0:
        _string.append(f"{prefactor}_{generate_id_tag(name, group)}")
        for index in range(1, _size + 1):
            _string.append(f"{prefactor}_{generate_id_tag(name, group)}[{index}]")
    elif _type == "mixed" and _size == 0:
        _string.append(f"{prefactor}_{generate_id_tag(name, group)}")
        _string.append(f"{prefactor}_{generate_id_tag(name, group)}[*]")

    if _type == "scalar":
        _string.append(f"{prefactor}_{generate_id_tag(name, group)}")

    if _type == "array":
        _string.append(f"{prefactor}_{generate_id_tag(name, group)}")

    return " ".join(_string)


def generate_id_tag(name: str, group: str) -> str:
    """Generate an id tag for fixes and/or computes.

    To standardize the naming of computes and/or fixes and to ensure that one
    can programmatically recreate them their name will consist of the name of the fix/compute
    with the group at which is applied appended plus the aiida keyword. Of this
    way one can always regenerate these tags by knowing which fix/computes
    were asked of the calculation.

    :param name: name of the fix/compute, defaults to None
    :type name: str, optional
    :param group: group which at which the fix/compute will be applied to, defaults to None
    :type group: str, optional
    :return: if tag for the compute/fix
    :rtype: str
    """

    return f"{name.replace('/','_')}_{group}_aiida"


def join_keywords(value: list[Any]) -> str:
    """
    Generate a string for the compute/fix options.

    Depending on the desired fix/compute several options might need to be passed
    to it to dictate its behavior. Having the user pass these options as a single string
    is a bad idea, instead it is simple if the user passes them as a list, where key,value
    pairs dictionaries can be present and or single entries. These items will be
    taken and concatenated to ensure that a LAMMPS compliant string is produced
    out of all these options.

    :param value: list with the options for a given fix/compute
    :type value: list
    :return: LAMMPS compliant string with the fix/compute options
    :rtype: str
    """

    return " ".join(
        [
            (
                f"{entry['keyword']} {entry['value']}"
                if isinstance(entry, dict)
                else f"{entry}"
            )
            for entry in value
        ]
    )
