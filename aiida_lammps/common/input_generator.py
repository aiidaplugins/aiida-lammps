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
from typing import Union
from aiida import orm


def write_control_block(parameters_control: dict = None) -> str:
    """
    Generate the input block with global control options.

    This takes the general options that affect the entire simulation, these
    are then used (or their default values) to generate the control block.

    :param parameters_control: dictionary with the basic control parameters, defaults to None
    :type parameters_control: dict, optional
    :return: control block with general parameters of the simulation.
    :rtype: str
    """

    default_timestep = {
        'si': 1.0e-8,
        'lj': 5.0e-3,
        'real': 1.0,
        'metal': 1.0e-3,
        'cgs': 1.0e-8,
        'electron': 1.0e-3,
        'micro': 2.0,
        'nano': 4.5e-4,
    }

    _time = default_timestep[parameters_control.get('units', 'si')]
    control_block = '# ---- Start of the Control information ----\n'
    control_block += 'clear\n'
    control_block += f"units {parameters_control.get('units', 'si')}\n"
    control_block += f"newton {parameters_control.get('newton', 'on')}\n"
    if 'processors' in parameters_control:
        control_block += f"processors {join_keywords(parameters_control['processors'])}\n"
    control_block += f"timestep {parameters_control.get('timestep', _time)}\n"
    control_block += '# ---- End of the Control information ----\n'
    return control_block


def write_potential_block(
    potential=None,
    parameters_potential: dict = None,
) -> str:
    """
    Generate the input block with potential options.

    This will take into consideration the type of potential, as well as other
    parameters which affect the usage of the potential (such as neighbor information)
    and generate a block that is written in the LAMMPS input file.

    :param potential: md-potential which will be used in the calculation, defaults to None
    :type potential: [type], optional
    :param parameters_potential: parameters which have to deal with the potential, defaults to None
    :type parameters_potential: dict, optional
    :return: block with the information needed to setup the potential part of
    the LAMMPS calculation.
    :rtype: str
    """
    potential_block = '# ---- Start of Potential information ----\n'
    potential_block += f'pair_style {potential.pair_style}\n'
    potential_block += f'{potential.potential_line}'
    if 'neighbor' in parameters_potential:
        potential_block += f"neighbor {join_keywords(parameters_potential['neighbor_update'])}\n"
    if 'neigh_modify' in parameters_potential:
        potential_block += f"neigh_modify {(parameters_potential['neigh_modify'])}\n"
    potential_block += '# ---- End of Potential information ----\n'
    return potential_block


def write_structure_block(
    parameters_structure: dict = None,
    structure: orm.StructureData = None,
    structure_filename: str = None,
) -> Union[str, list]:
    """
    Generate the input block with the structure options.

    Takes the AiiDA StructureData as well as as a series of user defined
    parameters to generate the structure related input block.
    This is also responsible of defining the distinct groups that can then
    be used for different compute and/or fixes operations.

    :param parameters_structure: set of user defined parameters relating to the
    structure, defaults to None
    :type parameters_structure: dict, optional
    :param structure: structure that will be studied, defaults to None
    :type structure: orm.StructureData, optional
    :param structure_filename: name of the file where the structure will be
    written so that LAMMPS can read it, defaults to None
    :type structure_filename: str, optional
    :return: block with the structural information and list of groups present
    :rtype: Union[str, list]
    """

    group_names = []

    kind_name_id_map = {}
    for site in structure.sites:
        if site.kind_name not in kind_name_id_map:
            kind_name_id_map[site.kind_name] = len(kind_name_id_map) + 1

    structure_block = '# ---- Start of the Structure information ----\n'
    structure_block += f"box tilt {parameters_structure.get('box_tilt','small')}\n"

    structure_block += f"dimension {structure.get_dimensionality()['dim']}\n"
    structure_block += 'boundary '
    for _bound in ['pbc1', 'pbc2', 'pbc3']:
        structure_block += f"{'p' if structure.attributes[_bound] else 'f'} "
    structure_block += '\n'
    structure_block += f"atom_style {parameters_structure['atom_style']}\n"
    structure_block += f'read_data {structure_filename}\n'
    # Set the groups which will be used for the calculations
    if 'groups' in parameters_structure:
        for _group in parameters_structure['group']:
            # Check if the given type name corresponds to the ones assigned to the atom types
            if 'type' in _group['args']:
                assert all(
                    kind in kind_name_id_map.values()
                    for kind in _group['args'][_group['args'].index('type') +
                                               1:]), 'atom type not defined'
            # Set the current group
            structure_block += f"group {_group['name']} {join_keywords(_group['args'])}\n"
            # Store the name of the group for later usage
            group_names.append(_group['name'])
    structure_block += '# ---- End of the Structure information ----\n'

    return structure_block, group_names


def write_minimize_block(parameters_minimize: dict = None) -> str:
    """
    Generate the input block with the minimization options.

    If the user wishes to do a minimization calculation the parameters will be passed
    to this routine and the necessary block for the input file will be generated.

    .. note: this mode is mutually exclusive with the md mode.

    :param parameters_minimize: user defined parameters for the minimization, defaults to None
    :type parameters_minimize: dict, optional
    :return: block with the minimization options.
    :rtype: str
    """
    minimize_block = '# ---- Start of the Minimization information ----\n'
    minimize_block += f"min_style {parameters_minimize.get('style', 'cg')}\n"
    minimize_block += f"minimize {parameters_minimize.get('energy_tolerance', 1e-4)}"
    minimize_block += f" {parameters_minimize.get('force_tolerance', 1e-4)}"
    minimize_block += f" {parameters_minimize.get('max_iterations', 1000)}"
    minimize_block += f" {parameters_minimize.get('max_evaluations', 1000)}\n"
    minimize_block += '# ---- End of the Minimization information ----\n'

    return minimize_block


def write_md_block(parameters_md: dict = None) -> str:
    """
    Generate the input block with the MD options.

    If the user wishes to perform an MD run this will take the user defined
    parameters and set them in a LAMMPS compliant form.

    .. note: For MD to function an integrator must be provided, this is done
    by providing a fix in the fix part of the input. The existence of at least one
    integrator is checked by the schema.

    .. note: this mode is mutually exclusive with the minimize mode.

    :param parameters_md: user defined parameters for the MD run, defaults to None
    :type parameters_md: dict, optional
    :return: block with the MD options.
    :rtype: str
    """

    md_block = '# ---- Start of the MD information ----\n'
    md_block += 'reset_timestep 0\n'
    if parameters_md.get('run_style', 'verlet') == 'rspa':
        md_block += f"run_style {parameters_md.get('run_style', 'verlet')} "
        md_block += f"{join_keywords(parameters_md['rspa_options'])}\n"
    else:
        md_block += f"run_style {parameters_md.get('run_style', 'verlet')}\n"
    md_block += f"run {parameters_md.get('max_number_steps', 10)}\n"
    md_block += '# ---- End of the MD information ----\n'

    return md_block


def write_fix_block(
    parameters_fix: dict = None,
    group_names: list = None,
) -> Union[str, list]:
    """
    Generate the input block with the fix options.

    This takes the user defined fixes and generates a block where each one of
    them is defined. They can be applied to different groups which can be
    selected by the user and are checked to exist with the previously defined groups
    in the structure setup.

    ..note: fixes which are incompatible with the minimize option are checked by
    the validation schema.

    ..note: the md mode required one of the integrators (nve, nvt, etc) to be defined
    their existence is checked by the schema.

    :param parameters_fix: fixes that will be applied to the calculation, defaults to None
    :type parameters_fix: dict, optional
    :param group_names: list of groups names as defined during structure
    generation, defaults to None
    :type group_names: list, optional
    :return: block with the fixes information, list of applied fixes
    :rtype: Union[str, list]
    """

    fixes_list = []

    fix_block = '# ---- Start of the Fix information ----\n'
    for key, value in parameters_fix.items():
        _group = value.get('group', 'all')
        assert _group in group_names + ['all'], 'group name not defined'
        fix_block += f'fix {generate_id_tag(key, _group)} {_group} {key} '
        fix_block += f"{join_keywords(value['type'])}\n"
        fixes_list.append(generate_id_tag(key, _group))
    fix_block += '# ---- End of the Fix information ----\n'
    return fix_block, fixes_list


def write_compute_block(
    parameters_compute: dict = None,
    group_names: list = None,
) -> Union[str, list]:
    """
    Generate the input block with the compute options.

    This takes the user defined computes and generates a block where each one of
    them is defined. They can be applied to different groups which can be
    selected by the user and are checked to exist with the previously defined groups
    in the structure setup.

    :param parameters_compute: computes that will be applied to the calculation,
    defaults to None
    :type parameters_compute: dict, optional
    :param group_names: list of groups names as defined during structure
    generation, defaults to None
    :type group_names: list, optional
    :return: block with the computes information, list with all the applied computes
    :rtype: Union[str, list]
    """

    computes_list = []

    compute_block = '# ---- Start of the Compute information ----\n'
    for key, value in parameters_compute.items():
        _group = value.get('group', 'all')
        assert _group in group_names + ['all'], 'group name not defined'
        compute_block += f'compute {generate_id_tag(key, _group)} {_group} {key} '
        compute_block += f"{join_keywords(value['type'])}\n"
        computes_list.append(generate_id_tag(key, _group))

    compute_block += '# ---- End of the Compute information ----\n'
    return compute_block, computes_list


def write_dump_block(
    parameters_dump: dict,
    trajectory_filename: str,
    atom_style: str,
    computes_list: list = None,
    fixes_list: list = None,
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
    :param computes_list: list with all the computes set in this calculation, defaults to None
    :type computes_list: list, optional
    :param fixes_list: list with all the fixes defined for the calculation, defaults to None
    :type fixes_list: list, optional
    :return: block with the dump options for the calculation
    :rtype: str
    """
    site_specific_computes = [
        compute for compute in computes_list if '_atom_' in compute
    ]

    site_specific_fixes = [fix for fix in fixes_list if 'ave_' in fix]

    dump_block = '# ---- Start of the Compute information ----\n'
    dump_block += f'dump aiida all custom {parameters_dump.get("dump_rate", 10)} '
    dump_block += f'{trajectory_filename} id type element x y z'
    dump_block += f'{" q" if atom_style=="charge" else ""}\n'
    dump_block += '# ---- End of the Compute information ----\n'

    return dump_block


def generate_id_tag(name: str = None, group: str = None) -> str:
    """Generate an id tag for fixes and/or computes.

    To standardize the naming of computes and/or fixes and to ensure that one
    can programatically recreate them their name will consist of the name of the fix/compute
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


def join_keywords(value: list) -> str:
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
    return ' '.join([
        f"{entry['keyword']} {entry['value']}"
        if isinstance(entry, dict) else f'{entry}' for entry in value
    ])
