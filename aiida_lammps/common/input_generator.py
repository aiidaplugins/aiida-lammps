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
import os
from typing import Union
import json
from aiida import orm
from aiida_lammps.data.lammps_potential import LammpsPotentialData


def write_control_block(parameters_control: dict) -> str:
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
    potential: LammpsPotentialData,
    structure: orm.StructureData,
    parameters_potential: dict,
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

    default_potential = LammpsPotentialData._default_potential_info

    kind_symbols = [kind.symbol for kind in structure.kinds]

    potential_block = '# ---- Start of Potential information ----\n'
    potential_block += f'pair_style {potential.pair_style}'
    potential_block += f' {" ".join(parameters_potential.get("potential_style_options", [""]))}\n'

    if default_potential[potential.pair_style].get('read_from_file'):
        potential_block += f'pair_coeff * * {potential_file} {" ".join(kind_symbols)}\n'
    if not default_potential[potential.pair_style].get('read_from_file'):
        potential_block += f'pair_coeff {potential.get_content()}\n'

    if 'neighbor' in parameters_potential:
        potential_block += f'neighbor {join_keywords(parameters_potential["neighbor"])}\n'
    if 'neighbor_modify' in parameters_potential:
        potential_block += f'neigh_modify {(parameters_potential["neighbor_modify"])}\n'
    potential_block += '# ---- End of Potential information ----\n'
    return potential_block


def write_structure_block(
    parameters_structure: dict,
    structure: orm.StructureData,
    structure_filename: str,
) -> Union[str, list]:
    """
    Generate the input block with the structure options.

    Takes the AiiDA StructureData as well as as a series of user defined
    parameters to generate the structure related input block.
    This is also responsible of defining the distinct groups that can then
    be used for different compute and/or fixes operations.

    :param parameters_structure: set of user defined parameters relating to the
        structure.
    :type parameters_structure: dict
    :param structure: structure that will be studied
    :type structure: orm.StructureData
    :param structure_filename: name of the file where the structure will be
        written so that LAMMPS can read it
    :type structure_filename: str
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

                _subset = _group['args'][_group['args'].index('type') + 1:]

                assert all(kind in kind_name_id_map.values()
                           for kind in _subset), 'atom type not defined'
            # Set the current group
            structure_block += f"group {_group['name']} {join_keywords(_group['args'])}\n"
            # Store the name of the group for later usage
            group_names.append(_group['name'])
    structure_block += '# ---- End of the Structure information ----\n'

    return structure_block, group_names


def write_minimize_block(parameters_minimize: dict) -> str:
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

    minimize_block = '# ---- Start of the Minimization information ----\n'
    minimize_block += f"min_style {parameters_minimize.get('style', 'cg')}\n"
    minimize_block += f"minimize {parameters_minimize.get('energy_tolerance', 1e-4)}"
    minimize_block += f" {parameters_minimize.get('force_tolerance', 1e-4)}"
    minimize_block += f" {parameters_minimize.get('max_iterations', 1000)}"
    minimize_block += f" {parameters_minimize.get('max_evaluations', 1000)}\n"
    minimize_block += '# ---- End of the Minimization information ----\n'

    return minimize_block


def write_md_block(parameters_md: dict) -> str:
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
    parameters_fix: dict,
    group_names: list = None,
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

    fix_block = '# ---- Start of the Fix information ----\n'
    for key, value in parameters_fix.items():
        for entry in value:
            _group = entry.get('group', 'all')
            assert _group in group_names + ['all'], 'group name not defined'
            fix_block += f'fix {generate_id_tag(key, _group)} {_group} {key} '
            fix_block += f'{join_keywords(entry["type"])}\n'
    fix_block += '# ---- End of the Fix information ----\n'
    return fix_block


def write_compute_block(
    parameters_compute: dict,
    group_names: list = None,
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

    compute_block = '# ---- Start of the Compute information ----\n'
    for key, value in parameters_compute.items():
        for entry in value:
            _group = entry.get('group', 'all')
            assert _group in group_names + ['all'], 'group name not defined'
            compute_block += f'compute {generate_id_tag(key, _group)} {_group} {key} '
            compute_block += f'{join_keywords(entry["type"])}\n'
    compute_block += '# ---- End of the Compute information ----\n'
    return compute_block


def write_dump_block(
    parameters_dump: dict,
    trajectory_filename: str,
    atom_style: str,
    parameters_compute: dict = None,
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

    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'variables_types.json',
    )

    with open(_file, 'r') as handler:
        _compute_variables = json.load(handler)['computes']

    computes_list = []

    for key, value in parameters_compute.items():
        for entry in value:
            _locality = _compute_variables[key]['locality']
            _printable = _compute_variables[key]['printable']

            if _locality == 'local' and _printable:
                computes_list.append(
                    generate_printing_string(
                        name=key,
                        group=entry['group'],
                        calculation_type='compute',
                    ))

    dump_block = '# ---- Start of the Dump information ----\n'
    dump_block += f'dump aiida all custom {parameters_dump.get("dump_rate", 10)} '
    dump_block += f'{trajectory_filename} id type element x y z '
    dump_block += f'{"q " if atom_style=="charge" else ""}'
    dump_block += f'{" ".join(computes_list)}\n'
    dump_block += '# ---- End of the Dump information ----\n'

    return dump_block


def write_thermo_block(
    parameters_thermo: dict,
    parameters_compute: dict = None,
) -> str:
    """Generate the block with the thermo command.

    This will take all the global computes which were generated during the calculation
    plus the 'common' thermodynamic parameters set by LAMMPS and set them so that
    they are printed to the LAMMPS log file.

    :param parameters_thermo: user defined parameters to control the log data.
    :type parameters_thermo: dict
    :param parameters_compute: computes that will be applied to the calculation
    :type parameters_compute: dict
    :return: block with the thermo options for the calculation.
    :rtype: str
    """

    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'variables_types.json',
    )

    with open(_file, 'r') as handler:
        _compute_variables = json.load(handler)['computes']

    computes_list = []

    for key, value in parameters_compute.items():
        for entry in value:
            _locality = _compute_variables[key]['locality']
            _printable = _compute_variables[key]['printable']

            if _locality == 'global' and _printable:
                computes_list.append(
                    generate_printing_string(
                        name=key,
                        group=entry['group'],
                        calculation_type='compute',
                    ))

    computes_printing = parameters_thermo.get('thermo_printing', None)

    if computes_printing is None or not computes_printing:
        fixed_thermo = ['step', 'temp', 'epair', 'emol', 'etotal', 'press']
    else:
        fixed_thermo = [
            key for key, value in computes_printing.items() if value
        ]

    thermo_block = '# ---- Start of the Thermo information ----\n'
    thermo_block += f'thermo_style {" ".join(fixed_thermo)} {" ".join(computes_list)}\n'
    thermo_block += f'thermo {parameters_thermo.get("printing_rate", 1000)}\n'
    thermo_block += '# ---- End of the Thermo information ----\n'

    return thermo_block


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

    if calculation_type == 'compute':
        prefactor = 'c'
    if calculation_type == 'fix':
        prefactor = 'f'

    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'variables_types.json',
    )

    with open(_file, 'r') as handler:
        _compute_variables = json.load(handler)['computes']

    _type = _compute_variables[name]['type']
    _size = _compute_variables[name]['size']

    _string = []

    if _type == 'vector' and _size > 0:
        for index in range(1, _size + 1):
            _string.append(
                f'{prefactor}_{generate_id_tag(name, group)}[{index}]')
    elif _type == 'vector' and _size == 0:
        _string.append(f'{prefactor}_{generate_id_tag(name, group)}[*]')

    if _type == 'mixed' and _size > 0:
        _string.append(f'{prefactor}_{generate_id_tag(name, group)}')
        for index in range(1, _size + 1):
            _string.append(
                f'{prefactor}_{generate_id_tag(name, group)}[{index}]')
    elif _type == 'mixed' and _size == 0:
        _string.append(f'{prefactor}_{generate_id_tag(name, group)}')
        _string.append(f'{prefactor}_{generate_id_tag(name, group)}[*]')

    if _type == 'scalar':
        _string.append(f'{prefactor}_{generate_id_tag(name, group)}')

    if _type == 'array':
        _string.append(f'{prefactor}_{generate_id_tag(name, group)}')

    return ' '.join(_string)


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
