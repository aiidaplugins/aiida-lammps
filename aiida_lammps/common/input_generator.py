"""
[summary]

[extended_summary]

"""
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
    control_block += f"processors {parameters_control.get('processors','* * * grid onelevel map cart')}\n"
    control_block += f"timestep {parameters_control.get('timestep', _time)}\n"
    control_block += '# ---- End of the Control information ----\n'
    return control_block


def write_potential_block(
    potential=None,
    parameters_potential: dict = None,
) -> str:
    """
    Generate the input block with potential options.

    [extended_summary]

    :param potential: [description], defaults to None
    :type potential: [type], optional
    :param parameters_potential: [description], defaults to None
    :type parameters_potential: dict, optional
    :return: [description]
    :rtype: str
    """
    potential_block = '# ---- Start of Potential information ----\n'
    potential_block += f'pair_style {potential.pair_style}\n'
    potential_block += f'{potential.potential_line}'
    if 'neighbor':
        potential_block += f"neighbor {join_keywords(parameters_potential['neighbor_update'])}\n"
    if 'neigh_modify' in parameters_potential:
        potential_block += f"neigh_modify {(parameters_potential['neigh_modify'])}\n"
    potential_block += '# ---- End of Potential information ----\n'
    return potential_block


def write_structure_block(
    parameters_structure: dict = None,
    structure: orm.StructureData = None,
    structure_filename: str = None,
) -> str:
    """
    Generate the input block with the structure options.

    [extended_summary]

    :param parameters_structure: [description], defaults to None
    :type parameters_structure: dict, optional
    :param structure: [description], defaults to None
    :type structure: orm.StructureData, optional
    :param structure_filename: [description], defaults to None
    :type structure_filename: str, optional
    :return: [description]
    :rtype: str
    """
    structure_block = '# ---- Start of the Structure information ----\n'
    structure_block += f"box tilt {parameters_structure.get('box_tilt','small')}\n"

    structure_block += f"dimension {structure.get_dimensionality()['dim']}\n"
    structure_block += 'boundary '
    for _bound in ['pbc1', 'pbc2', 'pbc3']:
        structure_block += f"{'p' if structure.attributes[_bound] else 'f'} "
    structure_block += '\n'
    structure_block += f"atom_style {parameters_structure['atom_style']}\n"
    structure_block += f'read_data {structure_filename}\n'
    structure_block += '# ---- End of the Structure information ----\n'

    return structure_block


def write_minimize_block(parameters_minimize: dict = None) -> str:
    """
    Generate the input block with the minimization options.

    [extended_summary]

    :param parameters_minimize: [description], defaults to None
    :type parameters_minimize: dict, optional
    :return: [description]
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


def write_md_block(
    parameters_md: dict = None,
    parameters_fix: dict = None,
) -> str:
    """
    Generate the input block with the MD options.

    [extended_summary]

    :param parameters_md: [description], defaults to None
    :type parameters_md: dict, optional
    :return: [description]
    :rtype: str
    """

    integrators = [
        'nvt',
        'nvp',
        'nph',
        'nvt/eff',
        'nvp/eff',
        'nph/eff',
        'nvt/uef',
        'npt/uef',
        'nph/asphere',
        'nph/body',
        'nph/sphere',
    ]

    assert any(x in parameters_fix.keys()
               for x in integrators), 'No integrator provided'

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


def write_fix_block(parameters_fix: dict = None) -> str:
    """
    Generate the input block with the fix options.

    [extended_summary]

    :param parameters_fix: [description], defaults to None
    :type parameters_fix: dict, optional
    :return: [description]
    :rtype: str
    """

    fix_block = '# ---- Start of the Fix information ----\n'
    for key, value in parameters_fix.items():
        fix_block += f"fix {key.replace('/','_')}_aiida {value['group']} {key} "
        fix_block += f"{join_keywords(value['type'])}\n"
    fix_block += '# ---- End of the Fix information ----\n'
    return fix_block


def write_compute_block(parameters_compute: dict = None) -> str:
    """
    Generate the input block with the compute options.

    [extended_summary]

    :param parameters_compute: [description], defaults to None
    :type parameters_compute: dict, optional
    :return: [description]
    :rtype: str
    """
    compute_block = '# ---- Start of the Compute information ----\n'
    for key, value in parameters_compute.items():
        compute_block += f"compute {key.replace('/','_')}_aiida {value['group']} {key} "
        compute_block += f"{join_keywords(value['type'])}\n"
    compute_block += '# ---- End of the Compute information ----\n'
    return compute_block


def join_keywords(value) -> str:
    """
    [summary]

    [extended_summary]

    :param value: [description]
    :type value: [type]
    :return: [description]
    :rtype: str
    """
    return ' '.join([
        f"{entry['keyword']} {entry['value']}"
        if isinstance(entry, dict) else f'{entry}' for entry in value
    ])
