"""Run a multi-stage molecular dynamic simulation."""
# pylint: disable=fixme
from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory

from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, get_path, join_keywords
from aiida_lammps.validation import validate_against_schema


class MdMultiCalculation(BaseLammpsCalculation):
    """Run a multi-stage molecular dynamic simulation."""
    @classmethod
    def define(cls, spec):
        super(MdMultiCalculation, cls).define(spec)

        spec.input(
            'metadata.options.parser_name',
            valid_type=str,
            default='lammps.md.multi',
        )
        spec.default_output_port = 'results'

        spec.output_namespace(
            'system',
            dynamic=True,
            valid_type=DataFactory('array'),
            help='selected system data per dump step of a stage',
        )

        spec.output_namespace(
            'trajectory',
            dynamic=True,
            valid_type=DataFactory('lammps.trajectory'),
            help='atomic configuration data per dump step of a stage',
        )

    @staticmethod
    def create_main_input_content(
        parameter_data,
        potential_data,
        kind_symbols,
        structure_filename,
        trajectory_filename,
        system_filename,
        restart_filename,
    ):
        # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

        pdict = parameter_data.get_dict()
        version_date = convert_date_string(
            pdict.get('lammps_version', '11 Aug 2017'))
        lammps_input_file = ''

        lammps_input_file += f'# Input written to comply with LAMMPS {version_date}\n'

        # Configuration setup
        lammps_input_file += '\n# Atomic Configuration\n'
        lammps_input_file += f'units           {potential_data.default_units}\n'
        lammps_input_file += 'boundary        p p p\n'  # TODO allow non-periodic
        lammps_input_file += 'box tilt large\n'
        lammps_input_file += f'atom_style      {potential_data.atom_style}\n'
        lammps_input_file += f'read_data       {structure_filename}\n'

        # Potential specification
        lammps_input_file += '\n# Potential Setup\n'
        lammps_input_file += potential_data.get_input_lines(kind_symbols)

        # Modify pairwise neighbour list creation
        lammps_input_file += '\n# General Setup\n'
        if 'neighbor' in pdict:
            # neighbor skin_dist bin/nsq/multi
            lammps_input_file += f'neighbor {pdict["neighbor"][0]} {pdict["neighbor"][1]}\n'
        if 'neigh_modify' in pdict:
            # e.g. 'neigh_modify every 1 delay 0 check no\n'
            lammps_input_file += f'neigh_modify {join_keywords(pdict["neigh_modify"])}\n'
        # Define Timestep
        lammps_input_file += f'timestep        {pdict["timestep"]}\n'

        # Define computation/printing of thermodynamic info
        lammps_input_file += '\n# Thermodynamic Information Output\n'
        thermo_keywords = ['step', 'temp', 'epair', 'emol', 'etotal', 'press']
        for kwd in pdict.get('thermo_keywords', []):
            if kwd not in thermo_keywords:
                thermo_keywords.append(kwd)
        lammps_input_file += f'thermo_style custom {" ".join(thermo_keywords)}\n'
        lammps_input_file += 'thermo          1000\n'  # TODO make variable?

        # Setup initial velocities of atoms
        if 'velocity' in pdict:
            lammps_input_file += '\n# Intial Atom Velocity\n'
        for vdict in pdict.get('velocity', []):
            lammps_input_file += f'velocity all {vdict["style"]} '
            lammps_input_file += f'{" ".join([str(a) for a in vdict["args"]])} '
            lammps_input_file += f'{join_keywords(vdict.get("keywords", {}))}\n'

        stage_names = []
        current_fixes = []
        current_dumps = []
        current_computes = []

        for stage_id, stage_dict in enumerate(pdict.get('stages')):

            stage_name = stage_dict.get('name')
            if stage_name in stage_names:
                raise ValueError(f'non-unique stage name: {stage_name}')
            stage_names.append(stage_name)

            lammps_input_file += f'\n# Stage {stage_id}: {stage_name}\n'

            # clear timestep
            # lammps_input_file += "reset_timestep  0\n"

            # Clear fixes, dumps and computes
            for fix in current_fixes:
                lammps_input_file += f'unfix {fix}\n'
            current_fixes = []
            for dump in current_dumps:
                lammps_input_file += f'undump {dump}\n'
            current_dumps = []
            for compute in current_computes:
                lammps_input_file += f'uncompute {compute}\n'
            current_computes = []

            # Define Computes
            for compute in stage_dict.get('computes', []):
                c_id = compute['id']
                c_style = compute['style']
                c_args = ' '.join([str(a) for a in compute.get('args', [])])
                lammps_input_file += f'compute         {c_id} all {c_style} {c_args}\n'
                current_computes.append(c_id)

            # Define Atom Level Outputs
            output_atom_dict = stage_dict.get('output_atom', {})
            if output_atom_dict.get('dump_rate', 0):
                atom_dump_cmnds, acomputes, afixes = atom_info_commands(
                    variables=output_atom_dict.get('variables', []),
                    ave_variables=output_atom_dict.get('ave_variables', []),
                    kind_symbols=kind_symbols,
                    atom_style=potential_data.atom_style,
                    dump_rate=output_atom_dict.get('dump_rate', 0),
                    average_rate=output_atom_dict.get('average_rate', 1),
                    filename='{}-{}'.format(stage_name, trajectory_filename),
                    version_date=version_date,
                    dump_name='atom_info',
                )
                if atom_dump_cmnds:
                    lammps_input_file += '\n'.join(atom_dump_cmnds) + '\n'
                    current_dumps.append('atom_info')
                current_computes.extend(acomputes)
                current_fixes.extend(afixes)

            # Define System Level Outputs
            output_sys_dict = stage_dict.get('output_system', {})
            if output_sys_dict.get('dump_rate', 0):
                sys_info_cmnds = sys_ave_commands(
                    variables=output_sys_dict.get('variables', []),
                    ave_variables=output_sys_dict.get('ave_variables', []),
                    dump_rate=output_sys_dict.get('dump_rate', 0),
                    filename='{}-{}'.format(stage_name, system_filename),
                    fix_name='sys_info',
                    average_rate=output_sys_dict.get('average_rate', 1),
                )
                if sys_info_cmnds:
                    lammps_input_file += '\n'.join(sys_info_cmnds) + '\n'
                    current_fixes.append('sys_info')

            # Define restart
            if stage_dict.get('restart_rate', 0):
                lammps_input_file += 'restart         '
                lammps_input_file += f'{stage_dict.get("restart_rate", 0)} '
                lammps_input_file += f'{"{}-{}".format(stage_name, restart_filename)}\n'
            else:
                lammps_input_file += 'restart         0\n'

            # Define time integration method
            lammps_input_file += 'fix             int all '
            lammps_input_file += f'{get_path(stage_dict, ["integration", "style"])} '
            _temp = join_keywords(
                get_path(
                    stage_dict,
                    ['integration', 'constraints'],
                    {},
                    raise_error=False,
                ))
            lammps_input_file += f'{_temp} '
            _temp = join_keywords(
                get_path(stage_dict, ['integration', 'keywords'], {},
                         raise_error=False))
            lammps_input_file += f'{_temp}\n'
            current_fixes.append('int')

            # Run
            lammps_input_file += f'run             {stage_dict.get("steps", 0)}\n'

            # check compute/fix/dump ids are unique
            if len(current_computes) != len(set(current_computes)):
                raise ValueError(
                    f'Stage {stage_name}: Non-unique compute ids; {current_computes}'
                )
            if len(current_fixes) != len(set(current_fixes)):
                raise ValueError(
                    f'Stage {stage_name}: Non-unique fix ids; {current_fixes}')
            if len(current_dumps) != len(set(current_dumps)):
                raise ValueError(
                    f'Stage {stage_name}: Non-unique dump ids; {current_dumps}'
                )

        lammps_input_file += '\n# Final Commands\n'
        # output final energy
        lammps_input_file += 'variable final_energy equal etotal\n'
        lammps_input_file += 'print "final_energy: ${final_energy}"\n'

        lammps_input_file += 'print "END_OF_COMP"\n'

        return lammps_input_file

    @staticmethod
    def validate_parameters(param_data, potential_object):
        if param_data is None:
            raise InputValidationError('parameter data not set')
        validate_against_schema(param_data.get_dict(), 'md-multi.schema.json')

        # ensure the potential and parameters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        punits = param_data.get_dict()['units']
        if not punits == potential_object.default_units:
            raise InputValidationError(
                f'the units of the parameters ({punits}) and potential '
                f'({potential_object.default_units}) are different')

        return True

    def get_retrieve_lists(self):
        return (
            [],
            [
                '*-' + self.options.trajectory_suffix,
                '*-' + self.options.system_suffix,
                '*-' + self.options.restart_filename + '.*',
            ],
        )


def sys_print_commands(
    variables,
    dump_rate,
    filename,
    fix_name: str = 'sys_info',
    append: bool = True,
    print_header: bool = True,
):
    """Create commands to output required system variables to a file."""
    # pylint: disable=too-many-arguments
    commands = []

    if not variables:
        return commands

    if 'step' not in variables:
        # always include 'step', so we can sync with the `dump` data
        variables.insert(0, 'step')

    var_aliases = []
    for var in variables:
        var_alias = var.replace('[', '_').replace(']', '_')
        var_aliases.append(var_alias)
        commands.append(f'variable {var_alias} equal {var}')

    commands.append('fix {0} all print {1} "{2}" {3} {4} {5} screen no'.format(  # pylint: disable=consider-using-f-string
        fix_name,
        dump_rate,
        ' '.join(['${{{0}}}'.format(v) for v in var_aliases]),
        'title "{}"'.format(' '.join(var_aliases)) if print_header else '',
        'append' if append else 'file',
        filename,
    ))

    return commands


def sys_ave_commands(
    variables,
    ave_variables,
    dump_rate,
    filename,
    fix_name: str = 'sys_info',
    average_rate: bool = None,
):
    """Create commands to output required system variables to a file."""
    # pylint: disable=too-many-arguments
    commands = []

    if not (variables or ave_variables):
        return commands

    if set(variables).intersection(ave_variables):
        raise ValueError(
            'variables cannot be in both "variables" and "ave_variables": '
            f'{set(variables).intersection(ave_variables)}')

    # Note step is included, by default, as the first arg
    var_aliases = []
    for var in variables + ave_variables:
        var_alias = var.replace('[', '_').replace(']', '_')
        var_aliases.append(var_alias)
        commands.append(f'variable {var_alias} equal {var}')

    if not ave_variables:
        nevery = dump_rate
        nrep = 1
    else:
        if dump_rate % average_rate != 0 or average_rate > dump_rate:
            raise ValueError(
                f'The dump rate ({dump_rate}) must be a multiple of the '
                f'average_rate ({average_rate})')
        nevery = average_rate
        nrep = int(dump_rate / average_rate)

    commands.append("""fix {fid} all ave/time {nevery} {nrepeat} {nfreq} &
    {variables} &
    {non_ave} &
    title1 "step {header}" &
    file {filename}""".format(  # pylint: disable=consider-using-f-string
        fid=fix_name,
        nevery=nevery,  # compute variables every n steps
        nfreq=
        dump_rate,  # nfreq is the dump rate and must be a multiple of nevery
        nrepeat=
        nrep,  # average is over nrepeat quantities, nrepeat*nevery <= nfreq
        variables=' '.join(['v_{0}'.format(v) for v in var_aliases]),
        non_ave=' '.join(
            ['off {0}'.format(i + 1) for i in range(len(variables))]),
        header=' '.join(var_aliases),
        filename=filename,
    ))

    return commands


def atom_info_commands(
    variables,
    ave_variables,
    kind_symbols,
    atom_style,
    dump_rate,
    average_rate,
    filename,
    version_date,
    dump_name: str = 'atom_info',
    append: bool = True,
):
    """Create commands to output required atom variables to a file.

    Parameters
    ----------
    variables : list[str]
    kind_symbols : list[str]
        atom symbols per type
    atom_style : str
        style of atoms e.g. charge
    dump_rate : int
    filename : str
    version_date : timedate
    dump_name : str
    append : bool
        Dump snapshots to the end of the dump file (if it exists).

    Returns
    -------
    list[str]

    """
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, unused-argument
    commands, computes, fixes = [], [], []

    if atom_style == 'charge':
        dump_variables = 'element x y z q'.split()
    else:
        dump_variables = 'element x y z'.split()

    for variable in variables:
        if variable not in dump_variables:
            dump_variables.append(variable)

    if ave_variables:
        if dump_rate % average_rate != 0 or average_rate > dump_rate:
            raise ValueError(
                f'The dump rate ({dump_rate}) must be a multiple of '
                f'the average_rate ({average_rate})')
        nevery = average_rate
        nrep = int(dump_rate / average_rate)

        # work out which variables need to be computed
        avar_props = [
            v for v in ave_variables
            if not any([v.startswith(s) for s in ['c_', 'f_', 'v_']])  # pylint: disable=use-a-generator
        ]
        avar_names = []
        c_at_vars = 1
        for ave_var in ave_variables:
            if any([ave_var.startswith(s) for s in ['c_', 'f_', 'v_']]):  # pylint: disable=use-a-generator
                avar_names.append(ave_var)
            else:
                if len(avar_props) > 1:
                    avar_names.append(f'c_at_vars[{c_at_vars}]')
                    c_at_vars += 1
                else:
                    avar_names.append('c_at_vars')

        # compute required variables
        if avar_props:
            commands.append(
                f'compute at_vars all property/atom {" ".join(avar_props)}')
            computes.append('at_vars')

        # compute means for variables
        commands.append(
            'fix at_means all ave/atom {nevery} {nrepeat} {nfreq} {variables}'.
            format(  # pylint: disable=consider-using-f-string
                nevery=nevery,  # compute variables every n steps
                nfreq=
                dump_rate,  # nfreq is the dump rate and must be a multiple of nevery
                nrepeat=
                nrep,  # average is over nrepeat quantities, nrepeat*nevery <= nfreq
                variables=' '.join(avar_names),
            ))
        fixes.append('at_means')

        # set the averages as variables, just so the dump names are decipherable
        for i, ave_var in enumerate(ave_variables):
            commands.append('variable ave_{0} atom f_at_means{1}'.format(
                ave_var,
                '[{}]'.format(i + 1) if len(ave_variables) > 1 else ''))

    commands.append(
        'dump     {dump_id} all custom {rate} {fname} {variables} {ave_vars}'.
        format(  # pylint: disable=consider-using-f-string
            dump_id=dump_name,
            rate=dump_rate,
            fname=filename,
            variables=' '.join(dump_variables),
            ave_vars=' '.join([f'v_ave_{v}' for v in ave_variables]),
        ))
    if append:
        commands.append(f'dump_modify     {dump_name} append yes')

    commands.extend([
        f'dump_modify     {dump_name} sort id',
        f'dump_modify     {dump_name} element {" ".join(kind_symbols)}',
    ])
    return commands, computes, fixes
