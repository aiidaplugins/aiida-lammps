"""Single stage MD calculation in LAMMPS."""
# pylint: disable=fixme, useless-super-delegation
from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory
import numpy as np

from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, get_path, join_keywords
from aiida_lammps.validation import validate_against_schema


class MdCalculation(BaseLammpsCalculation):
    """Calculation of a single MD stage in LAMMPS."""
    @classmethod
    def define(cls, spec):
        super(MdCalculation, cls).define(spec)

        spec.input(
            'metadata.options.parser_name',
            valid_type=str,
            default='lammps.md',
        )
        spec.default_output_port = 'results'

        spec.output(
            'trajectory_data',
            valid_type=DataFactory('lammps.trajectory'),
            required=True,
            help='atomic configuration data per dump step',
        )
        spec.output(
            'system_data',
            valid_type=DataFactory('array'),
            required=False,
            help='selected system data per dump step',
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
        # pylint: disable=too-many-locals, too-many-arguments¸ too-many-branches, too-many-statements
        pdict = parameter_data.get_dict()
        version_date = convert_date_string(
            pdict.get('lammps_version', '11 Aug 2017'))

        # Geometry Setup
        lammps_input_file = f'units           {potential_data.default_units}\n'
        lammps_input_file += 'boundary        p p p\n'
        lammps_input_file += 'box tilt large\n'
        lammps_input_file += 'atom_style      {0}\n'.format(
            potential_data.atom_style)
        lammps_input_file += 'read_data       {}\n'.format(structure_filename)

        # Potential Specification
        lammps_input_file += potential_data.get_input_lines(kind_symbols)

        # Pairwise neighbour list creation
        if 'neighbor' in pdict:
            # neighbor skin_dist bin/nsq/multi
            lammps_input_file += 'neighbor {0} {1}\n'.format(
                pdict['neighbor'][0], pdict['neighbor'][1])
        if 'neigh_modify' in pdict:
            # e.g. 'neigh_modify every 1 delay 0 check no\n'
            lammps_input_file += 'neigh_modify {}\n'.format(
                join_keywords(pdict['neigh_modify']))

        # Define Timestep
        lammps_input_file += 'timestep        {}\n'.format(pdict['timestep'])

        # Define computation/printing of thermodynamic info
        thermo_keywords = ['step', 'temp', 'epair', 'emol', 'etotal', 'press']
        for kwd in pdict.get('thermo_keywords', []):
            if kwd not in thermo_keywords:
                thermo_keywords.append(kwd)
        lammps_input_file += 'thermo_style custom {}\n'.format(
            ' '.join(thermo_keywords))
        lammps_input_file += 'thermo          1000\n'

        # Define output of restart file
        restart = pdict.get('restart', False)
        if restart:
            lammps_input_file += 'restart        {0} {1}\n'.format(
                restart, restart_filename)

        # Set the initial velocities of atoms, if a temperature is set
        initial_temp, _, _ = get_path(
            pdict,
            ['integration', 'constraints', 'temp'],
            default=[None, None, None],
            raise_error=False,
        )
        if initial_temp is not None:
            lammps_input_file += (
                'velocity        all create {0} {1} dist gaussian mom yes\n'.
                format(initial_temp,
                       pdict.get('rand_seed', np.random.randint(10000000))))
            lammps_input_file += 'velocity        all scale {}\n'.format(
                initial_temp)

        # Define Equilibration Stage
        lammps_input_file += 'fix             int all {0} {1} {2}\n'.format(
            get_path(pdict, ['integration', 'style']),
            join_keywords(
                get_path(pdict, ['integration', 'constraints'], {},
                         raise_error=False)),
            join_keywords(
                get_path(pdict, ['integration', 'keywords'], {},
                         raise_error=False)),
        )

        lammps_input_file += 'run             {}\n'.format(
            parameter_data.dict.equilibrium_steps)
        lammps_input_file += 'reset_timestep  0\n'

        if potential_data.atom_style == 'charge':
            dump_variables = 'element x y z q'
            dump_format = '%4s  %16.10f %16.10f %16.10f %16.10f'
        else:
            dump_variables = 'element x y z'
            dump_format = '%4s  %16.10f %16.10f %16.10f'

        lammps_input_file += 'dump            aiida all custom {0} {1} {2}\n'.format(
            parameter_data.dict.dump_rate, trajectory_filename, dump_variables)

        # TODO find exact version when changes were made
        if version_date <= convert_date_string('10 Feb 2015'):
            dump_mod_cmnd = 'format'
        else:
            dump_mod_cmnd = 'format line'

        lammps_input_file += 'dump_modify     aiida {0} "{1}"\n'.format(
            dump_mod_cmnd, dump_format)
        lammps_input_file += 'dump_modify     aiida sort id\n'
        lammps_input_file += 'dump_modify     aiida element {}\n'.format(
            ' '.join(kind_symbols))

        variables = pdict.get('output_variables', [])
        if variables and 'step' not in variables:
            # always include 'step', so we can sync with the `dump` data
            # NOTE `dump` includes step 0, whereas `print` starts from step 1
            variables.append('step')
        var_aliases = []
        for var in variables:
            var_alias = var.replace('[', '_').replace(']', '_')
            var_aliases.append(var_alias)
            lammps_input_file += 'variable {0} equal {1}\n'.format(
                var_alias, var)
        if variables:
            lammps_input_file += 'fix sys_info all print'
            lammps_input_file += f' {parameter_data.dict.dump_rate}'
            lammps_input_file += f' "{" ".join(["${{{0}}}".format(v) for v in var_aliases])}"'
            lammps_input_file += f' title "{" ".join(var_aliases)}"'
            lammps_input_file += f' file {system_filename} screen no\n'

        lammps_input_file += 'run             {}\n'.format(
            parameter_data.dict.total_steps)

        lammps_input_file += 'variable final_energy equal etotal\n'
        lammps_input_file += 'print "final_energy: ${final_energy}"\n'

        lammps_input_file += 'print "END_OF_COMP"\n'

        return lammps_input_file

    @staticmethod
    def validate_parameters(param_data, potential_object):
        if param_data is None:
            raise InputValidationError('parameter data not set')
        validate_against_schema(param_data.get_dict(), 'md.schema.json')

        # ensure the potential and paramters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        punits = param_data.get_dict()['units']
        if not punits == potential_object.default_units:
            raise InputValidationError(
                'the units of the parameters ({}) and potential ({}) are different'
                .format(punits, potential_object.default_units))

        return True

    def get_retrieve_lists(self):
        return [], [self.options.trajectory_suffix, self.options.system_suffix]
