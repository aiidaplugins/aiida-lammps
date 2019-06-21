import numpy as np
from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords, get_path
from aiida_lammps.validation import validate_with_json
import six


def generate_lammps_input(calc,
                          parameters,
                          potential_obj,
                          structure_filename,
                          trajectory_filename,
                          restart_filename,
                          info_filename,
                          add_thermo_keywords,
                          version_date='11 Aug 2017', **kwargs):

    pdict = parameters.get_dict()

    random_number = np.random.randint(10000000)

    # lammps_date = convert_date_string(pdict.get("lammps_version", None))

    lammps_input_file = 'units           {0}\n'.format(
        potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(
        potential_obj.atom_style)
    lammps_input_file += 'read_data       {}\n'.format(structure_filename)

    lammps_input_file += potential_obj.get_input_potential_lines()

    if "neighbor" in pdict:
        lammps_input_file += "neighbor {0} {1}\n".format(pdict["neighbor"][0],
                                                         pdict["neighbor"][1])
    if "neigh_modify" in pdict:
        lammps_input_file += "neigh_modify {}\n".format(
            join_keywords(pdict["neigh_modify"]))

    # lammps_input_file += 'neighbor        0.3 bin\n'
    # lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    lammps_input_file += 'timestep        {}\n'.format(pdict["timestep"])

    thermo_keywords = ["step", "temp", "epair", "emol", "etotal", "press"]
    for kwd in add_thermo_keywords:
        if kwd not in thermo_keywords:
            thermo_keywords.append(kwd)
    lammps_input_file += 'thermo_style custom {}\n'.format(" ".join(thermo_keywords))
    lammps_input_file += 'thermo          1000\n'

    restart = pdict.get("restart", False)
    if restart:
        lammps_input_file += 'restart        {0} {1}\n'.format(restart, restart_filename)

    initial_temp, _, _ = get_path(pdict, ["integration", "constraints", "temp"],
                                  default=[None, None, None], raise_error=False)

    if initial_temp is not None:
        lammps_input_file += 'velocity        all create {0} {1} dist gaussian mom yes\n'.format(
            initial_temp, random_number)
        lammps_input_file += 'velocity        all scale {}\n'.format(
            initial_temp)

    lammps_input_file += 'fix             int all {0} {1} {2}\n'.format(
        get_path(pdict, ["integration", "style"]),
        join_keywords(
            get_path(pdict, ["integration", "constraints"], {}, raise_error=False)),
        join_keywords(get_path(pdict, ["integration", "keywords"], {}, raise_error=False)))

    # lammps_input_file += 'fix             int all nvt temp {0} {0} {1}\n'.format(parameters.dict.temperature,
    #                                                                              parameters.dict.thermostat_variable)

    lammps_input_file += 'run             {}\n'.format(
        parameters.dict.equilibrium_steps)
    lammps_input_file += 'reset_timestep  0\n'

    if potential_obj.atom_style == 'charge':
        dump_variables = "element x y z q"
        dump_format = "%4s  %16.10f %16.10f %16.10f %16.10f"
    else:
        dump_variables = "element x y z"
        dump_format = "%4s  %16.10f %16.10f %16.10f"

    lammps_input_file += 'dump            aiida all custom {0} {1} {2}\n'.format(
        parameters.dict.dump_rate, trajectory_filename, dump_variables)

    # TODO find exact version when changes were made
    if version_date <= convert_date_string('10 Feb 2015'):
        dump_mod_cmnd = "format"
    else:
        dump_mod_cmnd = "format line"

    lammps_input_file += 'dump_modify     aiida {0} "{1}"\n'.format(dump_mod_cmnd, dump_format)
    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(' '.join(potential_obj.kind_elements))

    variables = pdict.get("output_variables", [])
    if variables and 'step' not in variables:
        # always include 'step', so we can sync with the `dump` data
        # NOTE `dump` includes step 0, whereas `print` starts from step 1
        variables.append('step')
    var_aliases = []
    for var in variables:
        var_alias = var.replace("[", "_").replace("]", "_")
        var_aliases.append(var_alias)
        lammps_input_file += 'variable {0} equal {1}\n'.format(var_alias, var)
    if variables:
        lammps_input_file += 'fix sys_info all print {0} "{1}" title "{2}" file {3} screen no\n'.format(
            parameters.dict.dump_rate,
            " ".join(["${{{0}}}".format(v) for v in var_aliases]),
            " ".join(var_aliases), info_filename)

    lammps_input_file += 'run             {}\n'.format(
        parameters.dict.total_steps)

    lammps_input_file += 'variable final_energy equal etotal\n'
    lammps_input_file += 'print "final_energy: ${final_energy}"\n'

    return lammps_input_file


class MdCalculation(BaseLammpsCalculation):

    _generate_input_function = generate_lammps_input

    @classmethod
    def define(cls, spec):
        super(MdCalculation, cls).define(spec)

        spec.input('metadata.options.parser_name',
                   valid_type=six.string_types, default='lammps.md')
        spec.default_output_port = 'results'

        spec.output('trajectory_data',
                    valid_type=DataFactory('array.trajectory'),
                    required=True,
                    help='atomic configuration data per dump step')
        spec.output('system_data',
                    valid_type=DataFactory('array'),
                    required=False,
                    help='selected system data per dump step')

    def validate_parameters(self, param_data, potential_object):
        if param_data is None:
            raise InputValidationError("parameter data not set")
        validate_with_json(param_data.get_dict(), "md")

        # ensure the potential and paramters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        punits = param_data.get_dict()['units']
        if not punits == potential_object.default_units:
            raise InputValidationError('the units of the parameters ({}) and potential ({}) are different'.format(
                punits, potential_object.default_units
            ))

        self._retrieve_list += []
        if self.options.trajectory_name not in self._retrieve_temporary_list:
            self._retrieve_temporary_list += [self.options.trajectory_name]
        if self.options.info_filename not in self._retrieve_temporary_list:
            self._retrieve_temporary_list += [self.options.info_filename]

        return True
