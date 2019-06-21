from aiida.plugins import DataFactory
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.validation import validate_with_json
from aiida_lammps.common.utils import convert_date_string
import six


def generate_lammps_input(calc,
                          parameters,
                          potential_obj,
                          structure_filename,
                          trajectory_filename,
                          add_thermo_keywords,
                          version_date='11 Aug 2017', **kwargs):

    lammps_input_file = 'units          {0}\n'.format(potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(potential_obj.atom_style)

    lammps_input_file += 'read_data       {}\n'.format(structure_filename)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    thermo_keywords = ["step", "temp", "epair", "emol", "etotal", "press"]
    for kwd in add_thermo_keywords:
        if kwd not in thermo_keywords:
            thermo_keywords.append(kwd)
    lammps_input_file += 'thermo_style custom {}\n'.format(" ".join(thermo_keywords))

    if potential_obj.atom_style == 'charge':
        dump_variables = "element x y z fx fy fz q"
        dump_format = "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f %16.10f"
    else:
        dump_variables = "element x y z fx fy fz"
        dump_format = "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"

    lammps_input_file += 'dump            aiida all custom 1 {0} {1}\n'.format(trajectory_filename, dump_variables)

    # TODO find exact version when changes were made
    if version_date <= convert_date_string('10 Feb 2015'):
        dump_mod_cmnd = "format"
    else:
        dump_mod_cmnd = "format line"

    lammps_input_file += 'dump_modify     aiida {0} "{1}"\n'.format(dump_mod_cmnd, dump_format)

    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(' '.join(potential_obj.kind_elements))

    lammps_input_file += 'run             0\n'

    variables = parameters.get_attribute("output_variables", [])
    for var in variables:
        var_alias = var.replace("[", "_").replace("]", "_")
        lammps_input_file += 'variable {0} equal {1}\n'.format(var_alias, var)
        lammps_input_file += 'print "final_variable: {0} = ${{{0}}}"\n'.format(var_alias)

    lammps_input_file += 'variable final_energy equal etotal\n'
    lammps_input_file += 'print "final_energy: ${final_energy}"\n'

    return lammps_input_file


class ForceCalculation(BaseLammpsCalculation):

    _generate_input_function = generate_lammps_input

    @classmethod
    def define(cls, spec):
        super(ForceCalculation, cls).define(spec)

        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='lammps.force')

        # spec.input('settings', valid_type=six.string_types, default='lammps.optimize')

        spec.output('arrays',
                    valid_type=DataFactory('array'),
                    required=True,
                    help='force data per atom')

    def validate_parameters(self, param_data, potential_object):
        if param_data is not None:
            validate_with_json(param_data.get_dict(), "force")
        if self.options.trajectory_name not in self._retrieve_list:
            self._retrieve_list += [self.options.trajectory_name]
        self._retrieve_temporary_list += []
