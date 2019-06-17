from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string
import six


def generate_LAMMPS_input(calc,
                          parameters_data,
                          potential_obj,
                          structure_file='data.gan',
                          trajectory_file='trajectory.lammpstr',
                          version_date='11 Aug 2017'):

    names_str = ' '.join(potential_obj.kind_names)

    lammps_input_file =  'units          {0}\n'.format(potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(potential_obj.atom_style)

    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'
    lammps_input_file += 'dump            aiida all custom 1 {0} element fx fy fz\n'.format(trajectory_file)

    # TODO find exact version when changes were made
    if version_date <= convert_date_string('10 Feb 2015'):
        lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f"\n'
    else:
        lammps_input_file += 'dump_modify     aiida format line "%4s  %16.10f %16.10f %16.10f"\n'

    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)

    lammps_input_file += 'run             0'

    return lammps_input_file


class ForceCalculation(BaseLammpsCalculation):

    _OUTPUT_TRAJECTORY_FILE_NAME = 'trajectory.lammpstrj'
    _generate_input_function = generate_LAMMPS_input

    @classmethod
    def define(cls, spec):
        super(ForceCalculation, cls).define(spec)

        spec.input('metadata.options.trajectory_name', valid_type=six.string_types, default=cls._OUTPUT_TRAJECTORY_FILE_NAME)
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='lammps.force')

        # spec.input('settings', valid_type=six.string_types, default='lammps.optimize')

    def validate_parameters(self, param_data, potential_object):
        self._retrieve_list += [self._OUTPUT_TRAJECTORY_FILE_NAME]
        self._retrieve_temporary_list += []
