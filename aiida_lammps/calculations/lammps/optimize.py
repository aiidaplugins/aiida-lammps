from aiida.common.exceptions import InputValidationError
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords
from aiida_lammps.validation import validate_with_json
from aiida.plugins import DataFactory
import six


def generate_lammps_input(calc,
                          parameters,
                          potential_obj,
                          structure_filename,
                          trajectory_filename,
                          version_date, **kwargs):

    names_str = ' '.join(potential_obj.kind_names)

    parameters = parameters.get_dict()

    # lammps_date = convert_date_string(parameters.get("lammps_version", None))

    lammps_input_file = 'units          {0}\n'.format(
        potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(
        potential_obj.atom_style)
    lammps_input_file += 'read_data       {}\n'.format(structure_filename)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'fix             int all box/relax {} {} {}\n'.format(parameters['relax']['type'],
                                                                               parameters['relax']['pressure'],
                                                                               join_keywords(parameters['relax'],
                                                                                             ignore=['type', 'pressure']))

    # TODO find exact version when changes were made
    if version_date <= convert_date_string('11 Nov 2013'):
        lammps_input_file += 'compute         stpa all stress/atom\n'
    else:
        lammps_input_file += 'compute         stpa all stress/atom NULL\n'

        #  xx,       yy,        zz,       xy,       xz,       yz
    lammps_input_file += 'compute         stgb all reduce sum c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]\n'
    lammps_input_file += 'variable        pr equal -(c_stgb[1]+c_stgb[2]+c_stgb[3])/(3*vol)\n'
    lammps_input_file += 'thermo_style    custom step temp press v_pr etotal c_stgb[1] c_stgb[2] c_stgb[3] c_stgb[4] c_stgb[5] c_stgb[6]\n'

    lammps_input_file += 'dump            aiida all custom 1 {0} element x y z  fx fy fz\n'.format(trajectory_filename)

    # TODO find exact version when changes were made
    if version_date <= convert_date_string('10 Feb 2015'):
        lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"\n'
    else:
        lammps_input_file += 'dump_modify     aiida format line "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"\n'

    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)
    lammps_input_file += 'min_style       {}\n'.format(
        parameters['minimize']['style'])
    # lammps_input_file += 'min_style       cg\n'
    lammps_input_file += 'minimize        {} {} {} {}\n'.format(parameters['minimize']['energy_tolerance'],
                                                                parameters['minimize']['force_tolerance'],
                                                                parameters['minimize']['max_iterations'],
                                                                parameters['minimize']['max_evaluations'])
    #  lammps_input_file += 'print           "$(xlo - xhi) $(xy) $(xz)"\n'
    #  lammps_input_file += 'print           "0.000 $(yhi - ylo) $(yz)"\n'
    #  lammps_input_file += 'print           "0.000 0.000   $(zhi-zlo)"\n'
    lammps_input_file += 'print           "$(xlo) $(xhi) $(xy)"\n'
    lammps_input_file += 'print           "$(ylo) $(yhi) $(xz)"\n'
    lammps_input_file += 'print           "$(zlo) $(zhi) $(yz)"\n'

    variables = parameters.get("output_variables", [])
    for var in variables:
        if var.startswith("reax_"):
            continue  # these variables are set in the potential lines
        lammps_input_file += 'variable {0} equal {0}\n'.format(var)
        lammps_input_file += 'print "final_variable: {0} = ${{{0}}}"\n'.format(var)

    return lammps_input_file


class OptimizeCalculation(BaseLammpsCalculation):

    _generate_input_function = generate_lammps_input

    @classmethod
    def define(cls, spec):
        super(OptimizeCalculation, cls).define(spec)

        spec.input('metadata.options.parser_name',
                   valid_type=six.string_types, default='lammps.optimize')

        spec.output('structure',
                    valid_type=DataFactory('structure'),
                    required=True,
                    help='the structure output from the calculation')
        spec.output('arrays',
                    valid_type=DataFactory('array'),
                    required=True,
                    help='forces, stresses and positions data per step')

    def validate_parameters(self, param_data, potential_object):
        if param_data is None:
            raise InputValidationError("parameter data not set")
        validate_with_json(param_data.get_dict(), "optimize")

        # ensure the potential and paramters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        if 'units' in param_data.get_dict():
            punits = param_data.get_dict()['units']
            if not punits == potential_object.default_units:
                raise InputValidationError('the units of the parameters ({}) and potential ({}) are different'.format(
                    punits, potential_object.default_units
                ))
        else:
            self.logger.log('No units defined, using:',
                            potential_object.default_units)

        # Update retrieve list
        if self.options.trajectory_name not in self._retrieve_list:
            self._retrieve_list += [self.options.trajectory_name]
        self._retrieve_temporary_list += []

        return True
