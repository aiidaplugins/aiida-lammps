from aiida.common.utils import classproperty
from aiida.orm.calculation.job import JobCalculation
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string


def generate_LAMMPS_input(parameters_data,
                          potential_obj,
                          structure_file='data.gan',
                          trajectory_file='path.lammpstrj'):

    names_str = ' '.join(potential_obj._names)

    parameters = parameters_data.get_dict()

    lammps_date = convert_date_string(parameters.get("lammps_version", None))

    lammps_input_file = 'units           {0}\n'.format(potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      atomic\n'
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'fix             int all box/relax {} {} vmax {}\n'.format(parameters['relaxation'],
                                                                                    parameters['pressure'] * 1000,  # pressure kb -> bar
                                                                                    parameters['vmax'])

    # TODO find exact version when changes were made
    if lammps_date <= convert_date_string('11 Nov 2013'):
        lammps_input_file += 'compute         stpa all stress/atom\n'
    else:
        lammps_input_file += 'compute         stpa all stress/atom NULL\n'

                                                              #  xx,       yy,        zz,       xy,       xz,       yz
    lammps_input_file += 'compute         stgb all reduce sum c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]\n'
    lammps_input_file += 'variable        pr equal -(c_stgb[1]+c_stgb[2]+c_stgb[3])/(3*vol)\n'
    lammps_input_file += 'thermo_style    custom step temp press v_pr etotal c_stgb[1] c_stgb[2] c_stgb[3] c_stgb[4] c_stgb[5] c_stgb[6]\n'

    lammps_input_file += 'dump            aiida all custom 1 {0} element x y z  fx fy fz\n'.format(trajectory_file)

    # TODO find exact version when changes were made
    if lammps_date <= convert_date_string('10 Feb 2015'):
        lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"\n'
    else:
        lammps_input_file += 'dump_modify     aiida format line "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"\n'

    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)
    lammps_input_file += 'min_style       cg\n'
    lammps_input_file += 'minimize        {} {} {} {}\n'.format(parameters['energy_tolerance'],
                                                                           parameters['force_tolerance'],
                                                                           parameters['max_iterations'],
                                                                           parameters['max_evaluations'])
  #  lammps_input_file += 'print           "$(xlo - xhi) $(xy) $(xz)"\n'
  #  lammps_input_file += 'print           "0.000 $(yhi - ylo) $(yz)"\n'
  #  lammps_input_file += 'print           "0.000 0.000   $(zhi-zlo)"\n'
    lammps_input_file += 'print           "$(xlo) $(xhi) $(xy)"\n'
    lammps_input_file += 'print           "$(ylo) $(yhi) $(xz)"\n'
    lammps_input_file += 'print           "$(zlo) $(zhi) $(yz)"\n'

    return lammps_input_file


class OptimizeCalculation(BaseLammpsCalculation, JobCalculation):

    _OUTPUT_TRAJECTORY_FILE_NAME = 'path.lammpstrj'
    _OUTPUT_FILE_NAME = 'log.lammps'

    def _init_internal_params(self):
        super(OptimizeCalculation, self)._init_internal_params()

        self._default_parser = 'lammps.optimize'

        self._retrieve_list = [self._OUTPUT_TRAJECTORY_FILE_NAME, self._OUTPUT_FILE_NAME]
        self._retrieve_temporary_list = [self._INPUT_UNITS]
        self._generate_input_function = generate_LAMMPS_input

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods
        retdict.update(BaseLammpsCalculation._baseclass_use_methods)

        return retdict

#$MPI -n $NSLOTS $LAMMPS -sf gpu -pk gpu 2 neigh no -in in.md_data
