from aiida.orm.calculation.job import JobCalculation
from aiida.common.exceptions import InputValidationError
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty
from aiida.orm import DataFactory
from aiida_lammps.calculations.lammps import BaseLammpsCalculation

from aiida_lammps.calculations.lammps.potentials import LammpsPotential
from aiida_lammps.common.utils import convert_date_string

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')


def generate_LAMMPS_input(parameters_data,
                          potential_obj,
                          structure_file='data.gan',
                          trajectory_file='trajectory.lammpstr'):

    lammps_date = convert_date_string(parameters_data.get_dict().get("lammps_version", None))

    names_str = ' '.join(potential_obj._names)

    lammps_input_file =  'units           metal\n'
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      atomic\n'

    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'
    lammps_input_file += 'dump            aiida all custom 1 {0} element fx fy fz\n'.format(trajectory_file)

    # TODO find exact version when changes were made
    if lammps_date <= convert_date_string('10 Feb 2015'):
        lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f"\n'
    else:
        lammps_input_file += 'dump_modify     aiida format line "%4s  %16.10f %16.10f %16.10f"\n'

    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)

    lammps_input_file += 'run             0'

    return lammps_input_file


class ForceCalculation(BaseLammpsCalculation, JobCalculation):

    _OUTPUT_TRAJECTORY_FILE_NAME = 'trajectory.lammpstrj'
    _OUTPUT_FILE_NAME = 'log.lammps'

    def _init_internal_params(self):
        super(ForceCalculation, self)._init_internal_params()

        self._default_parser = 'lammps.force'
        self.__retrieve_list = []
        self._generate_input_function = generate_LAMMPS_input
        self._retrieve_list = [self._OUTPUT_TRAJECTORY_FILE_NAME, self._OUTPUT_FILE_NAME]

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods
        retdict.update(BaseLammpsCalculation._baseclass_use_methods)

        return retdict
