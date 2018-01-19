from aiida.orm.calculation.job import JobCalculation
from aiida.common.utils import classproperty
from aiida_lammps.calculations.lammps import BaseLammpsCalculation

import numpy as np


def generate_LAMMPS_input(parameters,
                          potential_obj,
                          structure_file='potential.pot',
                          trajectory_file='trajectory.lammpstr'):

    random_number = np.random.randint(10000000)

    names_str = ' '.join(potential_obj._names)

    lammps_input_file = 'units           metal\n'
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      atomic\n'
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    lammps_input_file += 'neighbor        0.3 bin\n'
    lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    lammps_input_file += 'timestep        {}\n'.format(parameters.dict.timestep)

    lammps_input_file += 'thermo_style    custom step etotal temp vol press\n'
    lammps_input_file += 'thermo          1000\n'

    lammps_input_file += 'velocity        all create {0} {1} dist gaussian mom yes\n'.format(parameters.dict.temperature, random_number)
    lammps_input_file += 'velocity        all scale {}\n'.format(parameters.dict.temperature)

    lammps_input_file += 'fix             int all nvt temp {0} {0} {1}\n'.format(parameters.dict.temperature, parameters.dict.thermostat_variable)

    lammps_input_file += 'run             {}\n'.format(parameters.dict.equilibrium_steps)
    lammps_input_file += 'reset_timestep  0\n'

    lammps_input_file += 'dump            aiida all custom {0} {1} element x y z\n'.format(parameters.dict.dump_rate, trajectory_file)
    lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f"\n'
    lammps_input_file += 'dump_modify     aiida sort id\n'
    lammps_input_file += 'dump_modify     aiida element {}\n'.format(names_str)

    lammps_input_file += 'run             {}\n'.format(parameters.dict.total_steps)

    return lammps_input_file


class MdCalculation(BaseLammpsCalculation, JobCalculation):

    _OUTPUT_TRAJECTORY_FILE_NAME = 'trajectory.lammpstrj'
    _OUTPUT_FILE_NAME = 'log.lammps'

    def _init_internal_params(self):
        super(MdCalculation, self)._init_internal_params()

        self._default_parser = 'lammps.md'

        self._retrieve_list = [self._OUTPUT_FILE_NAME]
        self._retrieve_temporary_list = [self._OUTPUT_TRAJECTORY_FILE_NAME]
        self._generate_input_function = generate_LAMMPS_input

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods
        retdict.update(BaseLammpsCalculation._baseclass_use_methods)


        return retdict
