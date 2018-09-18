import numpy as np
from aiida.common.exceptions import InputValidationError
from aiida.common.utils import classproperty
from aiida.orm.calculation.job import JobCalculation
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords, get_path
from aiida_lammps.validation import validate_with_json


def generate_LAMMPS_input(parameters,
                          potential_obj,
                          structure_file='potential.pot',
                          trajectory_file='trajectory.lammpstr'):

    pdict = parameters.get_dict()

    random_number = np.random.randint(10000000)

    names_str = ' '.join(potential_obj._names)

    lammps_date = convert_date_string(pdict.get("lammps_version", None))

    lammps_input_file = 'units           {0}\n'.format(potential_obj.default_units)
    lammps_input_file += 'boundary        p p p\n'
    lammps_input_file += 'box tilt large\n'
    lammps_input_file += 'atom_style      {0}\n'.format(potential_obj.atom_style)
    lammps_input_file += 'read_data       {}\n'.format(structure_file)

    lammps_input_file += potential_obj.get_input_potential_lines()

    if "neighbor" in pdict:
        lammps_input_file += "neighbor {0} {1}\n".format(pdict["neighbor"][0],
                                                         pdict["neighbor"][1])
    if "neigh_modify" in pdict:
        lammps_input_file += "neigh_modify {}\n".format(join_keywords(pdict["neigh_modify"]))

    # lammps_input_file += 'neighbor        0.3 bin\n'
    # lammps_input_file += 'neigh_modify    every 1 delay 0 check no\n'

    lammps_input_file += 'timestep        {}\n'.format(pdict["timestep"])

    lammps_input_file += 'thermo_style    custom step etotal temp vol press\n'
    lammps_input_file += 'thermo          1000\n'

    initial_temp, _, _ = get_path(pdict, ["integration", "constraints", "temp"],
                                 default=[None, None, None], raise_error=False)

    if initial_temp is not None:
        lammps_input_file += 'velocity        all create {0} {1} dist gaussian mom yes\n'.format(
            initial_temp, random_number)
        lammps_input_file += 'velocity        all scale {}\n'.format(initial_temp)

    lammps_input_file += 'fix             int all {0} {1} {2}\n'.format(
        get_path(pdict, ["integration", "style"]),
        join_keywords(get_path(pdict, ["integration", "constraints"], {}, raise_error=False)),
        join_keywords(get_path(pdict, ["integration", "keywords"], {}, raise_error=False)))

    # lammps_input_file += 'fix             int all nvt temp {0} {0} {1}\n'.format(parameters.dict.temperature,
    #                                                                              parameters.dict.thermostat_variable)

    lammps_input_file += 'run             {}\n'.format(parameters.dict.equilibrium_steps)
    lammps_input_file += 'reset_timestep  0\n'

    lammps_input_file += 'dump            aiida all custom {0} {1} element x y z\n'.format(parameters.dict.dump_rate,
                                                                                           trajectory_file)

    # TODO find exact version when changes were made
    if lammps_date <= convert_date_string('10 Feb 2015'):
        lammps_input_file += 'dump_modify     aiida format "%4s  %16.10f %16.10f %16.10f"\n'
    else:
        lammps_input_file += 'dump_modify     aiida format line "%4s  %16.10f %16.10f %16.10f"\n'

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
        self._retrieve_temporary_list = [self._OUTPUT_TRAJECTORY_FILE_NAME, self._INPUT_UNITS]
        self._generate_input_function = generate_LAMMPS_input

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods
        retdict.update(BaseLammpsCalculation._baseclass_use_methods)

        return retdict

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

        return True
