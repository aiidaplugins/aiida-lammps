"""Class describing the calculation of the optimization of a structure
using LAMMPS (minimize method).
"""
# pylint: disable=fixme, duplicate-code, useless-super-delegation
from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory

from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords
from aiida_lammps.validation import validate_against_schema


class OptimizeCalculation(BaseLammpsCalculation):
    """Calculation for the optimization of the structure in LAMMPS."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default="lammps.optimize",
        )

        spec.output(
            "structure",
            valid_type=DataFactory("structure"),
            required=True,
            help="the structure output from the calculation",
        )
        spec.output(
            "trajectory_data",
            valid_type=DataFactory("lammps.trajectory"),
            required=True,
            help="forces, stresses and positions data per step",
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
        # pylint: disable=too-many-locals, too-many-arguments, too-many-statements

        parameter_data = parameter_data.get_dict()
        version_date = convert_date_string(
            parameter_data.get("lammps_version", "11 Aug 2017")
        )

        lammps_input_file = f"units          {potential_data.default_units}\n"
        lammps_input_file += "boundary        p p p\n"
        lammps_input_file += "box tilt large\n"
        lammps_input_file += f"atom_style      {potential_data.atom_style}\n"
        lammps_input_file += f"read_data       {structure_filename}\n"

        lammps_input_file += potential_data.get_input_lines(kind_symbols)

        lammps_input_file += "fix             int all box/relax"
        lammps_input_file += f' {parameter_data["relax"]["type"]}'
        lammps_input_file += f' {parameter_data["relax"]["pressure"]}'
        _temp = join_keywords(parameter_data["relax"], ignore=["type", "pressure"])
        lammps_input_file += f" {_temp}\n"

        # TODO find exact version when changes were made
        if version_date <= convert_date_string("11 Nov 2013"):
            lammps_input_file += "compute         stpa all stress/atom\n"
        else:
            lammps_input_file += "compute         stpa all stress/atom NULL\n"

        lammps_input_file += "compute         stgb all reduce sum "
        lammps_input_file += (
            "c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]\n"
        )
        lammps_input_file += (
            "variable        stress_pr equal -(c_stgb[1]+c_stgb[2]+c_stgb[3])/(3*vol)\n"
        )

        thermo_keywords = [
            "step",
            "temp",
            "press",
            "etotal",
            "v_stress_pr",
        ]
        for kwd in parameter_data.get("thermo_keywords", []):
            if kwd not in thermo_keywords:
                thermo_keywords.append(kwd)
        lammps_input_file += f'thermo_style custom {" ".join(thermo_keywords)}\n'

        if potential_data.atom_style == "charge":
            dump_variables = "element x y z  fx fy fz q"
            dump_variables += (
                " c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]"
            )
            dump_format = "%4s " + " ".join(["%16.10f"] * 13)
        else:
            dump_variables = "element x y z  fx fy fz"
            dump_variables += (
                " c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]"
            )
            dump_format = "%4s " + " ".join(["%16.10f"] * 12)

        lammps_input_file += "dump            aiida all custom 1 "
        lammps_input_file += f"{trajectory_filename} {dump_variables}\n"

        # TODO find exact version when changes were made
        if version_date <= convert_date_string("10 Feb 2015"):
            dump_mod_cmnd = "format"
        else:
            dump_mod_cmnd = "format line"

        lammps_input_file += f'dump_modify     aiida {dump_mod_cmnd} "{dump_format}"\n'

        lammps_input_file += "dump_modify     aiida sort id\n"
        lammps_input_file += f'dump_modify     aiida element {" ".join(kind_symbols)}\n'
        lammps_input_file += f'min_style       {parameter_data["minimize"]["style"]}\n'
        # lammps_input_file += 'min_style       cg\n'
        lammps_input_file += "minimize        "
        lammps_input_file += f'{parameter_data["minimize"]["energy_tolerance"]} '
        lammps_input_file += f'{parameter_data["minimize"]["force_tolerance"]} '
        lammps_input_file += f'{parameter_data["minimize"]["max_iterations"]} '
        lammps_input_file += f'{parameter_data["minimize"]["max_evaluations"]}\n'

        variables = parameter_data.get("output_variables", [])
        for var in variables:
            var_alias = var.replace("[", "_").replace("]", "_")
            lammps_input_file += f"variable {var_alias} equal {var}\n"
            lammps_input_file += (
                f'print "final_variable: {var_alias} = ${{{var_alias}}}"\n'
            )

        lammps_input_file += "variable final_energy equal etotal\n"
        lammps_input_file += 'print "final_energy: ${final_energy}"\n'

        lammps_input_file += 'print "END_OF_COMP"\n'

        return lammps_input_file

    @staticmethod
    def validate_parameters(param_data, potential_object) -> bool:
        """Validate the inputs for an optimization calculation.

        :param param_data: input parameters for the optimization calculations
        :type param_data: orm.Dict
        :param potential_object: LAMMPS potential
        :type potential_object: EmpiricalPotential
        :raises InputValidationError: if there is no parameters data passed
        :raises InputValidationError: if the units of the parameters and
            the potential are different.
        :return: whether the parameters are valid or not
        :rtype: bool
        """
        if param_data is None:
            raise InputValidationError("parameter data not set")
        validate_against_schema(param_data.get_dict(), "optimize.schema.json")

        # ensure the potential and paramters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        if "units" in param_data.get_dict():
            punits = param_data.get_dict()["units"]
            if not punits == potential_object.default_units:
                raise InputValidationError(
                    f"the units of the parameters ({punits}) and potential "
                    f"({potential_object.default_units}) are different"
                )

        return True

    def get_retrieve_lists(self):
        """Get the list of files that are supposed to be retrieved.

        :return: list with files that must be retrieved
        :rtype: list
        """
        return [], [self.options.trajectory_suffix]
