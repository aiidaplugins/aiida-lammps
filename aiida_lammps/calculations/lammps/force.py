"""Single point calculation of the energy in LAMMPS."""
# pylint: disable=fixme, duplicate-code, useless-super-delegation
from aiida import orm

from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string
from aiida_lammps.validation import validate_against_schema


class ForceCalculation(BaseLammpsCalculation):
    """Calculation of a single point in LAMMPS, to obtain the energy of the system."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default="lammps.force",
        )

        spec.output(
            "arrays",
            valid_type=orm.ArrayData,
            required=True,
            help="force data per atom",
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
        # pylint: disable=too-many-arguments, too-many-locals
        version_date = convert_date_string(
            parameter_data.base.attributes.get("lammps_version", "11 Aug 2017")
        )

        lammps_input_file = f"units          {potential_data.default_units}\n"
        lammps_input_file += "boundary        p p p\n"
        lammps_input_file += "box tilt large\n"
        lammps_input_file += f"atom_style      {potential_data.atom_style}\n"

        lammps_input_file += f"read_data       {structure_filename}\n"

        lammps_input_file += potential_data.get_input_lines(kind_symbols)

        lammps_input_file += "neighbor        0.3 bin\n"
        lammps_input_file += "neigh_modify    every 1 delay 0 check no\n"

        thermo_keywords = ["step", "temp", "epair", "emol", "etotal", "press"]
        for kwd in parameter_data.base.attributes.get("thermo_keywords", []):
            if kwd not in thermo_keywords:
                thermo_keywords.append(kwd)
        lammps_input_file += f'thermo_style custom {" ".join(thermo_keywords)}\n'

        if potential_data.atom_style == "charge":
            dump_variables = "element x y z fx fy fz q"
            dump_format = (
                "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f %16.10f"
            )
        else:
            dump_variables = "element x y z fx fy fz"
            dump_format = "%4s  %16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f"

        lammps_input_file += "dump            aiida all custom 1"
        lammps_input_file += f" {trajectory_filename} {dump_variables}\n"

        # TODO find exact version when changes were made
        if version_date <= convert_date_string("10 Feb 2015"):
            dump_mod_cmnd = "format"
        else:
            dump_mod_cmnd = "format line"

        lammps_input_file += f'dump_modify     aiida {dump_mod_cmnd} "{dump_format}"\n'

        lammps_input_file += "dump_modify     aiida sort id\n"
        lammps_input_file += f'dump_modify     aiida element {" ".join(kind_symbols)}\n'

        lammps_input_file += "run             0\n"

        variables = parameter_data.base.attributes.get("output_variables", [])
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
    def validate_parameters(param_data, potential_object):
        """Validate the parameters for a force calculation.

        :param param_data: parameters for the LAMMPS force calculation
        :type param_data: orm.Dict
        """
        if param_data is not None:
            validate_against_schema(param_data.get_dict(), "force.schema.json")

    def get_retrieve_lists(self):
        """Get list of files to be retrieved when the calculation is done.

        :return: list of files to be retrieved.
        :rtype: list
        """
        return [self.options.trajectory_suffix], []
