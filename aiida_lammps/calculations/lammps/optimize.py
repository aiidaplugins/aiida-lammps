from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory

from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords
from aiida_lammps.validation import validate_against_schema


class OptimizeCalculation(BaseLammpsCalculation):
    @classmethod
    def define(cls, spec):
        super(OptimizeCalculation, cls).define(spec)

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

        parameter_data = parameter_data.get_dict()
        version_date = convert_date_string(
            parameter_data.get("lammps_version", "11 Aug 2017")
        )

        lammps_input_file = "units          {0}\n".format(potential_data.default_units)
        lammps_input_file += "boundary        p p p\n"
        lammps_input_file += "box tilt large\n"
        lammps_input_file += "atom_style      {0}\n".format(potential_data.atom_style)
        lammps_input_file += "read_data       {}\n".format(structure_filename)

        lammps_input_file += potential_data.get_input_lines(kind_symbols)

        lammps_input_file += "fix             int all box/relax {} {} {}\n".format(
            parameter_data["relax"]["type"],
            parameter_data["relax"]["pressure"],
            join_keywords(parameter_data["relax"], ignore=["type", "pressure"]),
        )

        # TODO find exact version when changes were made
        if version_date <= convert_date_string("11 Nov 2013"):
            lammps_input_file += "compute         stpa all stress/atom\n"
        else:
            lammps_input_file += "compute         stpa all stress/atom NULL\n"

        lammps_input_file += "compute         stgb all reduce sum c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]\n"
        lammps_input_file += (
            "variable        stress_pr equal -(c_stgb[1]+c_stgb[2]+c_stgb[3])/(3*vol)\n"
        )
        # lammps_input_file += "variable        stress_xx equal c_stgb[1]\n"
        # lammps_input_file += "variable        stress_yy equal c_stgb[2]\n"
        # lammps_input_file += "variable        stress_zz equal c_stgb[3]\n"
        # lammps_input_file += "variable        stress_xy equal c_stgb[4]\n"
        # lammps_input_file += "variable        stress_xz equal c_stgb[5]\n"
        # lammps_input_file += "variable        stress_yz equal c_stgb[6]\n"

        thermo_keywords = [
            "step",
            "temp",
            "press",
            "etotal",
            "v_stress_pr",
            # "c_stgb[1]",
            # "c_stgb[2]",
            # "c_stgb[3]",
            # "c_stgb[4]",
            # "c_stgb[5]",
            # "c_stgb[6]",
        ]
        for kwd in parameter_data.get("thermo_keywords", []):
            if kwd not in thermo_keywords:
                thermo_keywords.append(kwd)
        lammps_input_file += "thermo_style custom {}\n".format(
            " ".join(thermo_keywords)
        )

        if potential_data.atom_style == "charge":
            dump_variables = "element x y z  fx fy fz q c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]"
            dump_format = "%4s " + " ".join(["%16.10f"] * 13)
        else:
            dump_variables = "element x y z  fx fy fz c_stpa[1] c_stpa[2] c_stpa[3] c_stpa[4] c_stpa[5] c_stpa[6]"
            dump_format = "%4s " + " ".join(["%16.10f"] * 12)

        lammps_input_file += "dump            aiida all custom 1 {0} {1}\n".format(
            trajectory_filename, dump_variables
        )

        # TODO find exact version when changes were made
        if version_date <= convert_date_string("10 Feb 2015"):
            dump_mod_cmnd = "format"
        else:
            dump_mod_cmnd = "format line"

        lammps_input_file += 'dump_modify     aiida {0} "{1}"\n'.format(
            dump_mod_cmnd, dump_format
        )

        lammps_input_file += "dump_modify     aiida sort id\n"
        lammps_input_file += "dump_modify     aiida element {}\n".format(
            " ".join(kind_symbols)
        )
        lammps_input_file += "min_style       {}\n".format(
            parameter_data["minimize"]["style"]
        )
        # lammps_input_file += 'min_style       cg\n'
        lammps_input_file += "minimize        {} {} {} {}\n".format(
            parameter_data["minimize"]["energy_tolerance"],
            parameter_data["minimize"]["force_tolerance"],
            parameter_data["minimize"]["max_iterations"],
            parameter_data["minimize"]["max_evaluations"],
        )

        variables = parameter_data.get("output_variables", [])
        for var in variables:
            var_alias = var.replace("[", "_").replace("]", "_")
            lammps_input_file += "variable {0} equal {1}\n".format(var_alias, var)
            lammps_input_file += 'print "final_variable: {0} = ${{{0}}}"\n'.format(
                var_alias
            )

        lammps_input_file += "variable final_energy equal etotal\n"
        lammps_input_file += 'print "final_energy: ${final_energy}"\n'

        # lammps_input_file += 'print "final_variable: stress_pr = ${stress_pr}"\n'
        # lammps_input_file += 'print "final_cell: $(xlo) $(xhi) $(xy) $(ylo) $(yhi) $(xz) $(zlo) $(zhi) $(yz)"\n'
        # lammps_input_file += 'print "final_stress: ${stress_xx} ${stress_yy} ${stress_zz} ${stress_xy} ${stress_xz} ${stress_yz}"\n'

        lammps_input_file += 'print "END_OF_COMP"\n'

        return lammps_input_file

    @staticmethod
    def validate_parameters(param_data, potential_object):
        if param_data is None:
            raise InputValidationError("parameter data not set")
        validate_against_schema(param_data.get_dict(), "optimize.schema.json")

        # ensure the potential and paramters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        if "units" in param_data.get_dict():
            punits = param_data.get_dict()["units"]
            if not punits == potential_object.default_units:
                raise InputValidationError(
                    "the units of the parameters ({}) and potential ({}) are different".format(
                        punits, potential_object.default_units
                    )
                )

        return True

    def get_retrieve_lists(self):
        return [], [self.options.trajectory_suffix]
