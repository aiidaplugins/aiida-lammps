from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords, get_path
from aiida_lammps.validation import validate_against_schema
import six


def sys_info_commands(
    stage_id,
    variables,
    dump_rate,
    filename,
    fix_name="sys_info",
    append=True,
    print_header=True,
):
    """Create commands to output required system variables to a file."""
    commands = []

    if not variables:
        return commands

    if "step" not in variables:
        # always include 'step', so we can sync with the `dump` data
        # NOTE `dump` includes step 0, whereas `print` starts from step 1
        variables.insert(0, "step")

    var_aliases = []
    for var in variables:
        var_alias = var.replace("[", "_").replace("]", "_")
        var_aliases.append(var_alias)
        commands.append("variable {0} equal {1}".format(var_alias, var))

    commands.append(
        'fix {0} all print {1} "{2} {3}" {4} {5} {6} screen no'.format(
            fix_name,
            dump_rate,
            stage_id,
            " ".join(["${{{0}}}".format(v) for v in var_aliases]),
            'title "stage_id {}"'.format(" ".join(var_aliases)) if print_header else "",
            "append" if append else "file",
            filename,
        )
    )

    return commands


def atom_info_commands(
    kind_symbols,
    atom_style,
    dump_rate,
    filename,
    version_date,
    dump_name="atom_info",
    append=True,
):
    """Create commands to output required atom variables to a file."""
    commands = []

    if atom_style == "charge":
        dump_variables = "element x y z q"
        dump_format = "%4s  %16.10f %16.10f %16.10f %16.10f"
    else:
        dump_variables = "element x y z"
        dump_format = "%4s  %16.10f %16.10f %16.10f"

    commands.append(
        "dump            {0} all custom {1} {2} {3}".format(
            dump_name, dump_rate, filename, dump_variables
        )
    )
    if append:
        commands.append("dump_modify     {0} append yes".format(dump_name))

    if version_date <= convert_date_string("10 Feb 2015"):
        # TODO find exact version when changes were made
        dump_mod_cmnd = "format"
    else:
        dump_mod_cmnd = "format line"

    commands.extend(
        [
            'dump_modify     {0} {1} "{2}"'.format(
                dump_name, dump_mod_cmnd, dump_format
            ),
            "dump_modify     {0} sort id".format(dump_name),
            "dump_modify     {0} element {1}".format(dump_name, " ".join(kind_symbols)),
        ]
    )
    return commands


class MdMultiCalculation(BaseLammpsCalculation):
    """Run a multi-stage molecular dynamic simulation."""

    @classmethod
    def define(cls, spec):
        super(MdMultiCalculation, cls).define(spec)

        spec.input(
            "metadata.options.parser_name",
            valid_type=six.string_types,
            default="lammps.md",
        )
        spec.default_output_port = "results"

        spec.output(
            "trajectory_data",
            valid_type=DataFactory("array.trajectory"),
            required=True,
            help="atomic configuration data per dump step",
        )
        spec.output(
            "system_data",
            valid_type=DataFactory("array"),
            required=False,
            help="selected system data per dump step",
        )

    @staticmethod
    def create_main_input_content(
        parameter_data,
        potential_data,
        kind_symbols,
        structure_filename,
        trajectory_filename,
        info_filename,
        restart_filename,
    ):

        pdict = parameter_data.get_dict()
        version_date = convert_date_string(pdict.get("lammps_version", "11 Aug 2017"))
        lammps_input_file = ""

        lammps_input_file += "# Input written to comply with LAMMPS {}\n".format(
            version_date
        )

        # Configuration setup
        lammps_input_file += "\n# Atomic Configuration\n"
        lammps_input_file += "units           {0}\n".format(
            potential_data.default_units
        )
        lammps_input_file += "boundary        p p p\n"
        lammps_input_file += "box tilt large\n"
        lammps_input_file += "atom_style      {0}\n".format(potential_data.atom_style)
        lammps_input_file += "read_data       {}\n".format(structure_filename)

        # Potential specification
        lammps_input_file += "\n# Potential Setup\n"
        lammps_input_file += potential_data.get_input_lines(kind_symbols)

        # Modify pairwise neighbour list creation
        lammps_input_file += "\n# General Setup\n"
        if "neighbor" in pdict:
            # neighbor skin_dist bin/nsq/multi
            lammps_input_file += "neighbor {0} {1}\n".format(
                pdict["neighbor"][0], pdict["neighbor"][1]
            )
        if "neigh_modify" in pdict:
            # e.g. 'neigh_modify every 1 delay 0 check no\n'
            lammps_input_file += "neigh_modify {}\n".format(
                join_keywords(pdict["neigh_modify"])
            )
        # Define Timestep
        lammps_input_file += "timestep        {}\n".format(pdict["timestep"])

        # Define computation/printing of thermodynamic info
        lammps_input_file += "\n# Thermodynamic Information Output\n"
        thermo_keywords = ["step", "temp", "epair", "emol", "etotal", "press"]
        for kwd in pdict.get("thermo_keywords", []):
            if kwd not in thermo_keywords:
                thermo_keywords.append(kwd)
        lammps_input_file += "thermo_style custom {}\n".format(
            " ".join(thermo_keywords)
        )
        lammps_input_file += "thermo          1000\n"  # TODO make variable?

        # Setup initial velocities of atoms
        for vdict in pdict.get("velocity", []):
            lammps_input_file += "\n# Intial Atom Velocity\n"
            lammps_input_file += "velocity all {0} {1} {2}\n".format(
                vdict["style"],
                " ".join([str(a) for a in vdict["args"]]),
                " ".join(
                    ["{} {}".format(k, v) for k, v in vdict.get("keywords", {}).items()]
                ),
            )

        current_fixes = []
        current_dumps = []
        print_sys_header = True

        for stage_id, stage_dict in enumerate(pdict.get("stages")):

            lammps_input_file += "\n# Stage {}: {}\n".format(
                stage_id, stage_dict.get("name")
            )

            # clear timestep
            # lammps_input_file += "reset_timestep  0\n"

            # Clear fixes and dumps
            for fix in current_fixes:
                lammps_input_file += "unfix {}\n".format(fix)
            current_fixes = []
            for dump in current_dumps:
                lammps_input_file += "undump {}\n".format(dump)
            current_dumps = []

            # Define File Outputs
            dump_rate = stage_dict.get("dump_rate", 0)
            if dump_rate:
                atom_dump_cmnds = atom_info_commands(
                    kind_symbols,
                    potential_data.atom_style,
                    dump_rate,
                    trajectory_filename,
                    version_date,
                    "atom_info",
                )
                if atom_dump_cmnds:
                    lammps_input_file += "\n".join(atom_dump_cmnds) + "\n"
                    current_dumps.append("atom_info")

                sys_info_cmnds = sys_info_commands(
                    stage_id,
                    pdict.get("output_variables", []),
                    dump_rate,
                    info_filename,
                    "sys_info",
                    print_header=print_sys_header,
                )
                if sys_info_cmnds:
                    lammps_input_file += "\n".join(sys_info_cmnds) + "\n"
                    current_fixes.append("sys_info")
                    print_sys_header = False  # only print the header once

            # Define restart
            if stage_dict.get("restart", 0):
                lammps_input_file += "restart         {0} {1}\n".format(
                    stage_dict.get("restart", 0),
                    "{}-{}".format(stage_dict.get("name", ""), restart_filename),
                )
            else:
                lammps_input_file += "restart         0\n"

            # Define time integration method
            lammps_input_file += "fix             int all {0} {1} {2}\n".format(
                get_path(stage_dict, ["integration", "style"]),
                join_keywords(
                    get_path(
                        stage_dict,
                        ["integration", "constraints"],
                        {},
                        raise_error=False,
                    )
                ),
                join_keywords(
                    get_path(
                        stage_dict, ["integration", "keywords"], {}, raise_error=False
                    )
                ),
            )
            current_fixes.append("int")

            # Run
            lammps_input_file += "run             {}\n".format(
                stage_dict.get("steps", 0)
            )

        lammps_input_file += "\n# Final Commands\n"
        # output final energy
        lammps_input_file += "variable final_energy equal etotal\n"
        lammps_input_file += 'print "final_energy: ${final_energy}"\n'

        return lammps_input_file

    def validate_parameters(self, param_data, potential_object):
        if param_data is None:
            raise InputValidationError("parameter data not set")
        validate_against_schema(param_data.get_dict(), "md-multi.schema.json")

        # ensure the potential and parameters are in the same unit systems
        # TODO convert between unit systems (e.g. using https://pint.readthedocs.io)
        punits = param_data.get_dict()["units"]
        if not punits == potential_object.default_units:
            raise InputValidationError(
                "the units of the parameters ({}) and potential ({}) are different".format(
                    punits, potential_object.default_units
                )
            )

        self._retrieve_list += []
        if self.options.trajectory_name not in self._retrieve_temporary_list:
            self._retrieve_temporary_list += [self.options.trajectory_name]
        if self.options.info_filename not in self._retrieve_temporary_list:
            self._retrieve_temporary_list += [self.options.info_filename]

        return True
