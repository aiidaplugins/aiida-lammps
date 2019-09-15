from aiida.common.exceptions import InputValidationError
from aiida.plugins import DataFactory
from aiida_lammps.calculations.lammps import BaseLammpsCalculation
from aiida_lammps.common.utils import convert_date_string, join_keywords, get_path
from aiida_lammps.validation import validate_against_schema
import six


class MdMultiCalculation(BaseLammpsCalculation):
    """Run a multi-stage molecular dynamic simulation."""

    @classmethod
    def define(cls, spec):
        super(MdMultiCalculation, cls).define(spec)

        spec.input(
            "metadata.options.parser_name",
            valid_type=six.string_types,
            default="lammps.md.multi",
        )
        spec.default_output_port = "results"

        spec.output_namespace(
            "system",
            dynamic=True,
            valid_type=DataFactory("array"),
            help="selected system data per dump step of a stage",
        )

        spec.output_namespace(
            "trajectory",
            dynamic=True,
            valid_type=DataFactory("array.trajectory"),
            help="atomic configuration data per dump step of a stage",
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
        lammps_input_file += "boundary        p p p\n"  # TODO allow non-periodic
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
        if "velocity" in pdict:
            lammps_input_file += "\n# Intial Atom Velocity\n"
        for vdict in pdict.get("velocity", []):
            lammps_input_file += "velocity all {0} {1} {2}\n".format(
                vdict["style"],
                " ".join([str(a) for a in vdict["args"]]),
                join_keywords(vdict.get("keywords", {})),
            )

        stage_names = []
        current_fixes = []
        current_dumps = []
        current_computes = []

        for stage_id, stage_dict in enumerate(pdict.get("stages")):

            stage_name = stage_dict.get("name")
            if stage_name in stage_names:
                raise ValueError("non-unique stage name: {}".format(stage_name))
            stage_names.append(stage_name)

            lammps_input_file += "\n# Stage {}: {}\n".format(stage_id, stage_name)

            # clear timestep
            # lammps_input_file += "reset_timestep  0\n"

            # Clear fixes, dumps and computes
            for fix in current_fixes:
                lammps_input_file += "unfix {}\n".format(fix)
            current_fixes = []
            for dump in current_dumps:
                lammps_input_file += "undump {}\n".format(dump)
            current_dumps = []
            for compute in current_computes:
                lammps_input_file += "uncompute {}\n".format(compute)
            current_computes = []

            # Define Computes
            for compute in stage_dict.get("computes", []):
                c_id = compute["id"]
                c_style = compute["style"]
                c_args = " ".join([str(a) for a in compute.get("args", [])])
                lammps_input_file += "compute         {0} all {1} {2}\n".format(
                    c_id, c_style, c_args
                )
                current_computes.append(c_id)

            # Define Atom Level Outputs
            output_atom_dict = stage_dict.get("output_atom", {})
            if output_atom_dict.get("dump_rate", 0):
                atom_dump_cmnds = atom_info_commands(
                    output_atom_dict.get("variables", []),
                    kind_symbols,
                    potential_data.atom_style,
                    output_atom_dict.get("dump_rate", 0),
                    "{}-{}".format(stage_name, trajectory_filename),
                    version_date,
                    "atom_info",
                )
                if atom_dump_cmnds:
                    lammps_input_file += "\n".join(atom_dump_cmnds) + "\n"
                    current_dumps.append("atom_info")

            # Define System Level Outputs
            output_sys_dict = stage_dict.get("output_system", {})
            if output_sys_dict.get("dump_rate", 0):
                sys_info_cmnds = sys_ave_commands(
                    variables=output_sys_dict.get("variables", []),
                    dump_rate=output_sys_dict.get("dump_rate", 0),
                    filename="{}-{}".format(stage_name, system_filename),
                    fix_name="sys_info",
                    average_rate=output_sys_dict.get("average_rate", None),
                )
                if sys_info_cmnds:
                    lammps_input_file += "\n".join(sys_info_cmnds) + "\n"
                    current_fixes.append("sys_info")

            # Define restart
            if stage_dict.get("restart_rate", 0):
                lammps_input_file += "restart         {0} {1}\n".format(
                    stage_dict.get("restart_rate", 0),
                    "{}-{}".format(stage_name, restart_filename),
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

        lammps_input_file += 'print "END_OF_COMP"\n'

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

        return True

    def get_retrieve_lists(self):
        return (
            [],
            ["*-" + self.options.trajectory_suffix, "*-" + self.options.system_suffix],
        )


def sys_print_commands(
    variables, dump_rate, filename, fix_name="sys_info", append=True, print_header=True
):
    """Create commands to output required system variables to a file."""
    commands = []

    if not variables:
        return commands

    if "step" not in variables:
        # always include 'step', so we can sync with the `dump` data
        variables.insert(0, "step")

    var_aliases = []
    for var in variables:
        var_alias = var.replace("[", "_").replace("]", "_")
        var_aliases.append(var_alias)
        commands.append("variable {0} equal {1}".format(var_alias, var))

    commands.append(
        'fix {0} all print {1} "{2}" {3} {4} {5} screen no'.format(
            fix_name,
            dump_rate,
            " ".join(["${{{0}}}".format(v) for v in var_aliases]),
            'title "{}"'.format(" ".join(var_aliases)) if print_header else "",
            "append" if append else "file",
            filename,
        )
    )

    return commands


def sys_ave_commands(
    variables, dump_rate, filename, fix_name="sys_info", average_rate=None
):
    """Create commands to output required system variables to a file."""
    commands = []

    if not variables:
        return commands

    # Note step is included, by default, as the first arg
    var_aliases = []
    for var in variables:
        var_alias = var.replace("[", "_").replace("]", "_")
        var_aliases.append(var_alias)
        commands.append("variable {0} equal {1}".format(var_alias, var))

    if average_rate is None:
        nevery = dump_rate
        nrep = 1
    else:
        if dump_rate % average_rate != 0:
            raise ValueError(
                "The dump rate ({}) must be a multiple of the average_rate ({})".format(
                    dump_rate, average_rate
                )
            )
        nevery = average_rate
        nrep = int(dump_rate / average_rate)

    commands.append(
        """fix {fid} all ave/time {nevery} {nrepeat} {nfreq} &
    {variables} &
    title1 "step {header}" &
    file {filename}""".format(
            fid=fix_name,
            nevery=nevery,  # compute variables every n steps
            nfreq=dump_rate,  # nfreq is the dump rate and must be a multiple of nevery
            nrepeat=nrep,  # average is over nrepeat quantities, nrepeat*nevery <= nfreq
            variables=" ".join(["v_{0}".format(v) for v in var_aliases]),
            header=" ".join(var_aliases),
            filename=filename,
        )
    )

    return commands


def atom_info_commands(
    variables,
    kind_symbols,
    atom_style,
    dump_rate,
    filename,
    version_date,
    dump_name="atom_info",
    append=True,
    pbc=True,
):
    """Create commands to output required atom variables to a file.

    Parameters
    ----------
    variables : list[str]
    kind_symbols : list[str]
        atom symbols per type
    atom_style : str
        style of atoms e.g. charge
    dump_rate : int
    filename : str
    version_date : timedate
    dump_name : str
    append : bool
        Dump snapshots to the end of the dump file (if it exists).
    pbc : bool
         Ensure all atoms are remapped to the periodic box,
         before the snapshot is written.

    Returns
    -------
    list[str]

    """
    commands = []

    if atom_style == "charge":
        dump_variables = "element x y z q".split()
    else:
        dump_variables = "element x y z".split()

    for variable in variables:
        if variable not in dump_variables:
            dump_variables.append(variable)

    commands.append(
        "dump            {0} all custom {1} {2} {3}".format(
            dump_name, dump_rate, filename, " ".join(dump_variables)
        )
    )
    if append:
        commands.append("dump_modify     {0} append yes".format(dump_name))
    # if pbc:
    # this is not available in older versions of lammps
    #     commands.append("dump_modify     {0} pbc yes".format(dump_name))

    commands.extend(
        [
            "dump_modify     {0} sort id".format(dump_name),
            "dump_modify     {0} element {1}".format(dump_name, " ".join(kind_symbols)),
        ]
    )
    return commands
