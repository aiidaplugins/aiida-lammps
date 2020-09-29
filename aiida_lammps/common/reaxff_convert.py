"""Module for parsing REAXFF input files.

Note: this module is copied directly from aiida-crystal17 v0.10.0b5
"""
import copy
import re

from aiida_lammps.validation import validate_against_schema

INDEX_SEP = "-"

KEYS_GLOBAL = (
    "reaxff0_boc1",
    "reaxff0_boc2",
    "reaxff3_coa2",
    "Triple bond stabilisation 1",
    "Triple bond stabilisation 2",
    "C2-correction",
    "reaxff0_ovun6",
    "Triple bond stabilisation",
    "reaxff0_ovun7",
    "reaxff0_ovun8",
    "Triple bond stabilization energy",
    "Lower Taper-radius",
    "Upper Taper-radius",
    "reaxff2_pen2",
    "reaxff0_val7",
    "reaxff0_lp1",
    "reaxff0_val9",
    "reaxff0_val10",
    "Not used 2",
    "reaxff0_pen2",
    "reaxff0_pen3",
    "reaxff0_pen4",
    "Not used 3",
    "reaxff0_tor2",
    "reaxff0_tor3",
    "reaxff0_tor4",
    "Not used 4",
    "reaxff0_cot2",
    "reaxff0_vdw1",
    "bond order cutoff",
    "reaxff3_coa4",
    "reaxff0_ovun4",
    "reaxff0_ovun3",
    "reaxff0_val8",
    "Not used 5",
    "Not used 6",
    "Not used 7",
    "Not used 8",
    "reaxff3_coa3",
)

# TODO some variables lammps sets as global are actually species dependant in GULP, how to handle these?

KEYS_1BODY = (
    "reaxff1_radii1",
    "reaxff1_valence1",
    "mass",
    "reaxff1_morse3",
    "reaxff1_morse2",
    "reaxff_gamma",
    "reaxff1_radii2",
    "reaxff1_valence3",
    "reaxff1_morse1",
    "reaxff1_morse4",
    "reaxff1_valence4",
    "reaxff1_under",
    "dummy1",
    "reaxff_chi",
    "reaxff_mu",
    "dummy2",
    "reaxff1_radii3",
    "reaxff1_lonepair2",
    "dummy3",
    "reaxff1_over2",
    "reaxff1_over1",
    "reaxff1_over3",
    "dummy4",
    "dummy5",
    "reaxff1_over4",
    "reaxff1_angle1",
    "dummy11",
    "reaxff1_valence2",
    "reaxff1_angle2",
    "dummy6",
    "dummy7",
    "dummy8",
)

KEYS_2BODY_BONDS = (
    "reaxff2_bond1",
    "reaxff2_bond2",
    "reaxff2_bond3",
    "reaxff2_bond4",
    "reaxff2_bo5",
    "reaxff2_bo7",
    "reaxff2_bo6",
    "reaxff2_over",
    "reaxff2_bond5",
    "reaxff2_bo3",
    "reaxff2_bo4",
    "dummy1",
    "reaxff2_bo1",
    "reaxff2_bo2",
    "reaxff2_bo8",
    "reaxff2_pen1",
)

KEYS_2BODY_OFFDIAG = [
    "reaxff2_morse1",
    "reaxff2_morse3",
    "reaxff2_morse2",
    "reaxff2_morse4",
    "reaxff2_morse5",
    "reaxff2_morse6",
]

KEYS_3BODY_ANGLES = (
    "reaxff3_angle1",
    "reaxff3_angle2",
    "reaxff3_angle3",
    "reaxff3_coa1",
    "reaxff3_angle5",
    "reaxff3_penalty",
    "reaxff3_angle4",
)

KEYS_3BODY_HBOND = (
    "reaxff3_hbond1",
    "reaxff3_hbond2",
    "reaxff3_hbond3",
    "reaxff3_hbond4",
)

KEYS_4BODY_TORSION = (
    "reaxff4_torsion1",
    "reaxff4_torsion2",
    "reaxff4_torsion3",
    "reaxff4_torsion4",
    "reaxff4_torsion5",
    "dummy1",
    "dummy2",
)

DEFAULT_TOLERANCES = (
    # ReaxFF angle/torsion bond order threshold,
    # for bond orders in valence, penalty and 3-body conjugation
    # GULP default: 0.001
    ("anglemin", 0.001),
    # ReaxFF bond order double product threshold,
    # for the product of bond orders (1-2 x 2-3, where 2 = pivot)
    # Hard coded to 0.001 in original code, but this leads to discontinuities
    # GULP default: 0.000001
    ("angleprod", 0.00001),
    # ReaxFF hydrogen-bond bond order threshold
    # Hard coded to 0.01 in original code.
    # GULP default: 0.01
    ("hbondmin", 0.01),
    # ReaxFF H-bond cutoff
    # Hard coded to 7.5 Ang in original code.
    # GULP default: 7.5
    ("hbonddist", 7.5),
    # ReaxFF bond order triple product threshold,
    # for the product of bond orders (1-2 x 2-3 x 3-4)
    # GULP default: 0.000000001
    ("torsionprod", 0.00001),
)


def split_numbers(string):
    """Get a list of numbers from a string (even with no spacing).

    :type string: str
    :type as_decimal: bool
    :param as_decimal: if True return floats as decimal.Decimal objects

    :rtype: list

    :Example:

    >>> split_numbers("1")
    [1.0]

    >>> split_numbers("1 2")
    [1.0, 2.0]

    >>> split_numbers("1.1 2.3")
    [1.1, 2.3]

    >>> split_numbers("1e-3")
    [0.001]

    >>> split_numbers("-1-2")
    [-1.0, -2.0]

    >>> split_numbers("1e-3-2")
    [0.001, -2.0]

    """
    _match_number = re.compile("-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *[+-]?\\ *[0-9]+)?")
    string = string.replace(" .", " 0.")
    string = string.replace("-.", "-0.")
    return [float(s) for s in re.findall(_match_number, string)]


def read_lammps_format(lines, tolerances=None):
    """Read a reaxff file, in lammps format, to a standardised potential dictionary.

    Parameters
    ----------
    lines : list[str]
    tolerances : dict or None
        tolerances to set, that are not specified in the file.

    Returns
    -------
    dict

    Notes
    -----
    Default tolerances:

    - anglemin: 0.001
    - angleprod: 0.001
    - hbondmin: 0.01
    - hbonddist: 7.5
    - torsionprod: 1e-05

    """
    output = {
        "description": lines[0],
        "global": {},
        "species": ["X core"],  # X is always first
        "1body": {},
        "2body": {},
        "3body": {},
        "4body": {},
    }

    lineno = 1

    # Global parameters
    if lines[lineno].split()[0] != str(len(KEYS_GLOBAL)):
        raise IOError("Expecting {} global parameters".format(len(KEYS_GLOBAL)))

    for key in KEYS_GLOBAL:
        lineno += 1
        output["global"][key] = float(lines[lineno].split()[0])

    output["global"][
        "reaxff2_pen3"
    ] = 1.0  # this is not provided by lammps, but is used by GULP

    tolerances = tolerances or {}
    output["global"].update({k: tolerances.get(k, v) for k, v in DEFAULT_TOLERANCES})

    # one-body parameters
    lineno += 1
    num_species = int(lines[lineno].split()[0])
    lineno += 3
    idx = 1
    for i in range(num_species):
        lineno += 1
        symbol, values = lines[lineno].split(None, 1)
        if symbol == "X":
            species_idx = 0  # the X symbol is always assigned index 0
        else:
            species_idx = idx
            idx += 1
            output["species"].append(symbol + " core")
        values = split_numbers(values)
        for _ in range(3):
            lineno += 1
            values.extend(split_numbers(lines[lineno]))

        if len(values) != len(KEYS_1BODY):
            raise Exception(
                "number of values different than expected for species {0}, "
                "{1} != {2}".format(symbol, len(values), len(KEYS_1BODY))
            )

        key_map = {k: v for k, v in zip(KEYS_1BODY, values)}
        key_map["reaxff1_lonepair1"] = 0.5 * (
            key_map["reaxff1_valence3"] - key_map["reaxff1_valence1"]
        )

        output["1body"][str(species_idx)] = key_map

    # two-body bond parameters
    lineno += 1
    num_lines = int(lines[lineno].split()[0])
    lineno += 2
    for _ in range(num_lines):
        values = split_numbers(lines[lineno]) + split_numbers(lines[lineno + 1])
        species_idx1 = int(values.pop(0))
        species_idx2 = int(values.pop(0))
        key_name = "{}-{}".format(species_idx1, species_idx2)
        lineno += 2

        if len(values) != len(KEYS_2BODY_BONDS):
            raise Exception(
                "number of bond values different than expected for key {0}, "
                "{1} != {2}".format(key_name, len(values), len(KEYS_2BODY_BONDS))
            )

        output["2body"][key_name] = {k: v for k, v in zip(KEYS_2BODY_BONDS, values)}

    # two-body off-diagonal parameters
    num_lines = int(lines[lineno].split()[0])
    lineno += 1
    for _ in range(num_lines):
        values = split_numbers(lines[lineno])
        species_idx1 = int(values.pop(0))
        species_idx2 = int(values.pop(0))
        key_name = "{}-{}".format(species_idx1, species_idx2)
        lineno += 1

        if len(values) != len(KEYS_2BODY_OFFDIAG):
            raise Exception(
                "number of off-diagonal values different than expected for key {0} (line {1}), "
                "{2} != {3}".format(
                    key_name, lineno - 1, len(values), len(KEYS_2BODY_OFFDIAG)
                )
            )

        output["2body"].setdefault(key_name, {}).update(
            {k: v for k, v in zip(KEYS_2BODY_OFFDIAG, values)}
        )

    # three-body angle parameters
    num_lines = int(lines[lineno].split()[0])
    lineno += 1
    for _ in range(num_lines):
        values = split_numbers(lines[lineno])
        species_idx1 = int(values.pop(0))
        species_idx2 = int(values.pop(0))
        species_idx3 = int(values.pop(0))
        key_name = "{}-{}-{}".format(species_idx1, species_idx2, species_idx3)
        lineno += 1

        if len(values) != len(KEYS_3BODY_ANGLES):
            raise Exception(
                "number of angle values different than expected for key {0} (line {1}), "
                "{2} != {3}".format(
                    key_name, lineno - 1, len(values), len(KEYS_3BODY_ANGLES)
                )
            )

        output["3body"].setdefault(key_name, {}).update(
            {k: v for k, v in zip(KEYS_3BODY_ANGLES, values)}
        )

    # four-body torsion parameters
    num_lines = int(lines[lineno].split()[0])
    lineno += 1
    for _ in range(num_lines):
        values = split_numbers(lines[lineno])
        species_idx1 = int(values.pop(0))
        species_idx2 = int(values.pop(0))
        species_idx3 = int(values.pop(0))
        species_idx4 = int(values.pop(0))
        key_name = "{}-{}-{}-{}".format(
            species_idx1, species_idx2, species_idx3, species_idx4
        )
        lineno += 1

        if len(values) != len(KEYS_4BODY_TORSION):
            raise Exception(
                "number of torsion values different than expected for key {0} (line {1}), "
                "{2} != {3}".format(
                    key_name, lineno - 1, len(values), len(KEYS_4BODY_TORSION)
                )
            )

        output["4body"].setdefault(key_name, {}).update(
            {k: v for k, v in zip(KEYS_4BODY_TORSION, values)}
        )

    # three-body h-bond parameters
    num_lines = int(lines[lineno].split()[0])
    lineno += 1
    for _ in range(num_lines):
        values = split_numbers(lines[lineno])
        species_idx1 = int(values.pop(0))
        species_idx2 = int(values.pop(0))
        species_idx3 = int(values.pop(0))
        key_name = "{}-{}-{}".format(species_idx1, species_idx2, species_idx3)
        lineno += 1

        if len(values) != len(KEYS_3BODY_HBOND):
            raise Exception(
                "number of h-bond values different than expected for key {0} (line {1}), "
                "{2} != {3}".format(
                    key_name, lineno - 1, len(values), len(KEYS_3BODY_HBOND)
                )
            )

        output["3body"].setdefault(key_name, {}).update(
            {k: v for k, v in zip(KEYS_3BODY_HBOND, values)}
        )

    return output


def format_lammps_value(value):
    return "{:.4f}".format(value)


def write_lammps_format(data):
    """Write a reaxff file, in lammps format, from a standardised potential dictionary."""
    # validate dictionary
    validate_against_schema(data, "reaxff.schema.json")

    output = [data["description"]]

    # Global parameters
    output.append("{} ! Number of general parameters".format(len(KEYS_GLOBAL)))
    for key in KEYS_GLOBAL:
        output.append("{0:.4f} ! {1}".format(data["global"][key], key))

    # one-body parameters
    output.extend(
        [
            "{0} ! Nr of atoms; cov.r; valency;a.m;Rvdw;Evdw;gammaEEM;cov.r2;#".format(
                len(data["species"])
            ),
            "alfa;gammavdW;valency;Eunder;Eover;chiEEM;etaEEM;n.u.",
            "cov r3;Elp;Heat inc.;n.u.;n.u.;n.u.;n.u.",
            "ov/un;val1;n.u.;val3,vval4",
        ]
    )
    idx_map = {}
    i = 1
    x_species_line = None
    for idx, species in enumerate(data["species"]):
        if species.endswith("shell"):
            raise ValueError(
                "only core species can be used for reaxff, not shell: {}".format(
                    species
                )
            )
        species = species[:-5]
        # X is not always present in 1body, even if it is used in nbody terms
        # see e.g. https://github.com/lammps/lammps/blob/master/potentials/ffield.reax.cho
        if species == "X" and str(idx) not in data["1body"]:
            species_lines = []
        else:
            species_lines = [
                species
                + " "
                + " ".join(
                    [
                        format_lammps_value(data["1body"][str(idx)][k])
                        for k in KEYS_1BODY[:8]
                    ]
                ),
                " ".join(
                    [
                        format_lammps_value(data["1body"][str(idx)][k])
                        for k in KEYS_1BODY[8:16]
                    ]
                ),
                " ".join(
                    [
                        format_lammps_value(data["1body"][str(idx)][k])
                        for k in KEYS_1BODY[16:24]
                    ]
                ),
                " ".join(
                    [
                        format_lammps_value(data["1body"][str(idx)][k])
                        for k in KEYS_1BODY[24:32]
                    ]
                ),
            ]
        if species == "X":
            # X is always index 0, but must be last in the species list
            idx_map[str(idx)] = "0"
            x_species_line = species_lines
        else:
            idx_map[str(idx)] = str(i)
            i += 1
            output.extend(species_lines)
    if x_species_line:
        output.extend(x_species_line)

    # two-body angle parameters
    suboutout = []
    for key in sorted(data["2body"]):
        subdata = data["2body"][key]
        if not set(subdata.keys()).issuperset(KEYS_2BODY_BONDS):
            continue
        suboutout.extend(
            [
                " ".join([idx_map[k] for k in key.split(INDEX_SEP)])
                + " "
                + " ".join(
                    [format_lammps_value(subdata[k]) for k in KEYS_2BODY_BONDS[:8]]
                ),
                " ".join(
                    [format_lammps_value(subdata[k]) for k in KEYS_2BODY_BONDS[8:16]]
                ),
            ]
        )

    output.extend(
        [
            "{0} ! Nr of bonds; Edis1;LPpen;n.u.;pbe1;pbo5;13corr;pbo6".format(
                int(len(suboutout) / 2)
            ),
            "pbe2;pbo3;pbo4;n.u.;pbo1;pbo2;ovcorr",
        ]
        + suboutout
    )

    # two-body off-diagonal parameters
    suboutout = []
    for key in sorted(data["2body"]):
        subdata = data["2body"][key]
        if not set(subdata.keys()).issuperset(KEYS_2BODY_OFFDIAG):
            continue
        suboutout.extend(
            [
                " ".join([idx_map[k] for k in key.split(INDEX_SEP)])
                + " "
                + " ".join(
                    [format_lammps_value(subdata[k]) for k in KEYS_2BODY_OFFDIAG]
                )
            ]
        )

    output.extend(
        [
            "{0} ! Nr of off-diagonal terms; Ediss;Ro;gamma;rsigma;rpi;rpi2".format(
                len(suboutout)
            )
        ]
        + suboutout
    )

    # three-body angle parameters
    suboutout = []
    for key in sorted(data["3body"]):
        subdata = data["3body"][key]
        if not set(subdata.keys()).issuperset(KEYS_3BODY_ANGLES):
            continue
        suboutout.extend(
            [
                " ".join([idx_map[k] for k in key.split(INDEX_SEP)])
                + " "
                + " ".join([format_lammps_value(subdata[k]) for k in KEYS_3BODY_ANGLES])
            ]
        )

    output.extend(
        ["{0} ! Nr of angles;at1;at2;at3;Thetao,o;ka;kb;pv1;pv2".format(len(suboutout))]
        + suboutout
    )

    # four-body torsion parameters
    suboutout = []
    for key in sorted(data["4body"]):
        subdata = data["4body"][key]
        if not set(subdata.keys()).issuperset(KEYS_4BODY_TORSION):
            continue
        suboutout.extend(
            [
                " ".join([idx_map[k] for k in key.split(INDEX_SEP)])
                + " "
                + " ".join(
                    [format_lammps_value(subdata[k]) for k in KEYS_4BODY_TORSION]
                )
            ]
        )

    output.extend(
        [
            "{0} ! Nr of torsions;at1;at2;at3;at4;;V1;V2;V3;V2(BO);vconj;n.u;n".format(
                len(suboutout)
            )
        ]
        + suboutout
    )

    # three-body h-bond parameters
    suboutout = []
    for key in sorted(data["3body"]):
        subdata = data["3body"][key]
        if not set(subdata.keys()).issuperset(KEYS_3BODY_HBOND):
            continue
        suboutout.extend(
            [
                " ".join([idx_map[k] for k in key.split(INDEX_SEP)])
                + " "
                + " ".join([format_lammps_value(subdata[k]) for k in KEYS_3BODY_HBOND])
            ]
        )

    output.extend(
        ["{0} ! Nr of hydrogen bonds;at1;at2;at3;Rhb;Dehb;vhb1".format(len(suboutout))]
        + suboutout
    )

    output.append("")

    return "\n".join(output)


def filter_by_species(data, species):
    """filter a potential dict by a subset of species

    Parameters
    ----------
    data : dict
        a potential or fitting dict
    species : list[str]
        the species to filter by

    Returns
    -------
    dict
        data filtered by species and with all species index keys re-indexed

    Raises
    ------
    KeyError
        if the data does not adhere to the potential or fitting jsonschema
    AssertionError
        if the filter set is not a subset of the available species

    """
    species = sorted(list(set(species)))

    if not set(species).issubset(data["species"]):
        raise AssertionError(
            "the filter set ({}) is not a subset of the available species ({})".format(
                set(species), set(data["species"])
            )
        )
    data = copy.deepcopy(data)
    indices = set([str(i) for i, s in enumerate(data["species"]) if s in species])

    def convert_indices(key):
        return INDEX_SEP.join(
            [str(species.index(data["species"][int(k)])) for k in key.split(INDEX_SEP)]
        )

    for key in ["1body", "2body", "3body", "4body"]:
        if key not in data:
            continue
        data[key] = {
            convert_indices(k): v
            for k, v in data[key].items()
            if indices.issuperset(k.split(INDEX_SEP))
        }

    data["species"] = species

    return data
