"""
Sets up an example for the calculation of bcc Fe using ``aiida-lammps``.
"""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import run_get_node, submit
from aiida.plugins import CalculationFactory
import numpy as np

from aiida_lammps.data.potential import LammpsPotentialData


def generate_structure() -> orm.StructureData:
    """
    Generates the structure for the calculation.

    It will create a bcc structure in a square lattice.

    :return: structure to be used in the calculation
    :rtype: orm.StructureData
    """

    cell = [
        [2.848116, 0.000000, 0.000000],
        [0.000000, 2.848116, 0.000000],
        [0.000000, 0.000000, 2.848116],
    ]

    positions = [
        (0.0000000, 0.0000000, 0.0000000),
        (0.5000000, 0.5000000, 0.5000000),
    ]
    fractional = True

    symbols = ["Fe", "Fe"]
    names = ["Fe1", "Fe2"]

    structure = orm.StructureData(cell=cell)
    for position, symbol, name in zip(positions, symbols, names):
        if fractional:
            position = np.dot(position, cell).tolist()
        structure.append_atom(position=position, symbols=symbol, name=name)

    return structure


def generate_potential() -> LammpsPotentialData:
    """
    Generate the potential to be used in the calculation.

    Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

    :return: potential to do the calculation
    :rtype: LammpsPotentialData
    """

    potential_parameters = {
        "species": ["Fe"],
        "atom_style": "atomic",
        "pair_style": "eam/fs",
        "units": "metal",
        "extra_tags": {
            "content_origin": "NIST IPRP: https: // www.ctcms.nist.gov/potentials/Fe.html",
            "content_other_locations": None,
            "data_method": "unknown",
            "description": """This Fe EAM potential parameter file is from the NIST repository,
            \"Fe_2.eam.fs\" as of the March 9, 2009 update.
            It is similar to \"Fe_mm.eam.fs\" in the LAMMPS distribution dated 2007-06-11,
            but gives different results for very small interatomic distances
            (The LAMMPS potential is in fact the deprecated potential referred to in the March 9,
            2009 update on the NIST repository).
            The file header includes a note from the NIST contributor:
            \"The potential was taken from v9_4_bcc (in C:\\SIMULATION.MD\\Fe\\Results\\ab_initio+Interstitials)\"
            """,
            "developer": ["Ronald E. Miller"],
            "disclaimer": """According to the developer Giovanni Bonny
            (as reported by the NIST IPRP), this potential was not stiffened and cannot
            be used in its present form for collision cascades.
            """,
            "properties": None,
            "publication_year": 2018,
            "source_citations": [
                {
                    "abstract": None,
                    "author": "Mendelev, MI and Han, S and Srolovitz, DJ and Ackland, GJ and Sun, DY and Asta, M",
                    "doi": "10.1080/14786430310001613264",
                    "journal": "{Phil. Mag.}",
                    "number": "{35}",
                    "pages": "{3977-3994}",
                    "recordkey": "MO_546673549085_000a",
                    "recordprimary": "recordprimary",
                    "recordtype": "article",
                    "title": "{Development of new interatomic potentials appropriate for crystalline and liquid iron}",
                    "volume": "{83}",
                    "year": "{2003}",
                }
            ],
            "title": "EAM potential (LAMMPS cubic hermite tabulation) for Fe developed by Mendelev et al. (2003) v000",
        },
    }

    potential = LammpsPotentialData.get_or_create(
        source="Fe_2.eam.fs",
        **potential_parameters,
    )

    return potential


def main(
    settings: orm.Dict,
    parameters: orm.Dict,
    structure: orm.StructureData,
    potential: LammpsPotentialData,
    options: AttributeDict,
    code: orm.Code,
) -> orm.Node:
    """
    Submission of the calculation for an MD run in ``LAMMPS``.

    :param settings: Additional settings that control the ``LAMMPS`` calculation
    :type settings: orm.Dict
    :param parameters: Parameters that control the input script generated for the ``LAMMPS`` calculation
    :type parameters: orm.Dict
    :param structure: structure to be used in the calculation
    :type structure: orm.StructureData
    :param potential: potential to be used in the calculation
    :type potential: LammpsPotentialData
    :param options: options to control the submission parameters
    :type options: AttributeDict
    :param code: code describing the ``LAMMPS`` calculation
    :type code: orm.Code
    :return: node containing the ``LAMMPS`` calculation
    :rtype: orm.Node
    """

    calculation = CalculationFactory("lammps.base")

    builder = calculation.get_builder()
    builder.code = code
    builder.settings = settings
    builder.structure = structure
    builder.parameters = parameters
    builder.potential = potential
    builder.metadata.options = options
    builder.input_restartfile = orm.load_node(13537)

    node = run_get_node(calculation, **builder)

    return node


if __name__ == "__main__":

    STRUCTURE = generate_structure()
    POTENTIAL = generate_potential()
    CODE = orm.load_code("lammps-23.06.2022@localhost")
    OPTIONS = AttributeDict()
    OPTIONS.resources = AttributeDict()
    # Total number of mpi processes
    OPTIONS.resources.num_machines = 1
    OPTIONS.resources.tot_num_mpiprocs = 2
    # Name of the parallel environment
    #    OPTIONS.resources.parallel_env = "mpi"
    # Maximum allowed execution time in seconds
    #    OPTIONS.max_wallclock_seconds = 18000
    # Whether to run in parallel
    #    OPTIONS.withmpi = True
    # Set the slot type for the calculation
    #    OPTIONS.custom_scheduler_commands = "#$ -l slot_type=execute\n#$ -l exclusive=true"

    _parameters = AttributeDict()
    _parameters.control = AttributeDict()
    _parameters.control.units = "metal"
    _parameters.control.timestep = 1e-5
    _parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
    }
    _parameters.md = {
        "integration": {
            "style": "npt",
            "constraints": {
                "temp": [300, 300, 100],
                "iso": [0.0, 0.0, 1000.0],
            },
        },
        "max_number_steps": 5000,
        # "velocity": [{"create": {"temp": 300}, "group": "all"}],
    }
    _parameters.structure = {"atom_style": "atomic"}
    _parameters.potential = {}
    _parameters.thermo = {
        "printing_rate": 100,
        "thermo_printing": {
            "step": True,
            "pe": True,
            "ke": True,
            "press": True,
            "pxx": True,
            "pyy": True,
            "pzz": True,
        },
    }
    _parameters.dump = {"dump_rate": 1000}
    _parameters.restart = {"print_final": True}

    PARAMETERS = orm.Dict(dict=_parameters)

    _settings = AttributeDict()
    _settings.store_restart = True

    SETTINGS = orm.Dict(dict=_settings)

    submission_node = main(
        settings=SETTINGS,
        structure=STRUCTURE,
        potential=POTENTIAL,
        parameters=PARAMETERS,
        options=OPTIONS,
        code=CODE,
    )

    print(f"Calculation node: {submission_node}")
