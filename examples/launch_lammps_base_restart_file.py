"""
Sets up an example for the calculation of bcc Fe using ``aiida-lammps``.
"""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import run_get_node
from aiida.plugins import CalculationFactory
from aiida_lammps.data.potential import LammpsPotentialData
import numpy as np


def generate_structure() -> orm.StructureData:
    """
    Generates the structure for the calculation.

    It will create a bcc structure in a cubic lattice.

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
            "publication_year": 2018,
            "developer": ["Ronald E. Miller"],
            "title": "EAM potential (LAMMPS cubic hermite tabulation) for Fe developed by Mendelev et al. (2003) v000",
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
            "disclaimer": """According to the developer Giovanni Bonny
            (as reported by the NIST IPRP), this potential was not stiffened and cannot
            be used in its present form for collision cascades.
            """,
            "properties": None,
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
        },
    }

    potential = LammpsPotentialData.get_or_create(
        source="Fe_2.eam.fs",
        **potential_parameters,
    )

    return potential


def preapre_restart_calculation(results: dict, parameters: orm.Dict) -> dict:
    """Prepare the parameters so that one can perform a calculation

    :param results: dictionary with the results from the parent calculation
    :type results: dict
    :param parameters: calculation parameters of the parent calculation
    :type parameters: AttributeDict
    :return: modified parameters and the node with the restart information
    :rtype: dict
    """
    _parameters = AttributeDict(parameters.get_dict())

    # The velocity command is removed so that one can see that the first step of the restarted
    # calculation matches with the final one in the previous calculation

    if "velocity" in _parameters.md:
        del _parameters["md"]["velocity"]

    if "restartfile" in results:
        node_restart = results["restartfile"]
        return {"parameters": orm.Dict(_parameters), "input_restartfile": node_restart}

    node_restart = results["remote_folder"]
    return {"parameters": orm.Dict(_parameters), "parent_folder": node_restart}


def main(
    parameters: orm.Dict,
    structure: orm.StructureData,
    potential: LammpsPotentialData,
    options: AttributeDict,
    code: orm.Code,
    settings: orm.Dict,
):
    """
    Submission of the calculation for an MD run in ``LAMMPS``.

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
    :param settings: parameters controlling how ``LAMMPS`` is ran
    :type settings: orm.Dict
    """

    calculation = CalculationFactory("lammps.base")

    builder = calculation.get_builder()
    builder.code = code
    builder.settings = settings
    builder.structure = structure
    builder.parameters = parameters
    builder.potential = potential
    builder.metadata.options = options

    results, node = run_get_node(calculation, **builder)

    print(f"Parent calculation node {node}")

    # Get the restart information from the parent calculation
    restart_data = preapre_restart_calculation(results=results, parameters=parameters)

    # Set the modified parameters so that the calculation can be ran again
    builder.parameters = restart_data["parameters"]
    # If we have stored the restartfile in the database we use this node
    if "input_restartfile" in restart_data:
        builder.input_restartfile = restart_data["input_restartfile"]
    # if we only have the restartfile in the remote folder we use that instead
    if "parent_folder" in restart_data:
        builder.parent_folder = restart_data["parent_folder"]

    # Run the calculation restarting from the previous calculation
    _, node = run_get_node(calculation, **builder)

    print(f"Restart calculation node {node}")


if __name__ == "__main__":
    # Get the structure that will be used in the calculation
    STRUCTURE = generate_structure()
    # Get the potential that will be used in the calculation
    POTENTIAL = generate_potential()
    # Get the lammps code defined in AiiDA database
    CODE = orm.load_code("lammps-23.06.2022@localhost")
    # Define the parameters for the resources requested for the calculation
    OPTIONS = AttributeDict()
    OPTIONS.resources = AttributeDict()
    # Total number of machines used
    OPTIONS.resources.num_machines = 1
    # Total number of mpi processes
    OPTIONS.resources.tot_num_mpiprocs = 2

    # Parameters to control the input file generation
    _parameters = AttributeDict()
    # Control section specifying global simulation parameters
    _parameters.control = AttributeDict()
    # Types of units to be used in the calculation
    _parameters.control.units = "metal"
    # Size of the time step in the units previously defined
    _parameters.control.timestep = 1e-5
    # Set of computes to be evaluated during the calculation
    _parameters.compute = {
        "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
        "stress/atom": [{"type": ["NULL"], "group": "all"}],
        "pressure": [{"type": ["thermo_temp"], "group": "all"}],
    }
    # Set of values to control the behaviour of the molecular dynamics calculation
    _parameters.md = {
        "integration": {
            "style": "npt",
            "constraints": {
                "temp": [300, 300, 100],
                "iso": [0.0, 0.0, 1000.0],
            },
        },
        "max_number_steps": 5000,
        "velocity": [{"create": {"temp": 300, "seed": 1}, "group": "all"}],
    }
    # Control how often the computes are printed to file
    _parameters.dump = {"dump_rate": 1000}
    # Parameters used to pass special information about the structure
    _parameters.structure = {"atom_style": "atomic"}
    # Parameters used to pass special information about the potential
    _parameters.potential = {}
    # Parameters controlling the global values written directly to the output
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
    # Tell lammps to print the final restartfile
    # (THIS DOES NOT STORE IT IN THE DATABASE JUST PRINTS IT)
    _parameters.restart = {"print_final": True}
    # Convert the parameters to an AiiDA data structure
    PARAMETERS = orm.Dict(dict=_parameters)

    # Controlling parameters on how the LAMMPS calculation is performed
    _settings = AttributeDict()
    # Whether or not to store the restart file in the database
    _settings.store_restart = True

    # Store the setting parameters in an AiiDA datastructure
    SETTINGS = orm.Dict(dict=_settings)

    # Run the aiida-lammps calculation
    main(
        structure=STRUCTURE,
        potential=POTENTIAL,
        parameters=PARAMETERS,
        settings=SETTINGS,
        options=OPTIONS,
        code=CODE,
    )
