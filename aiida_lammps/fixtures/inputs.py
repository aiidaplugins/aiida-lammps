"""Set of data for the testing of the input generation"""
from aiida.common import AttributeDict
import pytest


@pytest.fixture
def restart_data() -> AttributeDict:
    """Get the reference data for the restart information

    :return: reference data for the restart information
    :rtype: AttributeDict
    """
    data = AttributeDict()

    data.final = (
        "#---------------------Start of the write restart information---------------------#\n"
        "write_restart restart.aiida\n"
        "#----------------------End of the write restart information----------------------#\n"
    )
    data.intermediate = (
        "#--------------Start of the intermediate write restart information---------------#\n"
        "restart 100 restart.aiida\n"
        "#---------------End of the intermediate write restart information----------------#\n"
    )
    return data


@pytest.fixture
def parameters_md() -> AttributeDict:
    """Get the parameters to generate the md input parameter for calculation

    :return: set of parameters fo the md calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.compute = AttributeDict()
    parameters.compute["ke/atom"] = [
        {"group": "all", "type": [{"keyword": " ", "value": " "}]}
    ]
    parameters.compute["pe/atom"] = [
        {"group": "all", "type": [{"keyword": " ", "value": " "}]}
    ]
    parameters.compute.pressure = [{"group": "all", "type": ["thermo_temp"]}]
    parameters.compute["stress/atom"] = [{"group": "all", "type": ["NULL"]}]
    parameters.control = AttributeDict()
    parameters.control.timestep = 1.0e-05
    parameters.control.units = "metal"
    parameters.dump = AttributeDict()
    parameters.dump.dump_rate = 1000
    parameters.fix = AttributeDict()
    parameters.fix["box/relax"] = [
        {"group": "all", "type": ["iso", 0.0, "vmax", 0.001]}
    ]
    parameters.md = AttributeDict()
    parameters.md.integration = AttributeDict()
    parameters.md.integration.constraints = AttributeDict()
    parameters.md.integration.constraints.iso = [0.0, 0.0, 1000.0]
    parameters.md.integration.constraints.temp = [300, 300, 100]
    parameters.md.integration.style = "npt"
    parameters.potential = AttributeDict()
    parameters.structure = AttributeDict()
    parameters.structure.atom_style = "atomic"
    parameters.thermo = AttributeDict()
    parameters.thermo.printing_rate = 100
    parameters.thermo.thermo_printing = AttributeDict()
    parameters.thermo.thermo_printing.ke = True
    parameters.thermo.thermo_printing.pe = True
    parameters.thermo.thermo_printing.press = True
    parameters.thermo.thermo_printing.pxx = True
    parameters.thermo.thermo_printing.pyy = True
    parameters.thermo.thermo_printing.pzz = True
    parameters.thermo.thermo_printing.step = True
    parameters.restart = AttributeDict()
    parameters.restart.print_final = True

    return parameters


@pytest.fixture
def parameters_minimize() -> AttributeDict:
    """Get the parameters to generate the minimize input parameter for calculation

    :return: set of parameters fo the minimization calculation
    :rtype: AttributeDict
    """
    parameters = AttributeDict()
    parameters.compute = AttributeDict()
    parameters.compute["ke/atom"] = [
        {"group": "all", "type": [{"keyword": " ", "value": " "}]}
    ]
    parameters.compute["pe/atom"] = [
        {"group": "all", "type": [{"keyword": " ", "value": " "}]}
    ]
    parameters.compute.pressure = [{"group": "all", "type": ["thermo_temp"]}]
    parameters.compute["stress/atom"] = [{"group": "all", "type": ["NULL"]}]
    parameters.control = AttributeDict()
    parameters.control.timestep = 1.0e-05
    parameters.control.units = "metal"
    parameters.dump = AttributeDict()
    parameters.dump.dump_rate = 1000
    parameters.fix = AttributeDict()
    parameters.fix["box/relax"] = [
        {"group": "all", "type": ["iso", 0.0, "vmax", 0.001]}
    ]
    parameters.minimize = AttributeDict()
    parameters.minimize.energy_tolerance = 0.0001
    parameters.minimize.force_tolerance = 1.0e-05
    parameters.minimize.max_evaluations = 5000
    parameters.minimize.max_iterations = 5000
    parameters.minimize.style = "cg"
    parameters.potential = AttributeDict()
    parameters.structure = AttributeDict()
    parameters.structure.atom_style = "atomic"
    parameters.thermo = AttributeDict()
    parameters.thermo.printing_rate = 100
    parameters.thermo.thermo_printing = AttributeDict()
    parameters.thermo.thermo_printing.ke = True
    parameters.thermo.thermo_printing.pe = True
    parameters.thermo.thermo_printing.press = True
    parameters.thermo.thermo_printing.pxx = True
    parameters.thermo.thermo_printing.pyy = True
    parameters.thermo.thermo_printing.pzz = True
    parameters.thermo.thermo_printing.step = True
    parameters.restart = AttributeDict()
    parameters.restart.print_final = True

    return parameters
