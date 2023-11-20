#!/usr/bin/env runaiida
import io

from aiida.engine import run
from aiida.orm import Dict, StructureData, load_code
from ase.build import bulk
import requests

from aiida_lammps.data.potential import LammpsPotentialData

# Load the code configured for ``lmp``. Make sure to replace
# this string with the label used in the code setup.
code = load_code("lammps@localhost")
builder = code.get_builder()

structure = StructureData(ase=bulk("Fe", "bcc", 2.87, cubic=True))
builder.structure = structure

# Download the potential from the repository and store it as a BytesIO object
_stream = io.BytesIO(
    requests.get(
        "https://openkim.org/files/MO_546673549085_000/Fe_2.eam.fs", timeout=20
    ).text.encode("ascii")
)

# Set the metadata for the potential
potential_parameters = {
    "species": ["Fe"],
    "atom_style": "atomic",
    "pair_style": "eam/fs",
    "units": "metal",
    "extra_tags": {
        "title": "EAM potential (LAMMPS cubic hermite tabulation) for Fe developed by Mendelev et al. (2003) v000",
        "content_origin": "NIST IPRP: https: // www.ctcms.nist.gov/potentials/Fe.html",
        "developer": ["Ronald E. Miller"],
        "publication_year": 2018,
    },
}

# Store the potential in an AiiDA node
potential = LammpsPotentialData.get_or_create(source=_stream, **potential_parameters)

builder.potential = potential

parameters = Dict(
    {
        "control": {"units": "metal", "timestep": 1e-5},
        "compute": {
            "pe/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
            "ke/atom": [{"type": [{"keyword": " ", "value": " "}], "group": "all"}],
            "stress/atom": [{"type": ["NULL"], "group": "all"}],
            "pressure": [{"type": ["thermo_temp"], "group": "all"}],
        },
        "structure": {"atom_style": "atomic"},
        "thermo": {
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
        },
        "md":{
            "integration": {
                "style": "npt",
                "constraints": {
                    "temp": [300, 300, 100],
                    "iso": [0.0, 0.0, 1000.0],
                },
            },
            "max_number_steps": 5000,
            "velocity": [{"create": {"temp": 300}, "group": "all"}],
        },
    }
)
builder.parameters = parameters

builder.metadata.options = {
    "resources": {
        "num_machines": 1,
    },
    "max_wallclock_seconds": 1800,
    "withmpi": False,
}

results, node = run.get_node(builder)

print(
    f"Calculation: {node.process_class}<{node.pk}> {node.process_state.value} [{node.exit_status}]"
)
print(f"Results: {results}")
assert node.is_finished_ok, f"{node} failed: [{node.exit_status}] {node.exit_message}"
