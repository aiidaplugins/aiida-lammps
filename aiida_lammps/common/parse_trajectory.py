from collections import namedtuple

from aiida.orm import StructureData
import numpy as np

TRAJ_BLOCK = namedtuple(
    "TRAJ_BLOCK", ["lines", "timestep", "natoms", "cell", "pbc", "atom_fields"]
)


def iter_step_lines(file_obj):
    step_content = None
    init_line = 0
    for i, line in enumerate(file_obj):
        if "ITEM: TIMESTEP" in line:
            if step_content is not None:
                yield init_line, step_content
            init_line = i + 1
            step_content = []
        if step_content is not None:
            step_content.append(line.strip())
    if step_content is not None:
        yield init_line, step_content


def parse_step(lines, intial_line=0):
    if "ITEM: TIMESTEP" not in lines[0]:
        raise IOError("expected line {} to be TIMESTEP".format(intial_line))
    if "ITEM: NUMBER OF ATOMS" not in lines[2]:
        raise IOError("expected line {} to be NUMBER OF ATOMS".format(intial_line + 2))
    if "ITEM: BOX BOUNDS xy xz yz" not in lines[4]:
        raise IOError(
            "expected line {} to be BOX BOUNDS xy xz yz".format(intial_line + 4)
        )
        # TODO handle case when xy xz yz not present -> orthogonal box
    if "ITEM: ATOMS" not in lines[8]:
        raise IOError("expected line {} to be ATOMS".format(intial_line + 8))
    timestep = int(lines[1])
    number_of_atoms = int(lines[3])

    # each pbc contains two letters <lo><hi> such that:
    # p = periodic, f = fixed, s = shrink wrap, m = shrink wrapped with a minimum value
    pbc = lines[4].split()[6:]

    bounds = [line.split() for line in lines[5:8]]
    bounds = np.array(bounds, dtype=float)
    if bounds.shape[1] == 2:
        bounds = np.append(bounds, np.array([0, 0, 0])[None].T, axis=1)

    xy = bounds[0, 2]
    xz = bounds[1, 2]
    yz = bounds[2, 2]

    xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
    xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
    ylo = bounds[1, 0] - np.min([0.0, yz])
    yhi = bounds[1, 1] - np.max([0.0, yz])
    zlo = bounds[2, 0]
    zhi = bounds[2, 1]

    super_cell = np.array([[xhi - xlo, xy, xz], [0, yhi - ylo, yz], [0, 0, zhi - zlo]])
    cell = super_cell.T
    field_names = lines[8].split()[2:]
    fields = []
    for i in range(number_of_atoms):
        fields.append(lines[9 + i].split())
    atom_fields = {n: v.tolist() for n, v in zip(field_names, np.array(fields).T)}

    return TRAJ_BLOCK(lines, timestep, number_of_atoms, cell, pbc, atom_fields)


def iter_trajectories(file_obj):
    """Parse a LAMMPS Trajectory file, yielding data for each time step."""
    for line_num, lines in iter_step_lines(file_obj):
        yield parse_step(lines, line_num)


def create_structure(
    traj_block,
    symbol_field="element",
    position_fields=("x", "y", "z"),
    original_structure=None,
):
    symbols = traj_block.atom_fields[symbol_field]
    positions = np.array(
        [traj_block.atom_fields[f] for f in position_fields], dtype=float
    ).T

    if original_structure is not None:
        kind_names = original_structure.get_site_kindnames()
        kind_symbols = [original_structure.get_kind(n).symbol for n in kind_names]
        if symbols != kind_symbols:
            raise ValueError(
                "original_structure has different symbols:: {} != {}".format(
                    kind_symbols, symbols
                )
            )
        structure = original_structure.clone()
        structure.reset_cell(traj_block.cell)
        structure.reset_sites_positions(positions)

        return structure

    pbcs = []
    for pbc in traj_block.pbc:
        if pbc == "pp":
            pbcs.append(True)
        elif pbc == "ff":
            pbcs.append(False)
        else:
            raise NotImplementedError("pbc = {}".format(traj_block.pbc))

    structure = StructureData(cell=traj_block.cell, pbc=pbcs)
    for symbol, position in zip(symbols, positions):
        structure.append_atom(position=position, symbols=symbol)

    return structure
