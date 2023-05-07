"""Set of functions to parse the LAMMPS dump output.
"""
# pylint: disable=fixme
from collections import namedtuple
import re

from aiida import orm
import numpy as np

TrajectoryBlock = namedtuple(
    "TRAJ_BLOCK", ["lines", "timestep", "natoms", "cell", "pbc", "atom_fields"]
)


def _iter_step_lines(file_obj):
    """Parse the lines containing the time step information

    :param file_obj: file object that is being parsed
    :type file_obj: [type]
    :yield: initial line to start the parsing, content of the time step line
    :rtype: int, str
    """
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


def parse_step(lines, initial_line=0) -> namedtuple:
    """Parse a given trajectory step

    :param lines: subset fo the file content to be parsed.
    :type lines: str
    :param initial_line: place where the parsing starts, defaults to 0
    :type initial_line: int, optional
    :raises IOError: if no timestep is found
    :raises IOError: if no number of atoms is found
    :raises IOError: if the box bounds are not found
    :raises IOError: if the header for the atomic positions is not found
    :return: [description]
    :rtype: namedtuple
    """
    # pylint: disable=too-many-locals
    if "ITEM: TIMESTEP" not in lines[0]:
        raise OSError(f"expected line {initial_line} to be TIMESTEP")
    if "ITEM: NUMBER OF ATOMS" not in lines[2]:
        raise OSError(f"expected line {initial_line + 2} to be NUMBER OF ATOMS")
    if "ITEM: BOX BOUNDS xy xz yz" not in lines[4]:
        raise OSError(f"expected line {initial_line + 4} to be BOX BOUNDS xy xz yz")
        # TODO handle case when xy xz yz not present -> orthogonal box
    if "ITEM: ATOMS" not in lines[8]:
        raise OSError(f"expected line {initial_line + 8} to be ATOMS")
    timestep = int(lines[1])
    number_of_atoms = int(lines[3])

    # each pbc contains two letters <lo><hi> such that:
    # p = periodic, f = fixed, s = shrink wrap, m = shrink wrapped with a minimum value
    pbc = lines[4].split()[6:]

    bounds = [line.split() for line in lines[5:8]]
    bounds = np.array(bounds, dtype=float)
    if bounds.shape[1] == 2:
        bounds = np.append(bounds, np.array([0, 0, 0])[None].T, axis=1)

    box_xy = bounds[0, 2]
    box_xz = bounds[1, 2]
    box_yz = bounds[2, 2]

    xlo = bounds[0, 0] - np.min([0.0, box_xy, box_xz, box_xy + box_xz])
    xhi = bounds[0, 1] - np.max([0.0, box_xy, box_xz, box_xy + box_xz])
    ylo = bounds[1, 0] - np.min([0.0, box_yz])
    yhi = bounds[1, 1] - np.max([0.0, box_yz])
    zlo = bounds[2, 0]
    zhi = bounds[2, 1]

    super_cell = np.array(
        [
            [xhi - xlo, box_xy, box_xz],
            [0.0, yhi - ylo, box_yz],
            [0.0, 0.0, zhi - zlo],
        ]
    )
    cell = super_cell.T
    field_names = [
        re.sub("[^a-zA-Z0-9_]", "__", entry) for entry in lines[8].split()[2:]
    ]
    fields = []
    for i in range(number_of_atoms):
        fields.append(lines[9 + i].split())
    atom_fields = {n: v.tolist() for n, v in zip(field_names, np.array(fields).T)}

    return TrajectoryBlock(
        lines,
        timestep,
        number_of_atoms,
        cell,
        pbc,
        atom_fields,
    )


def iter_trajectories(file_obj):
    """Parse a LAMMPS Trajectory file, yielding data for each time step."""
    for line_num, lines in _iter_step_lines(file_obj):
        yield parse_step(lines, line_num)


def create_structure(
    trajectory_block: namedtuple,
    symbol_field: str = "element",
    position_fields: tuple = ("x", "y", "z"),
    original_structure: orm.StructureData = None,
) -> orm.StructureData:
    """Generate a structure from the atomic positions at a given step.

    :param trajectory_block: block with the trajectory information
    :type trajectory_block: namedtuple
    :param symbol_field: field name where the element symbols are found,
        defaults to 'element'
    :type symbol_field: str, optional
    :param position_fields: name of the files where the positions are found,
        defaults to ('x', 'y', 'z')
    :type position_fields: tuple, optional
    :param original_structure: original structure of the calculation, defaults to None
    :type original_structure: orm.StructureData, optional
    :raises ValueError: if the symbols of the structure and of the trajectory
        info differ
    :raises NotImplementedError: If the boundary conditions are not periodic or free
    :return: structure of the current time step
    :rtype: orm.StructureData
    """
    symbols = trajectory_block.atom_fields[symbol_field]
    positions = np.array(
        [trajectory_block.atom_fields[f] for f in position_fields], dtype=float
    ).T

    if original_structure is not None:
        kind_names = original_structure.get_site_kindnames()
        kind_symbols = [original_structure.get_kind(n).symbol for n in kind_names]
        if symbols != kind_symbols:
            raise ValueError(
                f"original_structure has different symbols:: {kind_symbols} != {symbols}"
            )
        structure = original_structure.clone()
        structure.reset_cell(trajectory_block.cell)
        structure.reset_sites_positions(positions)

        return structure

    boundary_conditions = []
    for pbc in trajectory_block.pbc:
        if pbc == "pp":
            boundary_conditions.append(True)
        elif pbc == "ff":
            boundary_conditions.append(False)
        else:
            raise NotImplementedError(f"pbc = {trajectory_block.pbc}")

    structure = orm.StructureData(
        cell=trajectory_block.cell,
        pbc=boundary_conditions,
    )
    for symbol, position in zip(symbols, positions):
        structure.append_atom(position=position, symbols=symbol)

    return structure
