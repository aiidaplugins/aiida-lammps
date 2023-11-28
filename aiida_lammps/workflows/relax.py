"""Workflow for the relaxation of a structure using the minimization procedure in LAMMPS."""
from itertools import groupby
import os
from typing import Union

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.exceptions import NotExistent
from aiida.engine import ToContext, WorkChain, append_, while_

from aiida_lammps.validation.utils import validate_against_schema
from aiida_lammps.workflows.base import LammpsBaseWorkChain


class LammpsRelaxWorkChain(WorkChain):
    """Workchain to relax a structure using the LAMMPS minimization procedure."""

    @classmethod
    def define(cls, spec):
        """Define the process specification"""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(LammpsBaseWorkChain, exclude=('parameters'))
        spec.input(
            "lammps.parameters",
            valid_type=orm.Dict,
            validator=cls._validate_parameters,
            help="""
            Parameters that control the input script generated for the ``LAMMPS`` calculation
            """,
        )
        spec.input(
            "relax.algo",
            required=False,
            valid_type=orm.Str,
            default=lambda: orm.Str("cg"),
            validator=cls._validate_relaxation_algorithms,
            help="""
            The algorithm to be used during relaxation.
            """,
        )
        spec.input(
            "relax.volume",
            required=False,
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="""
            Whether or not relaxation of the volume will be performed by using the ``box/relax``
            fix from LAMMPS.
            """,
        )
        spec.input(
            "relax.shape",
            required=False,
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="""
            Whether or not the shape of the cell will be relaxed by using the ``box/relax``
            fix from LAMMPS.
            """,
        )
        spec.input(
            "relax.positions",
            required=False,
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="""
            Whether or not to allow the relaxation of the atomic positions.
            """,
        )
        spec.input(
            "relax.steps",
            required=False,
            valid_type=orm.Int,
            default=lambda: orm.Int(1000),
            help="""
            Maximum number of steps during the relaxation.
            """,
        )
        spec.input(
            "relax.evaluations",
            required=False,
            valid_type=orm.Int,
            default=lambda: orm.Int(10000),
            help="""
            Maximum number of force/energy evaluations during the relaxation.
            """,
        )
        spec.input(
            "relax.energy_tolerance",
            required=False,
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-4),
            help="""
            The tolerance that determined whether the relaxation procedure is stopped. In this case
            it stops when the relative change between outer iterations of the relaxation run is
            less than the given value.
            """,
        )
        spec.input(
            "relax.force_tolerance",
            required=False,
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-4),
            help="""
            The tolerance that determines whether the relaxation procedure is stopped. In this case
            it stops when the 2-norm of the global force vector is less than the given value.
            """,
        )
        spec.input(
            "relax.target_pressure",
            required=False,
            valid_type=orm.Dict,
            validator=cls._validate_pressure_dictionary,
            help="""
            Dictionary containing the values for the target pressure tensor.
            """,
        )
        spec.input(
            "relax.max_volume_change",
            required=False,
            valid_type=orm.Float,
            help="""
            Maximum allowed change in one iteration (``vmax``)
            """,
        )
        spec.input(
            "relax.nreset",
            required=False,
            valid_type=orm.Int,
            help="""
            Reset the reference cell every this many minimizer iterations
            """,
        )
        spec.input(
            'relax.meta_convergence',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="""
            If `True` the workchain will perform a meta-convergence on the cell volume.
            """
        )
        spec.input(
            'relax.max_meta_convergence_iterations',
            valid_type=orm.Int,
            default=lambda: orm.Int(5),
            help="""
            The maximum number of variable cell relax iterations in the meta convergence cycle.
            """
        )
        spec.input(
            'relax.volume_convergence',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.01),
            help="""
            The volume difference threshold between two consecutive meta convergence iterations.
            """
        )
        spec.inputs.validator = cls.validate_inputs
        spec.outline(
            cls.setup,
            while_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.results,
        )
        spec.expose_outputs(LammpsBaseWorkChain)
        spec.exit_code(
            403,
            'ERROR_SUB_PROCESS_FAILED',
            message="The underlying LammpsBaseWorkChain failed",
        )
        # yapf: enable

    @classmethod
    def _validate_parameters(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument,inconsistent-return-statements
        """
        Validate the input parameters and compares them against a schema.

        Takes the input parameters dictionaries that will be used to generate the
        LAMMPS input parameter and will be checked against a schema for validation.
        """

        parameters = value.get_dict()
        if not any(key in parameters for key in ["md", "minimize"]):
            # Set a dummy value just so that the validation passes, the real parameters will
            # be filled later
            parameters["minimize"] = {}

        _file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "validation/schemas/lammps_schema.json",
        )

        validate_against_schema(data=parameters, filename=_file)

    @classmethod
    def _validate_relaxation_algorithms(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument,inconsistent-return-statements
        """Validate the algorithm used for the relaxation of the structure"""
        _algo = value.value

        _supported_algorithms = ["cg", "htfn", "sd", "quickmin", "fire"]

        if _algo not in _supported_algorithms:
            return f"Invalid/unsupported relaxation method, {_algo} not in {_supported_algorithms}"

    @classmethod
    def _validate_pressure_dictionary(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument,inconsistent-return-statements
        """Validate that the pressure dictionary does not have entries that are not permitted"""
        _valid_entries = ["x", "y", "z", "xy", "xz", "yz"]

        if not all(key in _valid_entries for key in value.get_dict()):
            return (
                f"The pressure dictionary {value.get_dict()} contains unexpected "
                f"entries not matching {_valid_entries}"
            )
        if not all(
            isinstance(_value, (float, int)) for _value in value.get_dict().values()
        ):
            return (
                f"The pressure dictionary {value.get_dict()} contains values that are not of "
                "type (float, int)"
            )

    @classmethod
    def validate_inputs(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument,inconsistent-return-statements
        """Validate the global inputs of the calculation"""

        def _all_equal(iterable):
            """Check if all the entries for an iterable are equal"""
            _group = groupby(iterable)
            return next(_group, True) and not next(_group, False)

        if value["relax"]["volume"].value and "target_pressure" not in value["relax"]:
            return "When relaxing the cell the ``target_pressure`` must be given"

        if (
            value["relax"]["volume"].value
            and value["relax"]["shape"].value
            and "target_pressure" not in value["relax"]
        ):
            return (
                "When relaxing the shape and the volume at the same time one needs to give the "
                "``target_pressure`` tensor as a dictionary in the form "
                "'{x:pxx, y:pyy, z:pzz, xy:pxy, xz:pxz, yz: pyz }' "
            )
        if (
            value["relax"]["volume"].value
            and not value["relax"]["shape"].value
            and "target_pressure" in value["relax"]
            and not _all_equal(value["relax"]["target_pressure"].get_dict().values())
        ):
            return (
                "Requesting a volume relaxation without shape optimization, the values of "
                "``target_pressure`` should all be equal or be just one value, instead "
                f"got {value.relax.target_pressure.get_dict()}"
            )

        if value["relax"]["shape"].value and not value["relax"]["volume"].value:
            return "Cannot vary only the shape while keeping the shape constant."

        if (
            "nreset" in value["relax"]
            and value["relax"]["nreset"].value > value["relax"]["steps"].value
        ):
            return (
                "Requesting that the reference cell is reset a number of steps: "
                f"{value['relax']['nreset'].value} larger than the number of steps of the "
                f"simulation: {value['relax']['steps'].value}"
            )

    def setup(self):
        """Setting up the context for the calculation"""
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(LammpsBaseWorkChain, namespace="lammps")
        )

        self.ctx.inputs.lammps.parameters = self.inputs.lammps.parameters.get_dict()

        self.ctx.current_structure = self.inputs.lammps.structure
        self.ctx.current_cell_volume = None
        self.ctx.iteration = 0
        self.ctx.is_converged = False
        self.ctx.meta_convergence = self.inputs.relax.meta_convergence.value

        if self.ctx.meta_convergence and not self.inputs.relax.volume.value:
            self.report(
                "The volume of the cell cannot change. Turning the meta convergence off"
            )
            self.ctx.meta_convergence = False

        # Remove any entry referring to possible molecular dynamics parameters
        if "md" in self.ctx.inputs.lammps.parameters:
            self.logger.warning(
                "Parameters for an 'md' simulation were found, removing them from the input"
            )
            del self.ctx.inputs.lammps.parameters["md"]

        # If the user has passed information about the fix box/relax remove them
        if (
            "fix" in self.ctx.inputs.lammps.parameters
            and "box/relax" in self.ctx.inputs.lammps.parameters["fix"]
        ):
            self.logger.warning(
                "Overriding 'fix box/relax' in the ``parameters`` with the values "
                "used in the inputs"
            )
            del self.ctx.inputs.lammps.parameters["fix"]["box/relax"]

        # Check if the volume is allowed to change, then apply the fix box/relax
        # This is only called if the volume is allowed to change, since one cannot vary only the
        # shape without varying the volume
        if self.inputs.relax.volume.value:
            self._update_fix_parameters("box/relax", self._generate_fix_box_relax())

        if not self.inputs.relax.positions.value:
            self._update_fix_parameters(
                "setforce", [{"group": "all", "type": [0.0, 0.0, 0.0]}]
            )

        if "minimize" in self.ctx.inputs.lammps.parameters:
            self.logger.warning(
                "Entry for 'minimize' was found in the ``parameters`` "
                "overriding with the values in the inputs"
            )

        self.ctx.inputs.lammps.parameters["minimize"] = self._generate_minimize_block()

    def _generate_minimize_block(self) -> AttributeDict:
        """Generate the minimization block for the parameters"""
        minimize = AttributeDict()
        minimize.style = self.inputs.relax.algo.value
        minimize.force_tolerance = self.inputs.relax.force_tolerance.value
        minimize.energy_tolerance = self.inputs.relax.energy_tolerance.value
        minimize.max_iterations = self.inputs.relax.steps.value
        minimize.max_evaluations = self.inputs.relax.evaluations.value

        return minimize

    def _generate_fix_box_relax(self) -> list:
        """Generate the parameters needed for the fix box/relax depending on the inputs given.

        :return: list with the information about fix the box/relax
        :rtype: list
        """
        _box_fix_dict = {"group": "all", "type": []}
        # If only the volume is relaxed
        if self.inputs.relax.volume.value and not self.inputs.relax.shape.value:
            _pressure = list(self.inputs.relax.target_pressure.get_dict().values())[-1]
            _box_fix_dict["type"].append("iso")
            _box_fix_dict["type"].append(_pressure)
        # If volume and shape are relaxed
        if self.inputs.relax.volume.value and self.inputs.relax.shape.value:
            for key, value in self.inputs.relax.target_pressure.get_dict().items():
                _box_fix_dict["type"].append(key)
                _box_fix_dict["type"].append(value)
        # If one wants to restrict how much the volume can change in each iteration
        if "max_volume_change" in self.inputs.relax:
            _box_fix_dict["type"].append("vmax")
            _box_fix_dict["type"].append(self.inputs.relax.max_volume_change.value)
        # If one wants to set when the reference cell is reset
        if "nreset" in self.inputs.relax:
            _box_fix_dict["type"].append("nreset")
            _box_fix_dict["type"].append(self.inputs.relax.nreset.value)
        return [_box_fix_dict]

    def _update_fix_parameters(self, key: str, value: list):
        """Update the fix dictionary to take into account the cases in which it might not exits

        :param key: type of fix to be added
        :type key: str
        :param value: list containing the fix parameters
        :type value: list
        """
        if "fix" not in self.ctx.inputs.lammps.parameters:
            self.ctx.inputs.lammps.parameters["fix"] = {}
        self.ctx.inputs.lammps.parameters["fix"][key] = value

    def should_run_relax(self):
        """Return whether a relaxation workchain should be run"""
        return (
            not self.ctx.is_converged
            and self.ctx.iteration
            < self.inputs.relax.max_meta_convergence_iterations.value
        )

    def run_relax(self):
        """Run the `LammpsBaseWorkChain` to run a relax `LammpsBaseCalculation`"""
        self.ctx.iteration += 1
        inputs = self.ctx.inputs
        inputs.lammps.structure = self.ctx.current_structure
        inputs.lammps.parameters = orm.Dict(inputs.lammps.parameters)

        inputs.lammps.metadata.call_link_label = f"iteration_{self.ctx.iteration:02d}"

        workchain = self.submit(LammpsBaseWorkChain, **inputs)
        self.report(f"Launching LammpsBaseWorkChain<{workchain.pk}>")
        return ToContext(workchains=append_(workchain))

    def inspect_relax(self):
        """Check the current state of the relaxation"""
        workchain = self.ctx.workchains[-1]

        if workchain.is_excepted or workchain.is_killed:
            self.report(
                f"The underlying LammpsBaseWorkChain<{workchain.pk}> was excepted or killed"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED  # pylint: disable=no-member

        if workchain.is_failed:
            self.report(
                f"The underlying LammpsBaseWorkChain<{workchain.pk}> failed with "
                f"exit status {workchain.exit_status}"
            )

        try:
            structure = workchain.outputs.structure
        except NotExistent:
            self.report(
                f"The underlying LammpsBaseWorkChain<{workchain.pk}> did not produce as structure"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED  # pylint: disable=no-member

        prev_cell_volume = self.ctx.current_cell_volume
        curr_cell_volume = structure.get_cell_volume()

        self.ctx.current_structure = structure

        self.report(
            f"After iteration {self.ctx.iteration} the cell volume of the relaxed structure "
            f"is {curr_cell_volume:.4e}"
        )

        # After first iteration, simply set the cell volume and restart the next base workchain
        if not prev_cell_volume:
            self.ctx.current_cell_volume = curr_cell_volume

            # If meta convergence is switched off we are done
            if not self.ctx.meta_convergence:
                self.ctx.is_converged = True
            return

        volume_tolerance = self.inputs.relax.volume_convergence.value
        volume_relative_difference = (
            abs(prev_cell_volume - curr_cell_volume) / prev_cell_volume
        )

        if volume_relative_difference < volume_tolerance:
            self.ctx.is_converged = True
            self.report(
                f"The relative volume relative difference {volume_relative_difference:.4e} "
                f"smaller than the tolerance {volume_tolerance:.4e}"
            )
        else:
            self.report(
                "The current relative cell volume relative difference "
                f"{volume_relative_difference:.4e} is larger than the "
                f"tolerance {volume_tolerance:.4e}"
            )

        self.ctx.current_cell_volume = curr_cell_volume
        return

    def results(self):
        """Attach the output parameters and structure of the last workchain to the outputs."""

        if (
            self.ctx.is_converged
            and self.ctx.iteration
            <= self.inputs.relax.max_meta_convergence_iterations.value
        ):
            self.report(f"Workchain completed after {self.ctx.iteration} iterations")
        else:
            self.report("Maximum number of meta convergence iterations exceeded")

        final_relax_workchain = self.ctx.workchains[-1]

        self.out_many(self.exposed_outputs(final_relax_workchain, LammpsBaseWorkChain))
