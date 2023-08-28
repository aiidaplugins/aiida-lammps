"""Workflow for a molecular dynamics simulation in LAMMPS."""
import os
from typing import Union

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, append_

from aiida_lammps.validation.utils import validate_against_schema
from aiida_lammps.workflows.base import LammpsBaseWorkChain


class LammpsMDWorkChain(WorkChain):
    """Workchain to perform a LAMMPS MD simulation."""

    @classmethod
    def define(cls, spec):
        """Define the process specification"""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(LammpsBaseWorkChain, exclude=('parameters'))
        spec.input(
            "lammps.parameters",
            valid_type=orm.Dict,
            help="""
            Parameters that control the input script generated for the ``LAMMPS`` calculation
            """,
        )
        spec.input(
            "md.steps",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(1000),
            help="Number of steps in the MD simulation",
        )
        spec.input(
            'md.algo',
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str('verlet'),
            validator=cls._validate_md_algorithms,
            help="Type of time integrator used for MD simulations in LAMMPS (``run_style``)",
        )
        spec.input(
            'md.integrator',
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str("npt"),
            help="Type of thermostat used for the MD simulation in LAMMPS, e.g. ``fix npt``"
        )
        spec.input(
            'md.integrator_constraints',
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict({"temp":[300,300,100], "iso":[0.0, 0.0, 1000]}),
            help="""Set of constraints that are applied to the thermostat"""
        )
        spec.input(
            'md.velocity',
            valid_type=orm.List,
            required=False,
            help="""
            List with the information describing how to generate the velocities for the
            initialization of the MD run
            """
        )
        spec.input(
            'md.respa_options',
            valid_type=orm.List,
            required=False,
            help="""
            List with the information needed to setup the respa options
            """
        )
        spec.inputs.validator = cls._validate_inputs
        spec.outline(
            cls.setup,
            cls.run_md,
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
    def _validate_md_algorithms(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument,inconsistent-return-statements
        """Validate that the given algorithm for the MD run is supported"""
        _algo = value.value
        _supported_algorithms = ["verlet", "verlet/split", "respa"]
        if _algo not in _supported_algorithms:
            return f"Invalid/unsupported relaxation method, {_algo} not in {_supported_algorithms}"

    @classmethod
    def _validate_inputs(cls, value, ctx) -> Union[str, None]:
        # pylint: disable=unused-argument
        """Validate that the given inputs are proper for a MD run"""
        parameters = value["lammps"]["parameters"].get_dict()
        parameters["md"] = {
            "integration": {
                "run_style": value["md"]["algo"].value,
                "style": value["md"]["integrator"].value,
                "constraints": value["md"]["integrator_constraints"].get_dict(),
            }
        }
        if "velocity" in value["md"]:
            parameters["md"].update({"velocity": value["md"]["velocity"].get_list()})
        if "respa_options" in value["md"]:
            parameters["md"].update(
                {"respa_options": value["md"]["respa_options"].get_list()}
            )

        if "minimize" in parameters:
            # Set a dummy value just so that the validation passes, the real parameters will
            # be filled later
            del parameters["minimize"]

        _file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "validation/schemas/lammps_schema.json",
        )

        validate_against_schema(data=parameters, filename=_file)

    def setup(self):
        """Setting up the context for the calculation"""
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(LammpsBaseWorkChain, namespace="lammps")
        )

        self.ctx.inputs.lammps.parameters = self.inputs.lammps.parameters.get_dict()

        # Remove any entry referring to possible minimization parameters
        if "minimize" in self.ctx.inputs.lammps.parameters:
            self.logger.warning(
                "Parameters for a 'minimize' simulation were found, removing them from the input"
            )
            del self.ctx.inputs.lammps.parameters["minimize"]

        if "md" in self.ctx.inputs.lammps.parameters:
            self.logger.warning(
                "Entry for 'md' was found in the ``parameters`` "
                "overriding with the values in the inputs"
            )
        self.ctx.inputs.lammps.parameters["md"] = self._generate_md_block()

    def _generate_md_block(self) -> AttributeDict:
        """Generate the md block for the parameters"""
        md_params = AttributeDict()
        md_params.integration = AttributeDict()
        md_params.integration.run_style = self.inputs.md.algo.value
        md_params.integration.style = self.inputs.md.integrator.value
        md_params.integration.constraints = (
            self.inputs.md.integrator_constraints.get_dict()
        )

        if "velocity" in self.inputs.md:
            md_params.update({"velocity": self.inputs.md.velocity.get_list()})
        if "respa_options" in self.inputs.md:
            md_params.update({"respa_options": self.inputs.md.respa_options.get_list()})
        return md_params

    def run_md(self):
        """Run the `LammpsBaseWorkChain` to run a md `LammpsBaseCalculation`"""

        inputs = self.ctx.inputs
        inputs.lammps.parameters = orm.Dict(inputs.lammps.parameters)

        workchain = self.submit(LammpsBaseWorkChain, **inputs)
        self.report(f"Launching LammpsBaseWorkChain<{workchain.pk}>")
        return ToContext(workchains=append_(workchain))

    def results(self):
        """Attach the output parameters of the last workchain to the outputs."""

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

        self.out_many(self.exposed_outputs(workchain, LammpsBaseWorkChain))
