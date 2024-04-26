"""Workchain to run a LAMMPS calculation with automated error handling and restarts."""
import re

from aiida import orm
from aiida.common import AttributeDict, NotExistentAttributeError
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    process_handler,
    while_,
)

from aiida_lammps.calculations.base import LammpsBaseCalculation
from aiida_lammps.utils import RestartTypes


class LammpsBaseWorkChain(BaseRestartWorkChain):
    """Base workchain for calculations using LAMMPS"""

    _process_class = LammpsBaseCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(LammpsBaseCalculation, namespace='lammps')
        spec.input(
            "store_restart",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="""
            Whether to store the restartfile in the repository.
            """,
        )
        spec.expose_outputs(LammpsBaseCalculation)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.exit_code(
            300,
            "ERROR_UNRECOVERABLE_FAILURE",
            message="""
            The calculation failed with an unidentified unrecoverable error.
            """,
        )
        # yapf: enable

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.

        This ``self.ctx.inputs`` dictionary will be used by the ``BaseRestartWorkChain`` to submit the calculations
        in the internal loop.

        The ``parameters`` and ``settings`` input ``Dict`` nodes are converted into a regular dictionary and the
        default namelists for the ``parameters`` are set to empty dictionaries if not specified.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(LammpsBaseCalculation, "lammps")
        )
        self.ctx.inputs.settings = (
            self.ctx.inputs.settings.get_dict()
            if "settings" in self.ctx.inputs
            else AttributeDict({})
        )
        if "store_restart" in self.ctx.inputs:
            self.ctx.inputs.settings.store_restart = self.ctx.inputs.store_restart.value

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition
        is met and an action was taken.
        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        self.report(
            f"{calculation.process_label}<{calculation.pk}> failed "
            f"with exit status {calculation.exit_status}: {calculation.exit_message}"
        )
        self.report(f"Action taken: {action}")

    def _check_restart_in_remote(self, calculation):
        """
        Check if the remote folder of a previous calculation contains a restartfile

        :param calculation: node from the previous calculation
        :return: latest restartfile found or None
        """

        try:
            return (
                calculation.outputs.results.get_dict()
                .get("compute_variables", {})
                .get("restartfile_name", None)
            )
        except NotExistentAttributeError:
            return None

    def set_restart_type(self, restart_type, calculation):
        """
        Set the parameters to run the restart calculation

        Depending on the type of restart several variables of the input parameters
        will be changed to try to ensure that the calculation can resume from
        the last stored structure

        :param restart_type: type of the restart approach to be used
        :param calculation: node from the previous calculation
        """
        if restart_type == RestartTypes.FROM_RESTARTFILE:
            self.ctx.inputs.input_restartfile = calculation.outputs.restartfile
            try:
                timestep = int(
                    re.sub("[^0-9]", "", calculation.outputs.restartfile.filename)
                )
            except ValueError:
                timestep = 0

            if "parameters" in self.ctx.inputs and "md" in self.ctx.inputs.parameters:
                self.ctx.inputs.parameters["md"]["reset_timestep"] = timestep
                del self.ctx.inputs.parameters["md"]["velocity"]
                self.logger.warning(
                    "Removing the velocity parameter from the MD control"
                )
        if restart_type == RestartTypes.FROM_STRUCTURE:
            self.ctx.inputs.structure = calculation.outputs.structure

            timestep = calculation.outputs.trajectory.time_steps[-1]
            if "parameters" in self.ctx.inputs and "md" in self.ctx.inputs.parameters:
                self.ctx.inputs.parameters["md"]["reset_timestep"] = timestep
                del self.ctx.inputs.parameters["md"]["velocity"]
                self.logger.warning(
                    "Removing the velocity parameter from the MD control"
                )
        if restart_type == RestartTypes.FROM_REMOTEFOLDER:
            latest_file = self._check_restart_in_remote(calculation=calculation)

            if latest_file is not None:
                timestep = int(re.sub("[^0-9]", "", latest_file))
                self.ctx.inputs.settings["previous_restartfile"] = latest_file
                self.ctx.inputs.parent_folder = calculation.outputs.remote_folder
                if (
                    "parameters" in self.ctx.inputs
                    and "md" in self.ctx.inputs.parameters
                ):
                    self.ctx.inputs.parameters["md"]["reset_timestep"] = timestep
                    del self.ctx.inputs.parameters["md"]["velocity"]
                    self.logger.warning(
                        "Removing the velocity parameter from the MD control"
                    )
        if restart_type == RestartTypes.FROM_SCRATCH:
            self.ctx.inputs.metadata.options["max_wallclock_seconds"] *= 1.50

    @process_handler(priority=600)
    def handle_unrecoverable_failure(self, calculation):
        """
        Handle calculations with unrecoverable errors.

        Checks if the calculation ended with an exit status below 400 if so
        abort the work chain.
        """
        # pylint: disable=inconsistent-return-statements
        if calculation.is_failed and calculation.exit_status < 400:
            self.report_error_handled(calculation, "unrecoverable error, aborting...")
            return ProcessHandlerReport(
                True,
                self.exit_codes.ERROR_UNRECOVERABLE_FAILURE,  # pylint: disable=no-member
            )
        return None

    @process_handler(
        priority=610,
        exit_codes=[
            LammpsBaseCalculation.exit_codes.ERROR_OUT_OF_WALLTIME,  # pylint: disable=no-member
        ],
    )
    def handle_out_of_walltime(self, calculation):
        """
        Handle calculations where the walltime was reached.

        The handler will try to find a configuration to restart from with the
        following priority

        1. Use a stored restart file in the repository from the previous calculation.
        2. Use a restartfile found in the remote folder from the previous calculation.
        3. Use the structure from the last step of the trajectory from the previous calculation.
        4. Restart from scratch
        """
        self.report("Walltime reached attempting restart")

        latest_file = self._check_restart_in_remote(calculation=calculation)
        if "restartfile" in calculation.outputs:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_RESTARTFILE,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the stored restartfile",
            )
        elif latest_file is not None:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_REMOTEFOLDER,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the remote folder of the previous calculation",
            )
        elif "trajectory" in calculation.outputs:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_STRUCTURE,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the last step of the trajectory",
            )
        else:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_SCRATCH,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from scratch and increasing the walltime",
            )
        return ProcessHandlerReport(True)

    @process_handler(
        priority=620,
        exit_codes=[
            LammpsBaseCalculation.exit_codes.ERROR_FORCE_NOT_CONVERGED,  # pylint: disable=no-member
            LammpsBaseCalculation.exit_codes.ERROR_ENERGY_NOT_CONVERGED,  # pylint: disable=no-member
        ],
    )
    def handle_minimization_not_converged(self, calculation):
        """
        Handle calculations where the minimization did not converge

        The handler will try to find a configuration to restart from with the
        following priority

        1. Use a stored restart file in the repository from the previous calculation.
        2. Use a restartfile found in the remote folder from the previous calculation.
        3. Use the structure from the last step of the trajectory from the previous calculation.

        This handler should never start from restart as at least the trajectory
        should always exist, if the calculation finished successfully.
        """

        if (
            LammpsBaseCalculation.exit_codes.ERROR_ENERGY_NOT_CONVERGED.status  # pylint: disable=no-member
        ):
            self.report("Energy not converged during minimization, attempting restart")
        if (
            LammpsBaseCalculation.exit_codes.ERROR_FORCE_NOT_CONVERGED.status  # pylint: disable=no-member
        ):
            self.report("Force not converged during minimization, attempting restart")
        latest_file = self._check_restart_in_remote(calculation=calculation)
        if "restartfile" in calculation.outputs:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_RESTARTFILE,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the stored restartfile",
            )
        elif latest_file is not None:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_REMOTEFOLDER,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the remote folder of the previous calculation",
            )
        elif "trajectory" in calculation.outputs:
            self.set_restart_type(
                restart_type=RestartTypes.FROM_STRUCTURE,
                calculation=calculation,
            )
            self.report_error_handled(
                calculation,
                "restarting from the last step of the trajectory",
            )
        else:
            self.report_error_handled(
                calculation,
                "did not find any configuration to restart from, something is wrong, aborting...",
            )
            return ProcessHandlerReport(
                True,
                self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE,  # pylint: disable=no-member
            )
        return ProcessHandlerReport(True)
