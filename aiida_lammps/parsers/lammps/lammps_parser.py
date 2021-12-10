"""
Base parser for LAMMPS calculations.

It takes care of parsing the log.lammps file, the trajectory file and the
yaml file with the final value of the variables printed in the ``thermo_style``.
"""
from aiida import orm
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida_lammps.common.raw_parsers import parse_final_data, parse_logfile
from aiida_lammps.data.lammps_potential import LammpsPotentialData


class LAMMPSBaseParser(Parser):
    """
    Base parser for LAMMPS calculations.

    It takes care of parsing the log.lammps file, the trajectory file and the
    yaml file with the final value of the variables printed in the
    ``thermo_style``.
    """
    def __init__(self, node):
        """Initialize the parser"""
        # pylint: disable=useless-super-delegation, super-with-arguments
        super(LAMMPSBaseParser, self).__init__(node)

    def parse(self, **kwargs):
        """
        Parse the files produced by lammps.

        It takes care of parsing the log.lammps file, the trajectory file and the
        yaml file with the final value of the variables printed in the
        ``thermo_style``.
        """
        # pylint: disable=too-many-return-statements

        try:
            out_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        list_of_files = out_folder.list_object_names()

        # check log file
        if self.node.get_option('output_filename') not in list_of_files:
            return self.exit_codes.ERROR_LOG_FILE_MISSING
        parsed_data = parse_logfile(
            filename=self.node.get_option('output_filename'))
        if parsed_data is None:
            return self.exit_codes.ERROR_PARSING_LOGFILE
        global_data = parsed_data['global']
        arrays = parsed_data['time_dependent']

        # check final variable file
        if self.node.get_option(
                'final_variable_filename') not in list_of_files:
            return self.exit_codes.ERROR_FINAL_VARIABLE_FILE_MISSING

        final_variables = parse_final_data(
            filename=self.node.get_option('final_variable_filename'))
        if final_variables is None:
            return self.exit_codes.ERROR_PARSING_FINAL_VARIABLES

        results = orm.Dict(dict={
            **final_variables, 'compute_variables': global_data
        })

        # Expose the results from the log.lammps outputs
        self.out('results', results)

        # Get the time-dependent outputs exposed as a dictionary
        time_dependent_computes = orm.Dict(dict=arrays)
        self.out('time_dependent_computes', time_dependent_computes)

        # check trajectory file
        if self.node.get_option('trajectory_filename') not in list_of_files:
            return self.exit_codes.ERROR_TRAJECTORY_FILE_MISSING
        # Gather the lammps trajectory data
        lammps_trajectory = LammpsPotentialData(
            self.node.get_option('trajectory_filename'))
        self.out('trajectories', lammps_trajectory)

        # check stdout
        if self.node.get_option('scheduler_stdout') not in list_of_files:
            return self.exit_codes.ERROR_STDOUT_FILE_MISSING

        # check stderr
        if self.node.get_option('scheduler_stderr') not in list_of_files:
            return self.exit_codes.ERROR_STDERR_FILE_MISSING

        return None
