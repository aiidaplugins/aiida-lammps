# ``LammpsRawCalculation``

The {class}`~aiida_lammps.calculations.raw.LammpsRawCalculation` performs a LAMMPS calculation from a given LAMMPS input script and a set of files.

## Inputs:

- **script**, ({class}`~aiida.orm.nodes.data.singlefile.SinglefileData`) - Complete input script to use. If specified, `structure`, `potential` and `parameters` are ignored.
- **files**, (Namespace of {class}`~aiida.orm.nodes.data.singlefile.SinglefileData`), *optional* - Optional files that should be written to the working directory. This is an
- **filenames**, ({class}`~aiida.orm.nodes.data.dict.Dict`), *optional* - Optional namespace to specify with which filenames the files of ``files`` input should be written.
- **metadata.options.input_filename**, (`str`), *optional* - Name of the input file for the calculation. Defaults to `input.in`.
- **metadata.options.output_filename**, (`str`). *optional* - Name of the main output file for LAMMPS. Defaults to `lammps.out`.
- **metadata.options.parser_name**, (`str`), *optional* - Name of the parser to be used for this calculation. Defaults to `lammps.raw`.

## Outputs:

- **results**, ({class}`~aiida.orm.nodes.data.dict.Dict`) - The parsed data extracted from the lammps output file.
- **remote_folder**, ({class}`~aiida.orm.nodes.data.remote.base.RemoteData`) - Folder in the remote machine where the calculation was performed.
- **remote_stash**, ({class}`~aiida.orm.nodes.data.remote.stash.base.RemoteStashData`), *optional* – Contents of the stash.source_list option are stored in this remote folder after job completion.
- **retrieved**, ({class}`~aiida.orm.nodes.data.folder.FolderData`) - Files that are retrieved by the daemon will be stored in this node. By default the stdout and stderr of the scheduler will be added, but one can add more by specifying them in `CalcInfo.retrieve_list`.
