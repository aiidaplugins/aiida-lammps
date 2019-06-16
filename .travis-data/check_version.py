"""Check that version numbers match.

Check version number in setup.json and aiida_crystal17/__init__.py and make sure
they match.
"""
import os
import json
import sys

this_path = os.path.split(os.path.realpath(__file__))[0]

# Get content of setup.json
setup_fname = 'setup.json'
setup_path = os.path.join(this_path, os.pardir, setup_fname)
with open(setup_path) as f:
    setup_content = json.load(f)

# Get version from python package
sys.path.insert(0, os.path.join(this_path, os.pardir))
import aiida_crystal17  # noqa
version = aiida_crystal17.__version__

if version != setup_content['version']:
    print("Version number mismatch detected:")
    print("Version number in '{}': {}".format(setup_fname,
                                              setup_content['version']))
    print("Version number in '{}/__init__.py': {}".format(
        'aiida_crystal17', version))
    sys.exit(1)

# Overwrite version in setup.json
# setup_content['version'] = version
# with open(setup_path, 'w') as f:
# 	json.dump(setup_content, f, indent=4, sort_keys=True)
