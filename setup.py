from setuptools import setup, find_packages

setup(
    name='aiida-lammps',
    version='0.1',
    description='AiiDA plugin for LAMMPS',
    url='https://github.com/abelcarreras/aiida_extensions',
    author='Abel Carreras',
    author_email='abelcarreras83@gmail.com',
    license='MIT license',
    packages=find_packages(exclude=['aiida']),
    requires=['phonopy', 'numpy', 'dynaphopy'],
    setup_requires=['reentry'],
    reentry_register=True,
    entry_points={
        'aiida.calculations': [
            'lammps.combinate = aiida_lammps.calculations.lammps.combinate:CombinateCalculation',
            'lammps.force = aiida_lammps.calculations.force:ForceCalculation',
            'lammps.md = aiida_lammps.calculations.lammps.md:MdCalculation',
            'lammps.optimize = aiida_lammps.calculations.lammps.optimize:OptimizeCalculation',
            'dynaphopy = aiida_lammps.calculations.dynaphopy: DynaphopyCalculation'],
        'aiida.parsers': [
            'lammps.force = aiida_lammps.parsers.lammps.force:ForceParser',
            'lammps.md = aiida_lammps.parsers.lammps.md:MdParser',
            'lammps.optimize = aiida_lammps.parsers.lammps.optimize:OptimizeParser',
            'dynaphopy = aiida_lammps.parsers.dynaphopy: DynaphopyParser']
        }
    )