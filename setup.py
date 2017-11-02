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
            'lammps.combinate = plugins.jobs.lammps.combinate:CombinateCalculation',
            'lammps.force = plugins.jobs.lammps.force:ForceCalculation',
            'lammps.md = plugins.jobs.lammps.md:MdCalculation',
            'lammps.optimize = plugins.jobs.lammps.optimize:OptimizeCalculation',
            'dynaphopy = plugins.jobs.dynaphopy: DynaphopyCalculation'],
        'aiida.parsers': [
            'lammps.force = plugins.parsers.lammps.force:ForceParser',
            'lammps.md = plugins.parsers.lammps.md:MdParser',
            'lammps.optimize = plugins.parsers.lammps.optimize:OptimizeParser',
            'dynaphopy = plugins.parsers.dynaphopy: DynaphopyParser']
        }
    )