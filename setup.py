from setuptools import setup, find_packages

setup(
    name='aiida-lammps',
    version='0.1.0a1',
    description='AiiDA plugin for LAMMPS',
    url='https://github.com/abelcarreras/aiida_extensions',
    author='Abel Carreras',
    author_email='abelcarreras83@gmail.com',
    license='MIT license',
    packages=find_packages(),
    install_requires=['aiida-core>=0.12.0', 'numpy', 'packaging'],
    setup_requires=['reentry'],
    reentry_register=True,
    entry_points={
        'aiida.calculations': [
            'lammps.combinate = aiida_lammps.calculations.lammps.combinate:CombinateCalculation',
            'lammps.force = aiida_lammps.calculations.lammps.force:ForceCalculation',
            'lammps.md = aiida_lammps.calculations.lammps.md:MdCalculation',
            'lammps.optimize = aiida_lammps.calculations.lammps.optimize:OptimizeCalculation',
            'dynaphopy = aiida_lammps.calculations.dynaphopy: DynaphopyCalculation'],
        'aiida.parsers': [
            'lammps.force = aiida_lammps.parsers.lammps.force:ForceParser',
            'lammps.md = aiida_lammps.parsers.lammps.md:MdParser',
            'lammps.optimize = aiida_lammps.parsers.lammps.optimize:OptimizeParser',
            'dynaphopy = aiida_lammps.parsers.dynaphopy: DynaphopyParser']
        },
    extras_require={
        "testing": {
            "mock==2.0.0",
            "pgtest==1.1.0",
            "sqlalchemy-diff==0.1.3",
            "pytest==3.6.3",
            "pytest-timeout",
            "wheel>=0.31"
        },
        "phonopy": {
            'dynaphopy',
            # 'aiida_phonopy' # needs to be added to pip
        }
    }
    )