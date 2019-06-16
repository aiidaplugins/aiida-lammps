import importlib
import io
import os
from aiida.orm import Dict


class EmpiricalPotential(Dict):
    """
    Store the band structure.
    """

    def __init__(self, **kwargs):

        structure = kwargs.pop('structure', None)
        potential_type = kwargs.pop('type', None)
        potential_data = kwargs.pop('data', None)
        data_from_file = kwargs.pop('data_from_file', None)

        super(EmpiricalPotential, self).__init__(**kwargs)

        dirname = os.path.dirname(__file__)
        types_list = [os.path.splitext(l)[0] for l in os.listdir(dirname)]
        types_list.remove('__init__')
        self.update_dict({'types_list': types_list})

        if structure:
            print(structure)
            names = [site.name for site in structure.kinds]

            self.update_dict({'names': names})
        else:
            raise RuntimeError('structure not provided')

        if potential_type:
            self.set_type(potential_type)
        if potential_data:
            self.set_data(potential_data)
        if data_from_file:
            self.set_data_from_file(data_from_file)

    def set_type(self, potential_type):
        """
        Store the potential type (ex. Tersoff, EAM, LJ, ..)
        """

        if potential_type in self.dict.types_list:
            self.update_dict({'potential_type': potential_type})
        else:
            print('This lammps potential is not implemented, set as generic')
            self.update_dict({'potential_type': 'generic'})

    def get_type(self):
        """
        Store the potential type (ex. Tersoff, EAM, LJ, ..)
        """
        if 'potential_type' not in self.get_dict():
            return 'generic'
        return self.dict.potential_type

    def set_data_from_file(self, filename):
        """
        read data from file
        """
        with io.open(filename) as handle:
            data = handle.readlines()
        self.update_dict({'potential_data': data})

    def set_data(self, data):
        """
        read data from file
        """
        self.update_dict({'potential_data': data})

    def create_potential_file(self, filename):
        with io.open(filename, 'r') as handle:
            handle.write(self.get_data_txt())

    def _get_module(self):

        pottype = self.get_type()
        if pottype not in self.dict.types_list:
            return None
        else:
            return importlib.import_module('.{}'.format(pottype), __name__)

    def get_potential_file(self):

        if self._get_module() is None:
            return self.dict.potential_data
        else:
            return self._get_module().generate_LAMMPS_potential(self.dict.potential_data)

    def get_input_potential_lines(self, names=None, potential_filename='potential.pot'):

        if self._get_module() is None:
            return self.dict.input_potential_lines
        else:
            return self._get_module().get_input_potential_lines(self.dict.potential_data,
                                                                names=self.names,
                                                                potential_filename=potential_filename)

    def set_input_potential_lines(self, data):
        """
        set the string command to put in lammps input to setup the potential
        Ex:
             pair_style      eam
             pair_coeff      * *  Si
        """
        self.update_dict({'input_potential_lines': data})

    def set_atom_style(self, data):
        """
        set lammps atom style
        Ex: atomic
        """
        self.update_dict({'atom_style': data})

    def set_default_units(self, data):
        """
        set default lammps unit set
        Ex: metal
        """
        self.update_dict({'atom_style': data})

    @property
    def default_units(self):
        if self._get_module() is None:
            return self.dict.default_units
        else:
            return self._get_module().DEFAULT_UNITS

    @property
    def atom_style(self):
        if self._get_module() is None:
            return self.dict.atom_style
        else:
            return self._get_module().ATOM_STYLE

    @property
    def names(self):
        return self.dict.names
