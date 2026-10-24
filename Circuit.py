from Bus import Bus
from Line import Line
from Transformer import Transformer
from Generator import Generator

class Circuit:
    def __init__(self, name):
        self.name = name
        self.buses = {}
        self.impedances = []
        self.impedancesSF = []
        """For Positive Sequence"""

        self.impedancesNS = []
        """For Negative Sequence"""

        self.impedancesZS = []
        """For Zero Sequence"""

        self.generators = []
        self.lines = {}

    def add_bus(self, bus_name):
        if bus_name not in self.buses:
            self.buses[bus_name] = Bus(bus_name)

    def add_Line_impedance_element(self, name, bus1, bus2, line_conductor, geometry, length, ampacity):
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedances.append(Line(name, bus1, bus2, line_conductor, geometry, length, ampacity))

    def add_Line_impedanceF_element(self, name, bus1, bus2, line_conductor, geometry, length, ampacity):
        """For Positive Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesSF.append(Line(name, bus1, bus2, line_conductor, geometry, length, ampacity))

    def add_Line_impedanceNS_element(self, name, bus1, bus2, line_conductor, geometry, length, ampacity):
        """For Negative Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesNS.append(Line(name, bus1, bus2, line_conductor, geometry, length, ampacity))

    def add_Line_impedanceZS_element(self, name, bus1, bus2, line_conductor, geometry, length, ampacity):
        """For Zero Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesZS.append(Line(name, bus1, bus2, line_conductor, geometry, length, ampacity))

    def add_Transformer_impedance_element(self, name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X):
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedances.append(Transformer(name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X))

    def add_Transformer_impedanceF_element(self, name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X):
        """For Positive Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesSF.append(Transformer(name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X))

    def add_Transformer_impedanceNS_element(self, name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X):
        """For Negative Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesNS.append(Transformer(name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X))

    def add_Transformer_impedanceZS_element(self, name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X):
        """For Zero Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesZS.append(Transformer(name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X))

    def add_generator_reactancePS_element(self, name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X):
        """For Positive Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesSF.append(Generator(name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X))

    def add_generator_reactanceNS_element(self, name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X):
        """For Negative Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesNS.append(Generator(name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X))

    def add_generator_reactanceZS_element(self, name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X):
        """For Zero Sequence Components"""
        self.add_bus(bus1)
        self.add_bus(bus2)
        self.impedancesZS.append(Generator(name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X))