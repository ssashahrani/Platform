import math
from Geometry import Geometry
from Bundle import Bundle

class Conductor:
    def __init__(self, rd: float, bundle_spacing: float, Ds: float, rc: float, coordinates, num_conductors):
        self.rd = rd
        self.bundle_spacing = bundle_spacing
        self.num_conductors = num_conductors
        self.Ds = Ds
        self.rc = rc
        self.geometry = Geometry(coordinates)
        self.bundle = Bundle(bundle_spacing, rd, num_conductors)

    def calc_resistance(self):
        Rp = self.rc / self.num_conductors
        return Rp

    def calc_reactance(self, Ds):
        DSL = self.bundle.calc_dsl(Ds)
        Lp = 2e-7 * math.log(self.geometry.Deq / DSL)
        Xp = 2 * math.pi * 60 * Lp * 1609
        return Xp

    def calc_susceptance(self, Deq):
        DSC = self.bundle.calc_dsc()
        Cp = (2 * math.pi * 8.854e-12) / math.log(Deq / DSC)
        Bp = 2 * math.pi * 60 * Cp * 1609
        return Bp

    def set_base_values(self, base_voltageHV, base_voltageLV1, base_voltageLV2 , base_power):
        self.base_voltageHV = base_voltageHV
        self.base_voltageLV1 = base_voltageLV1
        self.base_voltageLV2 = base_voltageLV2
        self.base_power = base_power
        self.base_impedance = base_voltageHV ** 2 / base_power
        self.base_impedanceL1 = base_voltageLV1 ** 2 / base_power
        self.base_impedanceL2 = base_voltageLV2 ** 2 / base_power
        self.base_current_L = base_power / (math.sqrt(3) * base_voltageHV)
        self.base_current_H1 = base_power / (math.sqrt(3) * base_voltageLV1)
        self.base_current_H2 = base_power / (math.sqrt(3) * base_voltageLV2)
        self.base_susceptance = 1 / self.base_impedance


    def calc_resistance_pu(self, Rp):
        return Rp / self.base_impedance

    def calc_reactance_pu(self, Xp):
        return Xp / self.base_impedance

    def calc_susceptance_pu(self, Bp):
        return Bp / self.base_susceptance

    def calc_ampacity_pu(self, ampacity):
        self.ampacity_pu= ampacity / self.base_current_L
        return self.ampacity_pu
