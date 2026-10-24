import math

class Bundle:
    def __init__(self, bundle_spacing, rd, num_conductors):
        self.bundle_spacing = bundle_spacing
        self.rd = rd
        self.num_conductors = num_conductors
        self.DSL = float
        self.DSC = float

    def calc_dsl(self, Ds):
        self.DSL = math.sqrt(Ds * self.bundle_spacing)
        return self.DSL

    def calc_dsc(self):
        self.DSC = math.sqrt(self.rd * self.bundle_spacing)
        return self.DSC