import numpy as np

class Geometry:
    def __init__(self, coordinates: np.array):
        self.xa = coordinates[0]
        self.ya = coordinates[1]
        self.xb = coordinates[2]
        self.yb = coordinates[3]
        self.xc = coordinates[4]
        self.yc = coordinates[5]
        self.Deq = self.calc_deq()

    def calc_deq(self):
        Dab = ((self.xb - self.xa)**2 + (self.yb - self.ya)**2)**0.5
        Dbc = ((self.xc - self.xb)**2 + (self.yc - self.yb)**2)**0.5
        Dca = ((self.xa - self.xc)**2 + (self.ya - self.yc)**2)**0.5
        return (Dab * Dbc * Dca)**(1/3)