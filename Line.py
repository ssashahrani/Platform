import pandas as pd

class Line:
    def __init__(self, name, bus1, bus2, line_conductor, geometry, length, ampacity):
        self.name = name
        self.bus1 = bus1
        self.bus2 = bus2
        self.length = length
        self.r = line_conductor.calc_resistance()
        self.geometry = geometry
        Deq = self.geometry.Deq
        self.x = line_conductor.calc_reactance(line_conductor.Ds)
        self.b = line_conductor.calc_susceptance(Deq)
        self.r_pu = line_conductor.calc_resistance_pu(self.r)
        self.x_pu = line_conductor.calc_reactance_pu(self.x)
        self.b_pu = line_conductor.calc_susceptance_pu(self.b)
        self.ampacity = ampacity
        self.ampacity_pu = line_conductor.calc_ampacity_pu(self.ampacity)

    def calc_y(self):
        """Creating Y-bus matrix for PowerFlow"""
        z_ij = complex(self.r_pu, self.x_pu) * self.length
        self.y_ij = 1 / z_ij
        self.y_ii = self.y_ij + 0.5 * complex(0, self.b_pu * self.length)
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_ii
        y_df.loc[self.bus2, self.bus2] = self.y_ii
        y_df.loc[self.bus1, self.bus2] = -self.y_ij
        y_df.loc[self.bus2, self.bus1] = -self.y_ij
        return y_df

    def calc_y_sf(self):
        """For Positive sequence bus admittance matrix"""
        z_ij = complex(self.r_pu, self.x_pu) * self.length
        self.y_ij = 1 / z_ij
        self.y_ii = self.y_ij + 0.5 * complex(0, self.b_pu * self.length)
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_ii
        y_df.loc[self.bus2, self.bus2] = self.y_ii
        y_df.loc[self.bus1, self.bus2] = -self.y_ij
        y_df.loc[self.bus2, self.bus1] = -self.y_ij
        return y_df

    def calc_y_ns(self):
        """For Negative sequence bus admittance matrix"""
        z_ij = complex(self.r_pu, self.x_pu) * self.length
        self.y_ij = 1 / z_ij
        self.y_ii = self.y_ij + 0.5 * complex(0, self.b_pu * self.length)
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_ii
        y_df.loc[self.bus2, self.bus2] = self.y_ii
        y_df.loc[self.bus1, self.bus2] = -self.y_ij
        y_df.loc[self.bus2, self.bus1] = -self.y_ij
        return y_df

    def calc_y_zs(self):
        """For Zero sequence bus admittance matrix"""
        z_ij = complex(self.r_pu, self.x_pu) * self.length
        self.y_ij = 1 / z_ij
        self.y_ii = self.y_ij + 0.5 * complex(0, self.b_pu * self.length)
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_ii
        y_df.loc[self.bus2, self.bus2] = self.y_ii
        y_df.loc[self.bus1, self.bus2] = -self.y_ij
        y_df.loc[self.bus2, self.bus1] = -self.y_ij
        return y_df


    """The following serves for calculating the current flowing through each line, Line Flows and Line Losses."""
    def calc_z_diaG(self):
        z_ij = 0.5 * complex(0, self.b_pu * self.length)
        self.y_ij = z_ij
        self.y_ii = 0
        calc_z_diaG = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        calc_z_diaG.loc[self.bus1, self.bus1] = self.y_ii
        calc_z_diaG.loc[self.bus2, self.bus2] = self.y_ii
        calc_z_diaG.loc[self.bus1, self.bus2] = self.y_ij
        calc_z_diaG.loc[self.bus2, self.bus1] = self.y_ij
        return calc_z_diaG

    """The following is just for practice purpose"""
    def calc_z(self):
        z_ij = complex(self.r_pu, self.x_pu) * self.length
        self.y_ij = 1 / z_ij
        self.y_ii = 0
        z_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        z_df.loc[self.bus1, self.bus1] = self.y_ii
        z_df.loc[self.bus2, self.bus2] = self.y_ii
        z_df.loc[self.bus1, self.bus2] = self.y_ij
        z_df.loc[self.bus2, self.bus1] = self.y_ij
        return z_df