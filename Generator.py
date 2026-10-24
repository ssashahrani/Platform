import pandas as pd
class Generator:
    def __init__(self, name, bus1, bus2, x_g, syst_base, generator_rating, voltage_base, grounding_R, grounding_X):
        self.name = name
        self.bus1 = bus1
        self.bus2 = bus2
        self.x_g = x_g
        self.syst_base = syst_base
        self.generator_rating = generator_rating
        self.syst_base = syst_base
        self.voltage_base = voltage_base
        self.grounding_R = grounding_R # in ohm
        self.grounding_X = grounding_X # in ohm


    def calc_y_sf(self):
        """For Positive sequence bus admittance matrix"""
        x_new = self.x_g * self.syst_base / self.generator_rating
        self.z_f= complex(0, x_new)
        self.y_pu = 1 / self.z_f
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_pu
        y_df.loc[self.bus2, self.bus2] = 0
        y_df.loc[self.bus1, self.bus2] = 0
        y_df.loc[self.bus2, self.bus1] = 0
        return y_df

    def calc_y_ns(self):
        """For Negative sequence bus admittance matrix"""
        x_new = self.x_g * self.syst_base / self.generator_rating
        self.z_f= complex(0, x_new)
        self.y_pu = 1 / self.z_f
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_pu
        y_df.loc[self.bus2, self.bus2] = 0
        y_df.loc[self.bus1, self.bus2] = 0
        y_df.loc[self.bus2, self.bus1] = 0
        return y_df

    def calc_y_zs(self):
        """For Zero sequence bus admittance matrix"""
        x_new = self.x_g * self.syst_base / self.generator_rating
        Grounding_base = self.voltage_base**2 / self.syst_base
        self.ground_R_pu = 3 * self.grounding_R / Grounding_base
        self.ground_X_pu = 3 * self.grounding_X / Grounding_base

        self.rg = self.ground_R_pu
        self.xg = x_new + self.ground_X_pu
        zg = complex(self.rg, self.xg)
        self.yg = 1 / zg

        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.yg
        y_df.loc[self.bus2, self.bus2] = 0
        y_df.loc[self.bus1, self.bus2] = 0
        y_df.loc[self.bus2, self.bus1] = 0
        return y_df