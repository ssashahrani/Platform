import math
import pandas as pd

class Transformer:
    def __init__(self, name, bus1, bus2, z_t, x_r, syst_base, transformer_rating, voltage_base, grounding_bus1, grounding_bus2, grounding_R, grounding_X):
        self.name = name
        self.bus1 = bus1
        self.bus2 = bus2
        self.z_t = z_t
        self.x_r = x_r
        self.syst_base = syst_base
        self.voltage_base = voltage_base
        self.transformer_rating = transformer_rating
        self.grounding_bus1 = grounding_bus1
        self.grounding_bus2 = grounding_bus2
        self.grounding_R = grounding_R # in ohm
        self.grounding_X = grounding_X # in ohm

    def calc_y(self):
        """Creating Y-bus matrix for PowerFlow"""
        theta = math.atan(self.x_r)
        r_old = self.z_t * math.cos(theta)
        x_old = self.z_t * math.sin(theta)
        r_new = r_old * self.syst_base / self.transformer_rating
        x_new = x_old * self.syst_base / self.transformer_rating
        z_total = complex(r_new, x_new)
        self.y_pu = 1 / z_total
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_pu
        y_df.loc[self.bus2, self.bus2] = self.y_pu
        y_df.loc[self.bus1, self.bus2] = -self.y_pu
        y_df.loc[self.bus2, self.bus1] = -self.y_pu
        return y_df

    def calc_y_sf(self):
        """For Positive sequence bus admittance matrix"""
        theta = math.atan(self.x_r)
        r_old = self.z_t * math.cos(theta)
        x_old = self.z_t * math.sin(theta)
        r_new = r_old * self.syst_base / self.transformer_rating
        x_new = x_old * self.syst_base / self.transformer_rating

        z_total = complex(r_new, x_new)
        self.y_pu = 1 / z_total
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_pu
        y_df.loc[self.bus2, self.bus2] = self.y_pu
        y_df.loc[self.bus1, self.bus2] = -self.y_pu
        y_df.loc[self.bus2, self.bus1] = -self.y_pu
        return y_df

    def calc_y_ns(self):
        """For Negative sequence bus admittance matrix"""
        theta = math.atan(self.x_r)
        r_old = self.z_t * math.cos(theta)
        x_old = self.z_t * math.sin(theta)
        r_new = r_old * self.syst_base / self.transformer_rating
        x_new = x_old * self.syst_base / self.transformer_rating
        z_total = complex(r_new, x_new)
        self.y_pu = 1 / z_total
        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        y_df.loc[self.bus1, self.bus1] = self.y_pu
        y_df.loc[self.bus2, self.bus2] = self.y_pu
        y_df.loc[self.bus1, self.bus2] = -self.y_pu
        y_df.loc[self.bus2, self.bus1] = -self.y_pu
        return y_df

    def calc_y_zs(self):
        """For Zero sequence bus admittance matrix taking into account sequence networks."""
        theta = math.atan(self.x_r)
        r_old = self.z_t * math.cos(theta)
        x_old = self.z_t * math.sin(theta)
        r_new = r_old * self.syst_base / self.transformer_rating
        x_new = x_old * self.syst_base / self.transformer_rating
        z_total = complex(r_new, x_new)
        self.y_pu = 1 / z_total

        Grounding_base = self.voltage_base**2 / self.syst_base
        self.ground_R_pu = 3 * self.grounding_R / Grounding_base
        self.ground_X_pu = 3 * self.grounding_X / Grounding_base
        self.ground_Z_pu = complex(self.ground_R_pu, self.ground_X_pu)
        self.rt = r_new + self.ground_R_pu
        self.xt = x_new + self.ground_X_pu
        zt = complex(self.rt, self.xt)
        self.no_connection = 0
        """No connection between Buses due to "Delta" and Ungrounded Wye"""
        self.y_pu = 1 / zt

        y_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        if self.grounding_bus1 == 'Delta' and self.grounding_bus2 == 'Delta':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Delta' and self.grounding_bus2 == 'Solidly Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.y_pu
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Delta' and self.grounding_bus2 == 'Ungrounded Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Delta' and self.grounding_bus2 == 'Impedance Grounded Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.y_pu
            y_df.loc[self.bus1, self.bus2] = - self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Solidly Y' and self.grounding_bus2 == 'Solidly Y':
            y_df.loc[self.bus1, self.bus1] =  self.y_pu
            y_df.loc[self.bus2, self.bus2] =  self.y_pu
            y_df.loc[self.bus1, self.bus2] = -self.y_pu
            y_df.loc[self.bus2, self.bus1] = -self.y_pu

        elif self.grounding_bus1 == 'Solidly Y' and self.grounding_bus2 == 'Delta':
            y_df.loc[self.bus1, self.bus1] = self.y_pu
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Solidly Y' and self.grounding_bus2 == 'Impedance Grounded Y':
            y_df.loc[self.bus1, self.bus1] =  self.y_pu
            y_df.loc[self.bus2, self.bus2] =  self.y_pu
            y_df.loc[self.bus1, self.bus2] = -self.y_pu
            y_df.loc[self.bus2, self.bus1] = -self.y_pu

        elif self.grounding_bus1 == 'Solidly Y' and self.grounding_bus2 == 'Ungrounded Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Ungrounded Y' and self.grounding_bus2 == 'Ungrounded Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Ungrounded Y' and self.grounding_bus2 == 'Solidly Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Ungrounded Y' and self.grounding_bus2 == 'Delta':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Ungrounded Y' and self.grounding_bus2 == 'Impedance Grounded Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Impedance Grounded Y' and self.grounding_bus2 == 'Ungrounded Y':
            y_df.loc[self.bus1, self.bus1] = self.no_connection
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        elif self.grounding_bus1 == 'Impedance Grounded Y' and self.grounding_bus2 == 'Solidly Y':
            y_df.loc[self.bus1, self.bus1] =  self.y_pu
            y_df.loc[self.bus2, self.bus2] =  self.y_pu
            y_df.loc[self.bus1, self.bus2] = -self.y_pu
            y_df.loc[self.bus2, self.bus1] = -self.y_pu

        elif self.grounding_bus1 == 'Impedance Grounded Y' and self.grounding_bus2 == 'Delta':
            y_df.loc[self.bus1, self.bus1] = self.y_pu
            y_df.loc[self.bus2, self.bus2] = self.no_connection
            y_df.loc[self.bus1, self.bus2] = self.no_connection
            y_df.loc[self.bus2, self.bus1] = self.no_connection

        else: # self.grounding_bus1 == 'Impedance Grounded Y' and self.grounding_bus2 == 'Impedance Grounded Y':
            y_df.loc[self.bus1, self.bus1] =  self.y_pu
            y_df.loc[self.bus2, self.bus2] =  self.y_pu
            y_df.loc[self.bus1, self.bus2] = -self.y_pu
            y_df.loc[self.bus2, self.bus1] = -self.y_pu
        return y_df


    """The following serves for calculating the current flowing through each line, Line Flows and Line Losses."""
    def calc_z_diaG(self):
        z_diaG = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        z_diaG.loc[self.bus1, self.bus1] = 0
        z_diaG.loc[self.bus2, self.bus2] = 0
        z_diaG.loc[self.bus1, self.bus2] = 0
        z_diaG.loc[self.bus2, self.bus1] = 0
        return z_diaG

    """The following is just for practice purpose"""
    def calc_z(self):
        theta = math.atan(self.x_r)
        r_old = self.z_t * math.cos(theta)
        x_old = self.z_t * math.sin(theta)
        r_new = r_old * self.syst_base / self.transformer_rating
        x_new = x_old * self.syst_base / self.transformer_rating
        z_total = complex(r_new, x_new)
        self.y_pu = 1 / z_total
        z_df = pd.DataFrame(0.0, index=[self.bus1, self.bus2], columns=[self.bus1, self.bus2], dtype=complex)
        z_df.loc[self.bus1, self.bus1] = 0
        z_df.loc[self.bus2, self.bus2] = 0
        z_df.loc[self.bus1, self.bus2] = self.y_pu
        z_df.loc[self.bus2, self.bus1] = self.y_pu
        return z_df

