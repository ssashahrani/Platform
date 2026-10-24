import numpy as np
import pandas as pd
from Circuit import Circuit
from Jacobian import Jacobian

class Solution:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.ybus = self.construct_ybus()
        self.ybus_SF = self.construct_ybusSF()
        """For Positive Sequence"""
        self.ybus_NS = self.construct_ybusNS()
        """For Negative Sequence"""
        self.ybus_ZS = self.construct_ybusZS()
        """For Zero Sequence"""

        self.zbus_offDiag_matrix = self.construct_zbus_offDiag()
        self.jacobian = Jacobian(circuit)

    def construct_ybus(self):
        """Creating Y-bus matrix for PowerFlow"""
        ybus = pd.DataFrame(np.zeros((len(self.circuit.buses), len(self.circuit.buses)), dtype=complex),
                            index=self.circuit.buses.keys(),
                            columns=self.circuit.buses.keys())

        for impedance in self.circuit.impedances:
            y = impedance.calc_y()
            ybus.loc[y.index, y.columns] += y
        return ybus

    def construct_ybusSF(self):
        """For Positive sequence bus admittance matrix"""
        ybusSF = pd.DataFrame(np.zeros((len(self.circuit.buses), len(self.circuit.buses)), dtype=complex),
                             index=self.circuit.buses.keys(),
                             columns=self.circuit.buses.keys())

        for impedance in self.circuit.impedancesSF:
            y_SF = impedance.calc_y_sf()
            ybusSF.loc[y_SF.index, y_SF.columns] += y_SF
        return ybusSF

    def construct_ybusNS(self):
        """For Negative sequence bus admittance matrix"""
        ybusNS = pd.DataFrame(np.zeros((len(self.circuit.buses), len(self.circuit.buses)), dtype=complex),
                             index=self.circuit.buses.keys(),
                             columns=self.circuit.buses.keys())

        for impedance in self.circuit.impedancesNS:
            y_NS = impedance.calc_y_ns()
            ybusNS.loc[y_NS.index, y_NS.columns] += y_NS
        return ybusNS

    def construct_ybusZS(self):
        """For Zero sequence bus admittance matrix"""
        ybusZS = pd.DataFrame(np.zeros((len(self.circuit.buses), len(self.circuit.buses)), dtype=complex),
                             index=self.circuit.buses.keys(),
                             columns=self.circuit.buses.keys())

        for impedance in self.circuit.impedancesZS:
            y_ZS = impedance.calc_y_zs()
            ybusZS.loc[y_ZS.index, y_ZS.columns] += y_ZS
        return ybusZS


    """The following serves for calculating the current flowing through each line, Line Flows and Line Losses."""
    def construct_zbus_offDiag(self):
        zbus_offDiag = pd.DataFrame(np.zeros((len(self.circuit.buses), len(self.circuit.buses)), dtype=complex),
                            index=self.circuit.buses.keys(),
                            columns=self.circuit.buses.keys())

        for impedance in self.circuit.impedances:
            z = impedance.calc_z()
            zbus_offDiag.loc[z.index, z.columns] += z
        return zbus_offDiag

    def ybus_to_polar(self):
        """This Matrix is for converting A Bus from rectangular form to Polar form."""
        ybus_polar = pd.DataFrame(index=self.ybus.index, columns=self.ybus.columns, dtype='object')

        for row in self.ybus.index:
            for col in self.ybus.columns:
                magnitude = np.abs(self.ybus.at[row, col])
                angle = np.degrees(np.arctan2(self.ybus.at[row, col].imag, self.ybus.at[row, col].real))
                ybus_polar.at[row, col] = (magnitude, angle)

        return ybus_polar



    """The following matrix is just for practice purpose"""
    def construct_zbus_Diag(self):
        zbus_Diag = pd.DataFrame(np.zeros((len(self.circuit.buses), len(self.circuit.buses)), dtype=complex),
                            index=self.circuit.buses.keys(),
                            columns=self.circuit.buses.keys())

        for impedance in self.circuit.impedances:
            z = impedance.calc_z_diaG()
            zbus_Diag.loc[z.index, z.columns] += z
        return zbus_Diag