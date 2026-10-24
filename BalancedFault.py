import numpy as np
"""Symmetrical Fault Analysis Using Bus Impedance Matrix"""
class BalancedFault:
    def __init__(self, circuit, ybus_matrix_F):

        self.circuit = circuit
        self.Zbus = np.linalg.inv(ybus_matrix_F)
        """Bus Impedance Matrix = inverse of ybus_matrix_F"""
        """ybus_matrix_F = the positive sequence bus admittance matrix"""

        self.I_matrix = None


    def fault_current(self, faulted_bus, Vf, Zf):
        Z_kk = self.Zbus[faulted_bus - 1, faulted_bus - 1]
        """To analyze the fault at bus k, I consider the Z_kk element from the bus impedance matrix."""

        self.I_f = Vf / (Z_kk + Zf)
        """Fault current I_f at the faulted bus."""
        self.I_matrix = np.zeros((len(self.Zbus), 1), dtype=complex)
        self.I_matrix[faulted_bus - 1, 0] = self.I_f
        return self.I_f

    def post_fault_voltages(self):

        voltages = 1 - self.Zbus.dot(self.I_matrix)
        """Voltages at each bus during the fault"""
        return voltages




