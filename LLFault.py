import numpy as np
import cmath
class LLFault:
    def __init__(self, circuit, ybus_matrix_F, ybus_matrix_NS, ybus_matrix_ZS):
        self.circuit = circuit
        """ybus_matrix_F = Positive sequence bus admittance matrix"""
        """ybus_matrix_NS = Negative sequence bus admittance matrix"""
        """ybus_matrix_ZS = Zero sequence bus admittance matrix"""
        self.Zbus_PS = np.linalg.inv(ybus_matrix_F)
        """Positive sequence bus impedance matrix"""
        self.Zbus_NS = np.linalg.inv(ybus_matrix_NS)
        """Negative sequence bus impedance matrix"""
        self.Zbus_ZS = np.linalg.inv(ybus_matrix_ZS)
        """Zero sequence bus impedance matrix"""

        self.I_matrix_ZS = None
        self.I_matrix_PS = None
        self.I_matrix_NS = None
        a = np.exp(1j * np.deg2rad(120))
        a2 = np.exp(1j * np.deg2rad(240))

        self.a_matrix = np.array([[1, 1, 1],[1, a2, a],[1, a, a2]])
        """a_matrix is a 3 by 3 transformation matrix."""
    def LLfault_current(self, faulted_bus, Vf, Zf):
        """ I consider a falut between phase B and Phase C (Line - Line) Fault """
        current_012 = []  # Symmetrical Component currents for each bus
        current_abc = []  # ABC Phase currents for each bus

        Z_ps_ii = self.Zbus_PS[faulted_bus - 1, faulted_bus - 1]
        Z_ns_ii = self.Zbus_NS[faulted_bus - 1, faulted_bus - 1]


        self.I_f = Vf / (Z_ps_ii+ Z_ns_ii + Zf)

        self.I_matrix_ZS = np.zeros((len(self.Zbus_PS), 1), dtype=complex)
        self.I_matrix_PS = np.zeros((len(self.Zbus_PS), 1), dtype=complex)
        self.I_matrix_NS = np.zeros((len(self.Zbus_PS), 1), dtype=complex)

        self.I_matrix_ZS[faulted_bus - 1, 0] = 0
        self.I_matrix_PS[faulted_bus - 1, 0] = self.I_f
        self.I_matrix_NS[faulted_bus - 1, 0] = - self.I_f

        self.V_ZS = 0  - self.Zbus_ZS.dot(self.I_matrix_ZS)
        self.V_PS = Vf - self.Zbus_PS.dot(self.I_matrix_PS)
        self.V_NS = 0  - self.Zbus_NS.dot(self.I_matrix_NS)

        # Iterating over the system buses
        for i in range(len(self.Zbus_PS)):
            bus_Current_012 = np.zeros((3, 1), dtype=complex)  # 3 here means: zero  , positive , negative sequence
            # Assigning the currents from I_ZS, I_PS, and I_NS to the matrix
            bus_Current_012[0, 0] = self.I_matrix_ZS[i]
            bus_Current_012[1, 0] = self.I_matrix_PS[i]
            bus_Current_012[2, 0] = self.I_matrix_NS[i]

            # Adding the matrix to the list of current matrices
            current_012.append(bus_Current_012)

            # Multiplying (self.a_matrix) by the (bus_current_012) matrix to calculate the phase currents:
            bus_current_abc = self.a_matrix.dot(bus_Current_012)

            # I added the result to the list of phase current matrices:
            current_abc.append(bus_current_abc)
        # We convert all currents to polar form
        polar_LL_current_012 = [[cmath.polar(amp[0]) for amp in bus_currents] for bus_currents in current_012]
        polar_LL_current_abc = [[cmath.polar(amp[0]) for amp in bus_currents] for bus_currents in current_abc]

        return polar_LL_current_012, polar_LL_current_abc # self.I_f, self.V_PS, self.V_NS, self.V_ZS,

    def post_fault_voltages(self):
        voltage_012 = []  # Symmetrical Component Voltages for each bus
        voltage_abc = []  # ABC Phase Voltages for each bus

        # Iterating over the system buses
        for i in range(len(self.Zbus_PS)):
            # Creating a 3x1 matrix for each bus
            bus_voltage_012 = np.zeros((3, 1), dtype=complex) # 3 here means: zero  , positive , negative sequence

            # Assigning the voltages from V_ZS, V_PS, and V_NS to the matrix
            bus_voltage_012[0, 0] = self.V_ZS[i]
            bus_voltage_012[1, 0] = self.V_PS[i]
            bus_voltage_012[2, 0] = self.V_NS[i]

            # Adding the matrix to the list of voltage matrices
            voltage_012.append(bus_voltage_012)

            # Multiplying (self.a_matrix) by the (bus_voltage_012) matrix to calculate the phase voltages:
            bus_voltage_abc = self.a_matrix.dot(bus_voltage_012)

            # I added the result to the list of phase voltage matrices:
            voltage_abc.append(bus_voltage_abc)

        # We convert all voltages to polar form
        polar_LL_voltage_012 = [[cmath.polar(v[0]) for v in bus_voltages] for bus_voltages in voltage_012]
        polar_LL_voltage_abc = [[cmath.polar(v[0]) for v in bus_voltages] for bus_voltages in voltage_abc]

        return polar_LL_voltage_012, polar_LL_voltage_abc