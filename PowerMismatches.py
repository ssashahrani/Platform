import numpy as np

class PowerMismatches:
    def __init__(self, circuit):
        self.circuit = circuit

    def Power_mismatch(self, ybus_polar,  PV_buses):
        num_buses = len(self.circuit.buses)
        nPV = len(PV_buses)  # Number of PV buses
        Delta_P = np.zeros(num_buses - 1)
        Delta_Q = np.zeros(num_buses - 1 - nPV)


        for k in range(1, num_buses + 1):
            bus = self.circuit.buses[str(k)]
            Vk = bus.v
            deltak = bus.delta


            Pk_calculated = Vk * sum(
                self.circuit.buses[str(n)].v * ybus_polar.at[str(k), str(n)][0] *
                np.cos(deltak - self.circuit.buses[str(n)].delta - np.radians(ybus_polar.at[str(k), str(n)][1]))
                for n in range(1, num_buses + 1))


            Delta_P[k-2] = bus.pg - bus.p - Pk_calculated

        for k in range(1, num_buses):
            bus = self.circuit.buses[str(k)]
            Vk = bus.v
            deltak = bus.delta

            Qk_calculated = Vk * sum(self.circuit.buses[str(n)].v * ybus_polar.at[str(k), str(n)][0] * np.sin(deltak - self.circuit.buses[str(n)].delta - np.radians(ybus_polar.at[str(k), str(n)][1]))
                                     for n in range(1, num_buses + 1))

            Delta_Q[k-2]= bus.q - Qk_calculated
        return np.concatenate((Delta_P, Delta_Q))

