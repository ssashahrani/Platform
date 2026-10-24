import numpy as np

class Jacobian:
    def __init__(self, circuit):
        self.circuit = circuit

    def J1_matrix(self, ybus_polar):
        N = len(self.circuit.buses)
        J1 = np.zeros((N - 1, N - 1))

        for k in range(2, N + 1):
            Vk = self.circuit.buses[str(k)].v
            deltak = self.circuit.buses[str(k)].delta
            for j in range(2, N + 1):
                if j == k:
                    for n in range(1, N + 1):
                        Vn = self.circuit.buses[str(n)].v
                        deltan = self.circuit.buses[str(n)].delta
                        Ykn, theta_kn = ybus_polar.at[str(k), str(n)]
                        J1[k - 2, j - 2] += Vk * Vn * (-Ykn) * np.sin(deltak - deltan - np.radians(theta_kn))
                    Ykk, theta_kk = ybus_polar.at[str(k), str(k)]
                    J1[k - 2, j - 2] -= Vk ** 2 * Ykk * np.sin(np.radians(theta_kk))
                else:
                    Vn = self.circuit.buses[str(j)].v
                    deltan = self.circuit.buses[str(j)].delta
                    Ykn, theta_kn = ybus_polar.at[str(k), str(j)]
                    J1[k - 2, j - 2] = Vk * Vn * Ykn * np.sin(deltak - deltan - np.radians(theta_kn))
        return J1
        pass

    def J2_matrix(self, ybus_polar, PQ_buses):
        N = len(self.circuit.buses)
        self.nPQ= len(PQ_buses) # Number of PQ buses.
        J2 = np.zeros((N - 1, self.nPQ))

        for k in range(2, N + 1):
            Vk = self.circuit.buses[str(k)].v
            deltak = self.circuit.buses[str(k)].delta
            for j in range(1, self.nPQ+1):
                n = PQ_buses[j - 1]
                if n == k:
                    for n in range(1, N + 1):
                        Vn = self.circuit.buses[str(n)].v
                        deltan = self.circuit.buses[str(n)].delta
                        Ykn, theta_kn = ybus_polar.at[str(k), str(n)]
                        J2[k - 2, j - 1] += Vn * Ykn * np.cos(deltak - deltan - np.radians(theta_kn))
                    Ykk, theta_kk = ybus_polar.at[str(k), str(k)]
                    J2[k - 2, j - 1] += Vk * Ykk * np.cos(np.radians(theta_kk))

                else:
                    deltan = self.circuit.buses[str(n)].delta
                    Ykn, theta_kn = ybus_polar.at[str(k), str(n)]
                    J2[k - 2, j - 1] = Vk * Ykn * np.cos(deltak - deltan - np.radians(theta_kn))
        return J2
        pass

    def J3_matrix(self, ybus_polar,PQ_buses):
        N = len(self.circuit.buses)
        J3 = np.zeros((self.nPQ, N - 1))
        for k in range(1, self.nPQ + 1):
            m = PQ_buses[k - 1]
            Vm = self.circuit.buses[str(m)].v
            deltam = self.circuit.buses[str(m)].delta
            for j in range(2, N + 1):
                if j == m:
                    for n in range(1, N + 1):
                        Vn = self.circuit.buses[str(n)].v
                        deltan = self.circuit.buses[str(n)].delta
                        Ymn, theta_mn = ybus_polar.at[str(m), str(n)]
                        J3[k - 1, j - 2] += Vm * Vn * Ymn * np.cos(deltam - deltan - np.radians(theta_mn))
                    Ymm, theta_mm = ybus_polar.at[str(m), str(m)]
                    J3[k - 1, j - 2] -= Vm ** 2 * Ymm * np.cos(np.radians(theta_mm))

                else:
                    Vn = self.circuit.buses[str(j)].v
                    deltan = self.circuit.buses[str(j)].delta
                    Ymj, theta_mj = ybus_polar.at[str(m), str(j)]
                    J3[k - 1, j - 2] = -Vm * Vn * Ymj * np.cos(deltam - deltan - np.radians(theta_mj))
        return J3
        pass


    def J4_matrix(self, ybus_polar, PQ_buses):
        N = len(self.circuit.buses)
        J4 = np.zeros((self.nPQ, self.nPQ))
        for k in range(1, self.nPQ + 1):
            m = PQ_buses[k - 1]
            deltam = self.circuit.buses[str(m)].delta
            Vm = self.circuit.buses[str(m)].v
            Ymm, theta_mm = ybus_polar.at[str(m), str(m)]
            for j in range(1, self.nPQ + 1):
                n = PQ_buses[j - 1]
                if n == m:
                    for n in range(1, N + 1):
                        Vn = self.circuit.buses[str(n)].v
                        deltan = self.circuit.buses[str(n)].delta
                        Ymn, theta_mn = ybus_polar.at[str(m), str(n)]
                        J4[k - 1, j - 1] += Vn * Ymn * np.sin(deltam - deltan - np.radians(theta_mn))
                    J4[k - 1, j - 1] += -Vm * Ymm * np.sin(np.radians(theta_mm))

                else:
                    deltan = self.circuit.buses[str(n)].delta
                    Ymn, theta_mn = ybus_polar.at[str(m), str(n)]
                    J4[k - 1, j - 1] = Vm * Ymn * np.sin(deltam - deltan - np.radians(theta_mn))
        return J4
        pass

    def Full_jacobian_matrix(self, ybus_polar_matrix, PQ_buses, PV_buses):
        N = len(self.circuit.buses)
        J1_matrix = self.J1_matrix(ybus_polar_matrix)
        J2_matrix = self.J2_matrix(ybus_polar_matrix, PQ_buses)
        J3_matrix = self.J3_matrix(ybus_polar_matrix, PQ_buses)
        J4_matrix = self.J4_matrix(ybus_polar_matrix, PQ_buses)

        nPV = len(PV_buses)  # Number of PV buses
        size_Jacobian = 2 * (N - 1) - nPV
        Jacobian_matrix = np.zeros((size_Jacobian, size_Jacobian))

        Jacobian_matrix[:N - 1, :N - 1] = J1_matrix
        Jacobian_matrix[:N - 1, N - 1:size_Jacobian] = J2_matrix
        Jacobian_matrix[N - 1:size_Jacobian, :N - 1] = J3_matrix
        Jacobian_matrix[N - 1:size_Jacobian, N - 1:size_Jacobian] = J4_matrix

        return Jacobian_matrix
