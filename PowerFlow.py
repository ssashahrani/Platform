import numpy as np
from Circuit import Circuit
from Solution import Solution
from Jacobian import Jacobian
from PowerMismatches import PowerMismatches



class PowerFlow:
    def __init__(self, circuit: Circuit, ampacity_pu: float, base_current_L, base_current_H1, base_current_H2, slack_bus, PV_buses):
        self.circuit = circuit
        self.solution = Solution(circuit)
        self.jacobian = Jacobian(circuit)
        self.power_mismatches = PowerMismatches(circuit)
        self.ybus_matrix = self.solution.construct_ybus()
        self.zbus_offDiag_matrix = self.solution.construct_zbus_offDiag()
        self.zbus_Diag_matrix = self.solution.construct_zbus_Diag()
        self.ybus_polar_matrix = self.solution.ybus_to_polar()
        self.ampacity_pu = ampacity_pu
        self.base_current_L = base_current_L
        self.base_current_H1 = base_current_H1
        self.base_current_H2 = base_current_H2
        self.slack_bus = slack_bus
        self.PV_buses = PV_buses


    def solve(self, max_iterations, tolerance,PQ_buses, PV_buses):
        for iteration in range(1, max_iterations):
            # Calculating the Jacobian matrix and mismatches at the current operating point
            self.jacobian_matrix = self.jacobian.Full_jacobian_matrix(self.ybus_polar_matrix,PQ_buses, PV_buses)
            print("\n"f"The Jacobian_matrix in iteration {iteration} =")
            print(self.jacobian_matrix)
            self.mismatches = self.power_mismatches.Power_mismatch(self.ybus_polar_matrix, PV_buses)
            print("\n"f"The power mismatches in iteration {iteration} =")
            print(self.mismatches)

            # We check if mismatches are within the specified tolerance
            if np.max(np.abs(self.mismatches)) < tolerance:
                print("\n")
                print(f"Power flow solution found and Convergence achieved in {iteration} iterations:\n")
                return True

            # We calculate delta_x updates
            delta_x = np.linalg.solve(self.jacobian_matrix, self.mismatches)
            print("\n"f"Delta-V and Delta-delta in iteration {iteration} =")
            print(delta_x)

            # We update the voltage magnitudes and angles
            self.update_variables(delta_x)

        print("Solution did not converge.")
        return False

    def update_variables(self, delta_x):
        # Extracting the voltage angle and magnitude updates
        delta_delta = delta_x[:len(self.circuit.buses) - 1]
        delta_v = delta_x[len(self.circuit.buses) - 1:]
        N = len(self.circuit.buses)

        for k in range(2, N + 1):
            self.circuit.buses[str(k)].delta += delta_delta[k - 2]

        for k in range(2, N):
            self.circuit.buses[str(k)].v += delta_v[k - 2]

    def calculate_line_currents(self):
        line_current_info = {}
        line_flow_info = {}
        power_losses_pu ={}
        power_losses = {}
        Total_losses=0


        for i in range(len(self.zbus_offDiag_matrix)):
            for j in range(len(self.zbus_offDiag_matrix)):
                if i != j and self.zbus_offDiag_matrix.iloc[i, j] != 0:

                    Vi = self.circuit.buses[str(i + 1)].v
                    delta_i = self.circuit.buses[str(i + 1)].delta
                    Vj = self.circuit.buses[str(j + 1)].v
                    delta_j = self.circuit.buses[str(j + 1)].delta
                    V_i_complex = Vi * np.exp(1j * delta_i)
                    V_j_complex = Vj * np.exp(1j * delta_j)

                    # Calculating currents
                    I_l = self.zbus_offDiag_matrix.iloc[i, j] * (V_i_complex - V_j_complex)
                    I = I_l + self.zbus_Diag_matrix.iloc[i, j] * V_i_complex
                    conj_I = np.conj(I)
                    S = V_i_complex * conj_I

                    base_current = self.base_current_L
                    if (i + 1) in self.slack_bus:
                        base_current = self.base_current_H1
                    elif (j + 1) in self.PV_buses:
                        base_current = self.base_current_H2

                    Line_current = abs(I) * base_current
                    overcurrent = abs(I) > self.ampacity_pu

                    Line_flow = S * 100 # 100 here is the Base Power.

                    # Storing the computed currents
                    line_current_info [(i + 1, j + 1)] = {"I": I, "Line_current": Line_current,"overcurrent": overcurrent}

                    # Storing the computed line flows
                    line_flow_info [(i + 1, j + 1)] = {"S": S, "Line_flow": Line_flow}

                    # Calculating power losses if both directions have been computed
                    if (j + 1, i + 1) in line_flow_info:
                        # Use the existing flows in the forward and reverse directions
                        forward_flow_pu = line_flow_info[(i + 1, j + 1)]["S"]
                        reverse_flow_pu = line_flow_info[(j + 1, i + 1)]["S"]

                        forward_flow = line_flow_info[(i + 1, j + 1)]["Line_flow"]
                        reverse_flow = line_flow_info[(j + 1, i + 1)]["Line_flow"]

                        Line_loss_pu = forward_flow_pu + reverse_flow_pu
                        Line_loss = forward_flow + reverse_flow

                        power_losses_pu[(i + 1, j + 1)] = Line_loss_pu
                        power_losses[(i + 1, j + 1)] = Line_loss
                        Total_losses+=Line_loss_pu


        return line_current_info, line_flow_info, power_losses_pu, power_losses, Total_losses





