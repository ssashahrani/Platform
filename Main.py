from Circuit import Circuit
from Conductor import Conductor
from Geometry import Geometry
from Solution import Solution
from Jacobian import Jacobian
from PowerMismatches import PowerMismatches
from PowerFlow import PowerFlow
from BalancedFault import BalancedFault
from LGFault import LGFault
from LLFault import LLFault
from LLGFault import LLGFault
import numpy as np
import pandas as pd
import cmath


np.set_printoptions(suppress=True, precision=3)
np.set_printoptions(linewidth=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

circuit = Circuit("Network")

# "Partridge" Conductor parameters:
rd = (0.642 / 2) * (1 / 12)  # The outside radius of the stranded conductor.
bundle_spacing = 1.5  # in feet
Ds = 0.0217  # GMR for stranded conductor, in feet
rc= 0.385 # Resistance per conductor per mile
num_conductors = 2
coordinates = np.array([0, 0, 19.5, 0, 39, 0])

# Base values for voltage and power
base_voltageHV = 230e3  # Base voltage in volts
base_voltageLV1 = 20e3
base_voltageLV2 = 18e3
base_power = 100e6    # Base power in VA

#Grounding elements: Resistance and Reactance values:
Grounding_R_G1= 0 # in ohm
Grounding_X_G1= 0 # in ohm
Grounding_R_T1= 0 # in ohm
Grounding_X_T1= 1 # in ohm
Grounding_R_T2= float('inf') # in ohm (Ungrounded wye)
Grounding_X_T2= float('inf') # in ohm (Ungrounded wye)
Grounding_R_G2= 1 # in ohm
Grounding_X_G2= 0 # in ohm

groundings = ['Delta', 'Solidly Y', 'Ungrounded Y', 'Impedance Grounded Y']


# A geometry instance:
line_geometry = Geometry(coordinates)
ampacity = 460*2 # Two bundle conductor with 460 A current carrying capacity
# A conductor instance:
line_conductor = Conductor(rd, bundle_spacing, Ds, rc, coordinates, num_conductors)
line_conductor.set_base_values(base_voltageHV, base_voltageLV1, base_voltageLV2 , base_power)
# The conductor's ampacity instance:
ampacity_pu = line_conductor.calc_ampacity_pu(ampacity)
# Adding line & transformer elements to the circuit:

circuit.add_Transformer_impedance_element("Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10, syst_base=100e6,transformer_rating=125e6, voltage_base=230e3, grounding_bus1=groundings[0], grounding_bus2=groundings[3], grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1)
circuit.add_Line_impedance_element("Line1", "2", "3", line_conductor, line_geometry, length=25, ampacity=ampacity)
circuit.add_Line_impedance_element("Line2", "2", "4", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedance_element("Line3", "3", "5", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedance_element("Line4", "4", "6", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedance_element("Line5", "5", "6", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedance_element("Line6", "4", "5", line_conductor, line_geometry, length=35, ampacity=ampacity)
circuit.add_Transformer_impedance_element("Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12, syst_base=100e6, transformer_rating=200e6, voltage_base=230e3, grounding_bus1=groundings[2], grounding_bus2=groundings[0], grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2)

"""For Positive Sequence"""
circuit.add_generator_reactancePS_element("Gen1", bus1="1", bus2="2", x_g=0.12, syst_base=100e6, generator_rating=90e6, voltage_base=20e3, grounding_R=Grounding_R_G1, grounding_X=Grounding_X_G1)
circuit.add_Transformer_impedanceF_element("Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10, syst_base=100e6,transformer_rating=125e6, voltage_base=230e3, grounding_bus1=groundings[0], grounding_bus2=groundings[3], grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1)
circuit.add_Line_impedanceF_element("Line1", "2", "3", line_conductor, line_geometry, length=25, ampacity=ampacity)
circuit.add_Line_impedanceF_element("Line2", "2", "4", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedanceF_element("Line3", "3", "5", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedanceF_element("Line4", "4", "6", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedanceF_element("Line5", "5", "6", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedanceF_element("Line6", "4", "5", line_conductor, line_geometry, length=35, ampacity=ampacity)
circuit.add_Transformer_impedanceF_element("Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12, syst_base=100e6, transformer_rating=200e6, voltage_base=230e3, grounding_bus1=groundings[2], grounding_bus2=groundings[0], grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2)
circuit.add_generator_reactancePS_element("Gen2", bus1="7", bus2="6", x_g=0.12, syst_base=100e6, generator_rating=90e6, voltage_base=18e3, grounding_R=Grounding_R_G2, grounding_X=Grounding_X_G2)

"""For Negative Sequence"""
circuit.add_generator_reactanceNS_element("Gen1", bus1="1", bus2="2", x_g=0.14, syst_base=100e6, generator_rating=90e6, voltage_base=20e3, grounding_R=Grounding_R_G1, grounding_X=Grounding_X_G1)
circuit.add_Transformer_impedanceNS_element("Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10, syst_base=100e6,transformer_rating=125e6, voltage_base=230e3, grounding_bus1=groundings[0], grounding_bus2=groundings[3], grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1)
circuit.add_Line_impedanceNS_element("Line1", "2", "3", line_conductor, line_geometry, length=25, ampacity=ampacity)
circuit.add_Line_impedanceNS_element("Line2", "2", "4", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedanceNS_element("Line3", "3", "5", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedanceNS_element("Line4", "4", "6", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedanceNS_element("Line5", "5", "6", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedanceNS_element("Line6", "4", "5", line_conductor, line_geometry, length=35, ampacity=ampacity)
circuit.add_Transformer_impedanceNS_element("Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12, syst_base=100e6, transformer_rating=200e6, voltage_base=230e3, grounding_bus1=groundings[2], grounding_bus2=groundings[0], grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2)
circuit.add_generator_reactanceNS_element("Gen2", bus1="7", bus2="6", x_g=0.14, syst_base=100e6, generator_rating=90e6, voltage_base=18e3, grounding_R=Grounding_R_G2, grounding_X=Grounding_X_G2)

"""For Zero Sequence"""
circuit.add_generator_reactanceZS_element("Gen1", bus1="1", bus2="2", x_g=0.05, syst_base=100e6, generator_rating=100e6, voltage_base=20e3, grounding_R=Grounding_R_G1, grounding_X=Grounding_X_G1)
circuit.add_Transformer_impedanceZS_element("Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10, syst_base=100e6,transformer_rating=125e6, voltage_base=230e3, grounding_bus1=groundings[0], grounding_bus2=groundings[3], grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1)
circuit.add_Line_impedanceZS_element("Line1", "2", "3", line_conductor, line_geometry, length=25, ampacity=ampacity)
circuit.add_Line_impedanceZS_element("Line2", "2", "4", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedanceZS_element("Line3", "3", "5", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedanceZS_element("Line4", "4", "6", line_conductor, line_geometry, length=20, ampacity=ampacity)
circuit.add_Line_impedanceZS_element("Line5", "5", "6", line_conductor, line_geometry, length=10, ampacity=ampacity)
circuit.add_Line_impedanceZS_element("Line6", "4", "5", line_conductor, line_geometry, length=35, ampacity=ampacity)
circuit.add_Transformer_impedanceZS_element("Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12, syst_base=100e6, transformer_rating=200e6, voltage_base=230e3, grounding_bus1=groundings[2], grounding_bus2=groundings[0], grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2)
circuit.add_generator_reactanceZS_element("Gen2", bus1="7", bus2="6", x_g=0.05, syst_base=100e6, generator_rating=100e6, voltage_base=18e3, grounding_R=Grounding_R_G2, grounding_X=Grounding_X_G2)

circuit.add_bus("1")
circuit.add_bus("2")
circuit.add_bus("3")
circuit.add_bus("4")
circuit.add_bus("5")
circuit.add_bus("6")
circuit.add_bus("7")

# Adding the per-unit of: Bus voltage(V, Delta); loads (P, Q) and Generator power (pg), respectively:
circuit.buses["1"].set_voltage_angle_power(1, 0, 0, 0, 0)
circuit.buses["2"].set_voltage_angle_power(1, 0, 0, 0, 0)
circuit.buses["3"].set_voltage_angle_power(1, 0, 1.1, -0.50, 0)
circuit.buses["4"].set_voltage_angle_power(1, 0, 1.0, -0.70, 0)
circuit.buses["5"].set_voltage_angle_power(1, 0, 1.0, -0.65, 0)
circuit.buses["6"].set_voltage_angle_power(1, 0, 0, 0, 0)
circuit.buses["7"].set_voltage_angle_power(1, 0, 0, 0, 2)


# A solution instance:
solution = Solution(circuit)

# Y-bus matrix construction:
ybus_matrix = solution.construct_ybus()

# The construction of The positive-sequence bus admittance matrix Under Fault Conditions:
ybus_matrix_F = solution.construct_ybusSF()
# The construction of The Negative-sequence bus admittance matrix Under Fault Conditions:
ybus_matrix_NS = solution.construct_ybusNS()
# The construction of The Zero-sequence bus admittance matrix Under Fault Conditions:
ybus_matrix_ZS = solution.construct_ybusZS()



zbus_offDiag_matrix = solution.construct_zbus_offDiag()
zbus_Diag_matrix = solution.construct_zbus_Diag()
ybus_polar_matrix = solution.ybus_to_polar()

# # The Y-bus matrix:
print("Y-bus Matrix:\n", ybus_matrix)
print("\nY-bus Matrix in Polar Form:\n", ybus_polar_matrix)

bus_types = {'1': 'slack', '2': 'PQ', '3': 'PQ', '4': 'PQ', '5': 'PQ', '6': 'PQ', '7': 'PV'}
slack_bus = [int(bus_id) for bus_id, bus_type in bus_types.items() if bus_type == 'slack']
PQ_buses = [int(bus_id) for bus_id, bus_type in bus_types.items() if bus_type == 'PQ']
PV_buses = [int(bus_id) for bus_id, bus_type in bus_types.items() if bus_type == 'PV']


jacobian = Jacobian(circuit)
Jacobian_matrix = jacobian.Full_jacobian_matrix(ybus_polar_matrix, PQ_buses, PV_buses)


power_mismatches = PowerMismatches(circuit)
mismatches = power_mismatches.Power_mismatch(ybus_polar_matrix, PV_buses)


solver = PowerFlow(circuit, ampacity_pu, line_conductor.base_current_L, line_conductor.base_current_H1, line_conductor.base_current_H2, slack_bus, PV_buses)

if solver.solve(max_iterations=10, tolerance=0.0001, PQ_buses=PQ_buses, PV_buses=PV_buses):
    for bus_id, bus in circuit.buses.items():
        print(f"Bus {bus_id}: Voltage = {bus.v:.4f} pu, Angle = {np.degrees(bus.delta):.2f} Degrees")
else:
    print("Power flow solution not found.")
print("\n")

line_current_info, line_flow_info, power_losses_pu, power_losses, Total_losses= solver.calculate_line_currents()


header = "{:<20} {:<30} {:<20} {:>34}"
print(header.format("Current Direction", "Current in pu", "Current in Ampere", "Overcurrent status:"))
line_format = "{:<30} {:>15} {:>15} {:>30}"

for (bus1, bus2), currents in line_current_info.items():
        current_direction = f"Bus {bus1} → Bus {bus2}"
        current_pu = currents.get("I")
        current_Amp = currents.get("Line_current")
        overcurrent_status = currents.get("overcurrent", False)
        sign = "-" if current_pu.imag < 0 else "+"
        current_in_pu = "{:.4f} {} {:.4f}j".format(current_pu.real, sign, abs(current_pu.imag))
        current_ampere = "{:.2f} A".format(currents['Line_current'])
        overcurrent_status = "True" if currents['overcurrent'] else "False"

        line = "{:<20} {:<30} {:<20} {:>20}"
        print(line.format(current_direction, current_in_pu, current_ampere, overcurrent_status))

print("\n")
print("{:^20} {:^30} {:^20}".format("Line Flows Direction", "Line Flows in pu", "Line Flows in MVA"))
for (bus1, bus2), flows in line_flow_info.items():
        flow_direction = f"Bus {bus1} → Bus {bus2}"
        Q_PV_pu = "{:.4f} + j{:.4f}".format(flows['S'].real, flows['S'].imag) if flows[
                                                                                     'S'].imag >= 0 else "{:.4f} - j{:.4f}".format(
            flows['S'].real, abs(flows['S'].imag))


        Q_PV = "{:.0f} + j{:.0f}".format(flows["Line_flow"].real, flows["Line_flow"].imag) \
            if flows["Line_flow"].imag >= 0 else "{:.0f} - j{:.0f}".format(
            flows["Line_flow"].real, abs(flows["Line_flow"].imag))
        print("{:^20} {:^30} {:^20}".format(flow_direction, Q_PV_pu, Q_PV))


double_arrow = '\u21C4'
print("\nPower Losses in PU:")
for (bus1, bus2), loss in power_losses_pu.items():
    loss_formatted = "{:.4f}".format(loss.real)
    if loss.imag < 0:
        loss_formatted += " - j{:.4f}".format(abs(loss.imag))
    else:
        loss_formatted += " + j{:.4f}".format(loss.imag)
    print(f"Bus {bus1} {double_arrow} Bus {bus2}:  {loss_formatted} pu")

print("\nPower Losses in MVA:")
for (bus1, bus2), loss in power_losses.items():
    loss_formatted = "{:.4f}".format(loss.real)
    if loss.imag < 0:
        loss_formatted += " - j{:.4f}".format(abs(loss.imag))
    else:
        loss_formatted += " + j{:.4f}".format(loss.imag)
    print(f"Bus {bus1} {double_arrow} Bus {bus2}:  {loss_formatted} MVA")
print("\n\n"f"Total Losses = {Total_losses.real:.5f} + j{Total_losses.imag:.5f} pu")

print("\n")


# To print the real and reactive power injected into the slack bus and the reactive power injected into the PV bus, here I am going to loop through the line flows and only print those that originate from the slack bus or a PV bus
for (bus1, bus2), flows in line_flow_info.items():
    # Checking if bus1 is the slack bus
    if int(bus1) in slack_bus:
        # P,Q_slack in per unit (pu)
        P_slack_pu = "{:.4f}".format(abs(flows['S'].real))
        Q_slack_pu = "{:.4f}".format(abs(flows['S'].imag))
        # P,Q_slack in MVA
        P_slack = "{:.4f}".format(abs(flows["Line_flow"].real))
        Q_slack = "{:.4f}".format(abs(flows["Line_flow"].imag))

        print("Real power injected into the slack bus (per unit)       : ", P_slack_pu, "PU")
        print("Reactive power injected into the slack bus (per unit)   : ", Q_slack_pu, "PU")
        print("Real power injected into the slack bus (MW)           : ", P_slack, "MW")
        print("Reactive power injected into the slack bus (MVAR)       : ", Q_slack, "MVAR")
print("\n")

for (bus1, bus2), flows in line_flow_info.items():
    # Checking if bus1 is a PV bus
    if int(bus1) in PV_buses:
        # Q_PV in per unit (pu)
        Q_PV_pu = "{:.4f}".format(abs(flows['S'].imag))
        # Q_PV in MVA
        Q_PV = "{:.4f}".format(abs(flows["Line_flow"].imag))

        print("Reactive power injected into the PV bus (per unit)     : ", Q_PV_pu, "PU")
        print("Reactive power injected into the PV bus (MVAR)         : ", Q_PV, "MVAR")
print("\n")


BalancedFault   = BalancedFault(circuit, ybus_matrix_F)
LGFault = LGFault(circuit, ybus_matrix_F, ybus_matrix_NS, ybus_matrix_ZS)
LLFault = LLFault(circuit, ybus_matrix_F, ybus_matrix_NS, ybus_matrix_ZS)
LLGFault = LLGFault(circuit, ybus_matrix_F, ybus_matrix_NS, ybus_matrix_ZS)
print("The positive-sequence bus admittance matrix Under Fault Conditions:\n", ybus_matrix_F)
print("\nThe Negative-sequence bus admittance matrix Under Fault Conditions:\n", ybus_matrix_NS)
print("\nThe Zero-sequence bus admittance matrix Under Fault Conditions:\n", ybus_matrix_ZS)

# Here I select Bus 4 to be faulted with pre-fault voltage of Vf=1.0 pu.
faulted_bus = 4
Vf = 1.0  #
Zf = 0 #  I assume zero fault impedance; that is, the short circuit is a solid or “bolted” fault

## Balanced fault Calculations
print("\n")
print("Balanced fault Calculations at Bus 4:")
Fault_Current = BalancedFault.fault_current(faulted_bus, Vf, Zf)
V_post_fault = BalancedFault.post_fault_voltages()

print("Post-fault bus voltages:")
for i, voltage in enumerate(V_post_fault):
    if voltage.size == 1:
        real_part = voltage.real if np.isscalar(voltage.real) else voltage.real[0]
        imag_part = voltage.imag if np.isscalar(voltage.imag) else voltage.imag[0]
        imag_sign = '-' if imag_part < 0 else '+'
        voltage_str = f"{real_part:.4f}{imag_sign}j{abs(imag_part):.4f}"
        print(f"Bus {i + 1}: Voltage = {voltage_str} pu")
print("\n")
print("Post-fault bus voltages (Polar Form):")
for i, voltage in enumerate(V_post_fault):
    # Here I am extracting the scalar value
    voltage_scalar = voltage.item()  # Here I am getting the scalar complex number from the array
    magnitude = abs(voltage_scalar)
    angle = cmath.phase(voltage_scalar)
    angle_degrees = np.degrees(angle)
    print(f"Bus {i + 1}: Voltage = {magnitude:.4f} ∠ {angle_degrees:.2f}°")
np.savetxt('ybus_matrix_F.txt', ybus_matrix_F, fmt='%.18e', delimiter=',')

print("\n")

## Unbalanced fault Calculations:
# (Single Line to Ground Fault):
print("Unbalanced fault Calculations:")
print("\nSingle Line-to-Ground Fault at Bus 4:")
polar_LLG_current_012, polar_LLG_current_abc = LGFault.LGfault_current(faulted_bus, Vf, Zf)
polar_LLG_voltages_012, polar_LLG_voltages_abc = LGFault.post_fault_voltages()

# The polar form of the phase Currents:
print("LG Fault Currents (ABC Components) in polar form:")
for i, current_matrix in enumerate(polar_LLG_current_abc):
    print(f"LG Fault Currents (abc_current) at Bus {i+1}:")
    for amp in current_matrix:
        print(f"{amp[0]:.4f} ∠ {np.degrees(amp[1]):.2f}° pu")
    print()

# The polar form of the phase voltages:
print("Post-LGfault bus voltages (ABC Components) in polar form:")
for i, voltage_matrix in enumerate(polar_LLG_voltages_abc):
    print(f"Post-LGfault (abc_Voltage) at Bus {i+1}:")
    for v in voltage_matrix:
        print(f"{v[0]:.4f} ∠ {np.degrees(v[1]):.2f}° pu")
    print()

print("\n")

## Line - Line Fault Calculations
print("Line - Line Fault at Bus 4:")
polar_LLG_current_012, polar_LLG_current_abc = LLFault.LLfault_current(faulted_bus, Vf, Zf)
polar_LLG_voltages_012, polar_LLG_voltages_abc = LLFault.post_fault_voltages()

# The polar form of the phase Currents:
print("LL Fault Currents (ABC Components) in polar form:")
for i, current_matrix in enumerate(polar_LLG_current_abc):
    print(f"LL Fault Currents (abc_current) at Bus {i+1}:")
    for amp in current_matrix:
        print(f"{amp[0]:.4f} ∠ {np.degrees(amp[1]):.2f}° pu")
    print()

# The polar form of the phase voltages:
print("Post-LLfault bus voltages (ABC Components) in polar form:")
for i, voltage_matrix in enumerate(polar_LLG_voltages_abc):
    print(f"Post-LLfault (abc_Voltage) at Bus {i+1}:")
    for v in voltage_matrix:
        print(f"{v[0]:.4f} ∠ {np.degrees(v[1]):.2f}° pu")
    print()

print("\n")

## Line - Line - Ground Fault Calculations
print("Line - Line - Ground Fault at Bus 4:")
polar_LLG_current_012, polar_LLG_current_abc = LLGFault.LLGfault_current(faulted_bus, Vf, Zf)
polar_LLG_voltages_012, polar_LLG_voltages_abc = LLGFault.post_fault_voltages()

# The polar form of the phase Currents:
print("LLG Fault Currents (ABC Components) in polar form:")
for i, current_matrix in enumerate(polar_LLG_current_abc):
    print(f"LLG Fault Currents (abc_current) at Bus {i+1}:")
    for amp in current_matrix:
        print(f"{amp[0]:.4f} ∠ {np.degrees(amp[1]):.2f}° pu")
    print()

# The polar form of the phase voltages:
print("Post-LLGfault bus voltages (ABC Components) in polar form:")
for i, voltage_matrix in enumerate(polar_LLG_voltages_abc):
    print(f"Post-LLGfault (abc_Voltage) at Bus {i+1}:")
    for v in voltage_matrix:
        print(f"{v[0]:.4f} ∠ {np.degrees(v[1]):.2f}° pu")
    print()

