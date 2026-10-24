"""
compare_ieee33_with_pandapower_bfsw.py

Optional validation script for the IEEE/Baran-Wu 33-bus radial feeder.
It builds the same 33-bus system in pandapower and runs:

    pp.runpp(net, algorithm="bfsw")

Install first, if needed:
    pip install pandapower

Expected comparison target from this simulator's FBS test JSON:
    min voltage ≈ 0.91309 pu at Bus 18
    total real loss ≈ 0.20268 MW on 10 MVA base
"""

import pandapower as pp

BASE_KV = 12.66
BASE_MVA = 10.0

loads = {
    1:(0,0), 2:(100,60), 3:(90,40), 4:(120,80), 5:(60,30), 6:(60,20),
    7:(200,100), 8:(200,100), 9:(60,20), 10:(60,20), 11:(45,30),
    12:(60,35), 13:(60,35), 14:(120,80), 15:(60,10), 16:(60,20),
    17:(60,20), 18:(90,40), 19:(90,40), 20:(90,40), 21:(90,40),
    22:(90,40), 23:(90,50), 24:(420,200), 25:(420,200), 26:(60,25),
    27:(60,25), 28:(60,20), 29:(120,70), 30:(200,600), 31:(150,70),
    32:(210,100), 33:(60,40)
}

branches_ohm = [
    (1,2,0.0922,0.0470), (2,3,0.4930,0.2511), (3,4,0.3660,0.1864),
    (4,5,0.3811,0.1941), (5,6,0.8190,0.7070), (6,7,0.1872,0.6188),
    (7,8,0.7114,0.2351), (8,9,1.0300,0.7400), (9,10,1.0440,0.7400),
    (10,11,0.1966,0.0650), (11,12,0.3744,0.1238), (12,13,1.4680,1.1550),
    (13,14,0.5416,0.7129), (14,15,0.5910,0.5260), (15,16,0.7463,0.5450),
    (16,17,1.2890,1.7210), (17,18,0.7320,0.5740), (2,19,0.1640,0.1565),
    (19,20,1.5042,1.3554), (20,21,0.4095,0.4784), (21,22,0.7089,0.9373),
    (3,23,0.4512,0.3083), (23,24,0.8980,0.7091), (24,25,0.8960,0.7011),
    (6,26,0.2030,0.1034), (26,27,0.2842,0.1447), (27,28,1.0590,0.9337),
    (28,29,0.8042,0.7006), (29,30,0.5075,0.2585), (30,31,0.9744,0.9630),
    (31,32,0.3105,0.3619), (32,33,0.3410,0.5302),
]

net = pp.create_empty_network(sn_mva=BASE_MVA)
pp_buses = {}
for bus in range(1, 34):
    pp_buses[bus] = pp.create_bus(net, vn_kv=BASE_KV, name=f"Bus {bus}")

pp.create_ext_grid(net, bus=pp_buses[1], vm_pu=1.0, va_degree=0.0, name="Slack")

for bus, (p_kw, q_kvar) in loads.items():
    if bus == 1:
        continue
    pp.create_load(
        net,
        bus=pp_buses[bus],
        p_mw=p_kw / 1000.0,
        q_mvar=q_kvar / 1000.0,
        name=f"Load {bus}",
    )

for fb, tb, r_ohm, x_ohm in branches_ohm:
    pp.create_line_from_parameters(
        net,
        from_bus=pp_buses[fb],
        to_bus=pp_buses[tb],
        length_km=1.0,
        r_ohm_per_km=r_ohm,
        x_ohm_per_km=x_ohm,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
        name=f"L{fb}-{tb}",
    )

pp.runpp(net, algorithm="bfsw", tolerance_mva=1e-10, max_iteration=100)

print("pandapower converged:", net.converged)
print("Minimum voltage pu:", net.res_bus.vm_pu.min())
print("Minimum-voltage bus:", net.res_bus.vm_pu.idxmin() + 1)
print("Total real loss MW:", net.res_line.pl_mw.sum())
print("Total reactive loss MVAr:", net.res_line.ql_mvar.sum())
print(net.res_bus[["vm_pu", "va_degree"]].head(33))
