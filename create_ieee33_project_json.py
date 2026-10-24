"""
create_ieee33_project_json.py

Generate an IEEE/Baran-Wu 33-bus radial feeder project JSON compatible with
project_model.py and the v23 reliability dashboard.

Run:
    python create_ieee33_project_json.py

Output:
    ieee33_project_template.json
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_KV = 12.66
BASE_MVA = 10.0
BASE_Z_OHM = (BASE_KV ** 2) / BASE_MVA

loads_kw_kvar = {
    1:(0,0), 2:(100,60), 3:(90,40), 4:(120,80), 5:(60,30), 6:(60,20),
    7:(200,100), 8:(200,100), 9:(60,20), 10:(60,20), 11:(45,30),
    12:(60,35), 13:(60,35), 14:(120,80), 15:(60,10), 16:(60,20),
    17:(60,20), 18:(90,40), 19:(90,40), 20:(90,40), 21:(90,40),
    22:(90,40), 23:(90,50), 24:(420,200), 25:(420,200), 26:(60,25),
    27:(60,25), 28:(60,20), 29:(120,70), 30:(200,600), 31:(150,70),
    32:(210,100), 33:(60,40),
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

project = {
    "settings": {
        "name": "IEEE 33-Bus Radial Distribution Feeder",
        "base_power_mva": BASE_MVA,
        "frequency_hz": 60.0,
        "default_base_current_a": 1.0,
        "default_ampacity_pu": 10.0,
        "notes": "IEEE/Baran-Wu 33-bus radial feeder template for FBS and Phase-1 reliability testing. Loads are converted from kW/kVAr to per-unit on 10 MVA base. Inductive Q loads use the simulator's negative q_load_pu convention.",
    },
    "buses": [],
    "branches": [],
    "sequence_sources": [],
}

for bus in range(1, 34):
    p_kw, q_kvar = loads_kw_kvar[bus]
    project["buses"].append({
        "bus_id": str(bus),
        "bus_type": "Slack" if bus == 1 else "PQ",
        "voltage_pu": 1.0,
        "angle_deg": 0.0,
        "p_load_pu": p_kw / 1000.0 / BASE_MVA,
        "q_load_pu": -q_kvar / 1000.0 / BASE_MVA,
        "p_gen_pu": 0.0,
        "base_kv": BASE_KV,
    })

for fb, tb, r_ohm, x_ohm in branches_ohm:
    project["branches"].append({
        "name": f"L{fb}-{tb}",
        "from_bus": str(fb),
        "to_bus": str(tb),
        "r_pu": r_ohm / BASE_Z_OHM,
        "x_pu": x_ohm / BASE_Z_OHM,
        "b_pu": 0.0,
        "ampacity_pu": 10.0,
        "element_type": "Line",
    })

Path("ieee33_project_template.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
print("Wrote ieee33_project_template.json")
