"""
adapter.py

Thin application/API adapter for the uploaded Python-Based Power System Simulator.

Purpose
-------
This file lets a desktop GUI, such as CustomTkinter, call the existing OOP
backend without importing or executing Main.py. It keeps the backend untouched
and exposes clean functions that return pandas DataFrames/dictionaries instead
of relying on printed terminal output.

Place this file in the same folder as:
    Bus.py, Circuit.py, Conductor.py, Geometry.py, Solution.py, PowerFlow.py,
    BalancedFault.py, LGFault.py, LLFault.py, LLGFault.py, etc.

Recommended GUI call:
    result = run_power_flow()
    result.bus_table
    result.line_current_table
    result.line_flow_table
    result.losses_table_pu
    result.summary
"""

from __future__ import annotations

import cmath
import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from Circuit import Circuit
from Conductor import Conductor
from Geometry import Geometry
from Solution import Solution
from PowerFlow import PowerFlow
from BalancedFault import BalancedFault
from LGFault import LGFault
from LLFault import LLFault
from LLGFault import LLGFault


Number = Union[int, float, complex]
BusID = Union[int, str]


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class PowerFlowResult:
    """Structured result returned by run_power_flow()."""

    converged: bool
    message: str
    circuit: Circuit
    solver: Optional[PowerFlow]
    bus_table: pd.DataFrame
    line_current_table: pd.DataFrame
    line_flow_table: pd.DataFrame
    losses_table_pu: pd.DataFrame
    losses_table_mva: pd.DataFrame
    summary: Dict[str, Any]
    matrices: Dict[str, pd.DataFrame] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    log: str = ""


@dataclass
class FaultResult:
    """Structured result returned by run_fault_study()."""

    fault_type: str
    faulted_bus: int
    message: str
    current_table: pd.DataFrame
    voltage_table: pd.DataFrame
    sequence_current_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    sequence_voltage_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw: Dict[str, Any] = field(default_factory=dict)
    log: str = ""


# -----------------------------------------------------------------------------
# Default 7-bus system data copied from Main.py, but organized for UI use
# -----------------------------------------------------------------------------

DEFAULT_BUS_TYPES: Dict[str, str] = {
    "1": "slack",
    "2": "PQ",
    "3": "PQ",
    "4": "PQ",
    "5": "PQ",
    "6": "PQ",
    "7": "PV",
}

# UI-friendly convention: angle is entered in degrees, then converted to radians.
DEFAULT_BUS_DATA: Dict[str, Dict[str, float]] = {
    "1": {"v": 1.0, "angle_deg": 0.0, "p": 0.0, "q": 0.0, "pg": 0.0},
    "2": {"v": 1.0, "angle_deg": 0.0, "p": 0.0, "q": 0.0, "pg": 0.0},
    "3": {"v": 1.0, "angle_deg": 0.0, "p": 1.1, "q": -0.50, "pg": 0.0},
    "4": {"v": 1.0, "angle_deg": 0.0, "p": 1.0, "q": -0.70, "pg": 0.0},
    "5": {"v": 1.0, "angle_deg": 0.0, "p": 1.0, "q": -0.65, "pg": 0.0},
    "6": {"v": 1.0, "angle_deg": 0.0, "p": 0.0, "q": 0.0, "pg": 0.0},
    "7": {"v": 1.0, "angle_deg": 0.0, "p": 0.0, "q": 0.0, "pg": 2.0},
}

DEFAULT_LINE_LENGTHS: Dict[str, float] = {
    "Line1": 25.0,
    "Line2": 10.0,
    "Line3": 20.0,
    "Line4": 20.0,
    "Line5": 10.0,
    "Line6": 35.0,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _deep_update_dict(
    base: Mapping[str, Mapping[str, Any]],
    override: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return a copy of nested dict `base`, updated by `override`."""
    merged = {str(k): dict(v) for k, v in base.items()}
    if override:
        for key, values in override.items():
            bus_key = str(key)
            if bus_key not in merged:
                merged[bus_key] = {}
            merged[bus_key].update(dict(values))
    return merged


def _complex_parts(value: complex) -> Dict[str, float]:
    """Return real, imaginary, magnitude, and angle-degree parts for a complex value."""
    z = complex(value)
    return {
        "real": float(np.real(z)),
        "imag": float(np.imag(z)),
        "magnitude": float(abs(z)),
        "angle_deg": float(np.degrees(cmath.phase(z))),
    }


def _to_complex(value: Number) -> complex:
    """Convert numeric scalar to complex while preserving complex values."""
    return complex(value)


def _polar_tuple_to_row(
    bus: int,
    label_name: str,
    label_value: str,
    polar_tuple: Tuple[float, float],
) -> Dict[str, Any]:
    """Convert a (magnitude, angle_radians) tuple to a DataFrame row."""
    magnitude, angle_rad = polar_tuple
    return {
        "Bus": bus,
        label_name: label_value,
        "Magnitude_pu": float(magnitude),
        "Angle_deg": float(np.degrees(angle_rad)),
    }


def _polar_nested_to_dataframe(
    data: Iterable[Iterable[Tuple[float, float]]],
    labels: List[str],
    label_name: str,
) -> pd.DataFrame:
    """
    Convert fault-class nested polar output to a UI-friendly table.

    Expected backend shape:
        [bus1: [(mag, rad), (mag, rad), (mag, rad)],
         bus2: [(mag, rad), ...], ...]
    """
    rows: List[Dict[str, Any]] = []
    for bus_index, bus_values in enumerate(data, start=1):
        for label, polar_tuple in zip(labels, bus_values):
            rows.append(_polar_tuple_to_row(bus_index, label_name, label, polar_tuple))
    return pd.DataFrame(rows)


def _complex_vector_to_dataframe(vector: Any, value_name: str = "Voltage") -> pd.DataFrame:
    """Convert an n x 1 complex vector to a bus-indexed DataFrame."""
    arr = np.asarray(vector).reshape(-1)
    rows: List[Dict[str, Any]] = []
    for idx, value in enumerate(arr, start=1):
        parts = _complex_parts(value)
        rows.append(
            {
                "Bus": idx,
                f"{value_name}_real": parts["real"],
                f"{value_name}_imag": parts["imag"],
                f"{value_name}_mag_pu": parts["magnitude"],
                f"{value_name}_angle_deg": parts["angle_deg"],
            }
        )
    return pd.DataFrame(rows)


def get_bus_type_lists(
    bus_types: Optional[Mapping[BusID, str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Return slack, PQ, and PV bus lists from a bus-type dictionary."""
    bus_types = {str(k): str(v) for k, v in (bus_types or DEFAULT_BUS_TYPES).items()}
    slack_bus = [int(bus_id) for bus_id, bus_type in bus_types.items() if bus_type.lower() == "slack"]
    pq_buses = [int(bus_id) for bus_id, bus_type in bus_types.items() if bus_type.upper() == "PQ"]
    pv_buses = [int(bus_id) for bus_id, bus_type in bus_types.items() if bus_type.upper() == "PV"]
    return slack_bus, pq_buses, pv_buses


def get_default_inputs() -> Dict[str, Any]:
    """Return default UI inputs for the 7-bus example system."""
    return {
        "bus_types": dict(DEFAULT_BUS_TYPES),
        "bus_data": _deep_update_dict(DEFAULT_BUS_DATA),
        "line_lengths": dict(DEFAULT_LINE_LENGTHS),
        "max_iterations": 10,
        "tolerance": 1e-4,
        "faulted_bus": 4,
        "prefault_voltage": 1.0,
        "fault_impedance": 0.0,
    }


# -----------------------------------------------------------------------------
# Case builder
# -----------------------------------------------------------------------------

def build_default_case(
    bus_data: Optional[Mapping[BusID, Mapping[str, Any]]] = None,
    line_lengths: Optional[Mapping[str, float]] = None,
    bus_types: Optional[Mapping[BusID, str]] = None,
) -> Tuple[Circuit, Dict[str, Any]]:
    """
    Build the original 7-bus case from Main.py without executing Main.py.

    Parameters
    ----------
    bus_data:
        Optional override for bus values. Use keys "v", "angle_deg", "p", "q", "pg".
        Example: {"3": {"p": 1.2, "q": -0.55}}
    line_lengths:
        Optional override for line lengths by line name.
        Example: {"Line1": 30.0}
    bus_types:
        Optional override for bus types.

    Returns
    -------
    circuit, metadata
    """
    circuit = Circuit("Network")

    # Conductor parameters: Partridge conductor from Main.py
    rd = (0.642 / 2) * (1 / 12)  # ft
    bundle_spacing = 1.5  # ft
    Ds = 0.0217  # ft
    rc = 0.385  # ohm/mile per conductor
    num_conductors = 2
    coordinates = np.array([0, 0, 19.5, 0, 39, 0])

    # Base values
    base_voltageHV = 230e3
    base_voltageLV1 = 20e3
    base_voltageLV2 = 18e3
    base_power = 100e6
    base_power_mva = base_power / 1e6

    # Grounding elements
    Grounding_R_G1 = 0
    Grounding_X_G1 = 0
    Grounding_R_T1 = 0
    Grounding_X_T1 = 1
    Grounding_R_T2 = float("inf")
    Grounding_X_T2 = float("inf")
    Grounding_R_G2 = 1
    Grounding_X_G2 = 0

    groundings = ["Delta", "Solidly Y", "Ungrounded Y", "Impedance Grounded Y"]

    # Geometry and conductor
    line_geometry = Geometry(coordinates)
    ampacity = 460 * 2
    line_conductor = Conductor(rd, bundle_spacing, Ds, rc, coordinates, num_conductors)
    line_conductor.set_base_values(base_voltageHV, base_voltageLV1, base_voltageLV2, base_power)
    ampacity_pu = line_conductor.calc_ampacity_pu(ampacity)

    lengths = dict(DEFAULT_LINE_LENGTHS)
    if line_lengths:
        lengths.update({str(k): float(v) for k, v in line_lengths.items()})

    # Normal power-flow network elements
    circuit.add_Transformer_impedance_element(
        "Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10,
        syst_base=base_power, transformer_rating=125e6, voltage_base=230e3,
        grounding_bus1=groundings[0], grounding_bus2=groundings[3],
        grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1,
    )
    _add_all_lines(circuit.add_Line_impedance_element, circuit, line_conductor, line_geometry, lengths, ampacity)
    circuit.add_Transformer_impedance_element(
        "Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12,
        syst_base=base_power, transformer_rating=200e6, voltage_base=230e3,
        grounding_bus1=groundings[2], grounding_bus2=groundings[0],
        grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2,
    )

    # Positive-sequence network
    circuit.add_generator_reactancePS_element(
        "Gen1", bus1="1", bus2="2", x_g=0.12, syst_base=base_power,
        generator_rating=90e6, voltage_base=20e3,
        grounding_R=Grounding_R_G1, grounding_X=Grounding_X_G1,
    )
    circuit.add_Transformer_impedanceF_element(
        "Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10,
        syst_base=base_power, transformer_rating=125e6, voltage_base=230e3,
        grounding_bus1=groundings[0], grounding_bus2=groundings[3],
        grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1,
    )
    _add_all_lines(circuit.add_Line_impedanceF_element, circuit, line_conductor, line_geometry, lengths, ampacity)
    circuit.add_Transformer_impedanceF_element(
        "Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12,
        syst_base=base_power, transformer_rating=200e6, voltage_base=230e3,
        grounding_bus1=groundings[2], grounding_bus2=groundings[0],
        grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2,
    )
    circuit.add_generator_reactancePS_element(
        "Gen2", bus1="7", bus2="6", x_g=0.12, syst_base=base_power,
        generator_rating=90e6, voltage_base=18e3,
        grounding_R=Grounding_R_G2, grounding_X=Grounding_X_G2,
    )

    # Negative-sequence network
    circuit.add_generator_reactanceNS_element(
        "Gen1", bus1="1", bus2="2", x_g=0.14, syst_base=base_power,
        generator_rating=90e6, voltage_base=20e3,
        grounding_R=Grounding_R_G1, grounding_X=Grounding_X_G1,
    )
    circuit.add_Transformer_impedanceNS_element(
        "Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10,
        syst_base=base_power, transformer_rating=125e6, voltage_base=230e3,
        grounding_bus1=groundings[0], grounding_bus2=groundings[3],
        grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1,
    )
    _add_all_lines(circuit.add_Line_impedanceNS_element, circuit, line_conductor, line_geometry, lengths, ampacity)
    circuit.add_Transformer_impedanceNS_element(
        "Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12,
        syst_base=base_power, transformer_rating=200e6, voltage_base=230e3,
        grounding_bus1=groundings[2], grounding_bus2=groundings[0],
        grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2,
    )
    circuit.add_generator_reactanceNS_element(
        "Gen2", bus1="7", bus2="6", x_g=0.14, syst_base=base_power,
        generator_rating=90e6, voltage_base=18e3,
        grounding_R=Grounding_R_G2, grounding_X=Grounding_X_G2,
    )

    # Zero-sequence network
    circuit.add_generator_reactanceZS_element(
        "Gen1", bus1="1", bus2="2", x_g=0.05, syst_base=base_power,
        generator_rating=100e6, voltage_base=20e3,
        grounding_R=Grounding_R_G1, grounding_X=Grounding_X_G1,
    )
    circuit.add_Transformer_impedanceZS_element(
        "Transformer1", bus1="1", bus2="2", z_t=0.085, x_r=10,
        syst_base=base_power, transformer_rating=125e6, voltage_base=230e3,
        grounding_bus1=groundings[0], grounding_bus2=groundings[3],
        grounding_R=Grounding_R_T1, grounding_X=Grounding_X_T1,
    )
    _add_all_lines(circuit.add_Line_impedanceZS_element, circuit, line_conductor, line_geometry, lengths, ampacity)
    circuit.add_Transformer_impedanceZS_element(
        "Transformer2", bus1="6", bus2="7", z_t=0.105, x_r=12,
        syst_base=base_power, transformer_rating=200e6, voltage_base=230e3,
        grounding_bus1=groundings[2], grounding_bus2=groundings[0],
        grounding_R=Grounding_R_T2, grounding_X=Grounding_X_T2,
    )
    circuit.add_generator_reactanceZS_element(
        "Gen2", bus1="7", bus2="6", x_g=0.05, syst_base=base_power,
        generator_rating=100e6, voltage_base=18e3,
        grounding_R=Grounding_R_G2, grounding_X=Grounding_X_G2,
    )

    # Add buses explicitly and set power-flow bus data.
    data = _deep_update_dict(DEFAULT_BUS_DATA, bus_data)
    for bus_id in sorted(data.keys(), key=lambda x: int(x)):
        circuit.add_bus(str(bus_id))
        values = data[str(bus_id)]
        angle_rad = float(values.get("angle_rad", np.radians(float(values.get("angle_deg", 0.0)))))
        circuit.buses[str(bus_id)].set_voltage_angle_power(
            float(values.get("v", 1.0)),
            angle_rad,
            float(values.get("p", 0.0)),
            float(values.get("q", 0.0)),
            float(values.get("pg", 0.0)),
        )

    bt = {str(k): str(v) for k, v in (bus_types or DEFAULT_BUS_TYPES).items()}
    slack_bus, pq_buses, pv_buses = get_bus_type_lists(bt)

    metadata = {
        "bus_types": bt,
        "slack_bus": slack_bus,
        "PQ_buses": pq_buses,
        "PV_buses": pv_buses,
        "line_conductor": line_conductor,
        "line_geometry": line_geometry,
        "ampacity": ampacity,
        "ampacity_pu": ampacity_pu,
        "base_current_L": line_conductor.base_current_L,
        "base_current_H1": line_conductor.base_current_H1,
        "base_current_H2": line_conductor.base_current_H2,
        "base_power": base_power,
        "base_power_MVA": base_power_mva,
        "base_voltageHV": base_voltageHV,
        "base_voltageLV1": base_voltageLV1,
        "base_voltageLV2": base_voltageLV2,
        "line_lengths": lengths,
    }
    return circuit, metadata


def _add_all_lines(add_method, circuit, line_conductor, line_geometry, lengths, ampacity) -> None:
    """Add the six transmission lines using the selected Circuit add method."""
    add_method("Line1", "2", "3", line_conductor, line_geometry, length=lengths["Line1"], ampacity=ampacity)
    add_method("Line2", "2", "4", line_conductor, line_geometry, length=lengths["Line2"], ampacity=ampacity)
    add_method("Line3", "3", "5", line_conductor, line_geometry, length=lengths["Line3"], ampacity=ampacity)
    add_method("Line4", "4", "6", line_conductor, line_geometry, length=lengths["Line4"], ampacity=ampacity)
    add_method("Line5", "5", "6", line_conductor, line_geometry, length=lengths["Line5"], ampacity=ampacity)
    add_method("Line6", "4", "5", line_conductor, line_geometry, length=lengths["Line6"], ampacity=ampacity)


# -----------------------------------------------------------------------------
# Extraction functions for UI tables/plots
# -----------------------------------------------------------------------------

def extract_bus_voltage_table(
    circuit: Circuit,
    bus_types: Optional[Mapping[BusID, str]] = None,
) -> pd.DataFrame:
    """Return bus voltages, angles, loads, and generation as a DataFrame."""
    bus_types = {str(k): str(v) for k, v in (bus_types or {}).items()}
    rows: List[Dict[str, Any]] = []
    for bus_id in sorted(circuit.buses.keys(), key=lambda x: int(x)):
        bus = circuit.buses[bus_id]
        rows.append(
            {
                "Bus": int(bus_id),
                "Type": bus_types.get(bus_id, ""),
                "Voltage_pu": float(bus.v),
                "Angle_rad": float(bus.delta),
                "Angle_deg": float(np.degrees(bus.delta)),
                "P_load_pu": float(bus.p),
                "Q_load_pu": float(bus.q),
                "P_gen_pu": float(bus.pg),
            }
        )
    return pd.DataFrame(rows)


def extract_voltage_profile(circuit: Circuit) -> Tuple[List[int], List[float]]:
    """Return x/y data for a voltage-profile plot."""
    df = extract_bus_voltage_table(circuit)
    return df["Bus"].tolist(), df["Voltage_pu"].tolist()


def line_current_info_to_dataframe(line_current_info: Mapping[Tuple[int, int], Mapping[str, Any]]) -> pd.DataFrame:
    """Convert PowerFlow.calculate_line_currents() current output to a DataFrame."""
    rows: List[Dict[str, Any]] = []
    for (bus1, bus2), currents in line_current_info.items():
        I = _to_complex(currents.get("I", 0.0))
        parts = _complex_parts(I)
        rows.append(
            {
                "From_Bus": int(bus1),
                "To_Bus": int(bus2),
                "I_pu_real": parts["real"],
                "I_pu_imag": parts["imag"],
                "I_pu_mag": parts["magnitude"],
                "I_pu_angle_deg": parts["angle_deg"],
                "Line_current_A": float(currents.get("Line_current", 0.0)),
                "Overcurrent": bool(currents.get("overcurrent", False)),
            }
        )
    return pd.DataFrame(rows)


def line_flow_info_to_dataframe(line_flow_info: Mapping[Tuple[int, int], Mapping[str, Any]]) -> pd.DataFrame:
    """Convert PowerFlow.calculate_line_currents() line-flow output to a DataFrame."""
    rows: List[Dict[str, Any]] = []
    for (bus1, bus2), flows in line_flow_info.items():
        S_pu = _to_complex(flows.get("S", 0.0))
        S_mva = _to_complex(flows.get("Line_flow", 0.0))
        pu = _complex_parts(S_pu)
        mva = _complex_parts(S_mva)
        rows.append(
            {
                "From_Bus": int(bus1),
                "To_Bus": int(bus2),
                "S_pu_real": pu["real"],
                "S_pu_imag": pu["imag"],
                "S_pu_mag": pu["magnitude"],
                "S_pu_angle_deg": pu["angle_deg"],
                "P_MW": mva["real"],
                "Q_MVAR": mva["imag"],
                "S_MVA_mag": mva["magnitude"],
                "S_MVA_angle_deg": mva["angle_deg"],
            }
        )
    return pd.DataFrame(rows)


def losses_to_dataframe(losses: Mapping[Tuple[int, int], complex], value_prefix: str) -> pd.DataFrame:
    """Convert power-loss dictionary to a DataFrame."""
    rows: List[Dict[str, Any]] = []
    for (bus1, bus2), loss in losses.items():
        parts = _complex_parts(loss)
        rows.append(
            {
                "Bus_i": int(bus1),
                "Bus_j": int(bus2),
                f"{value_prefix}_real": parts["real"],
                f"{value_prefix}_imag": parts["imag"],
                f"{value_prefix}_mag": parts["magnitude"],
                f"{value_prefix}_angle_deg": parts["angle_deg"],
            }
        )
    return pd.DataFrame(rows)


def build_solution_matrices(circuit: Circuit) -> Dict[str, pd.DataFrame]:
    """Build matrices needed for display or fault analysis."""
    solution = Solution(circuit)
    return {
        "ybus": solution.construct_ybus(),
        "ybus_positive_sequence": solution.construct_ybusSF(),
        "ybus_negative_sequence": solution.construct_ybusNS(),
        "ybus_zero_sequence": solution.construct_ybusZS(),
        "zbus_offdiag": solution.construct_zbus_offDiag(),
        "zbus_diag": solution.construct_zbus_Diag(),
        "ybus_polar": solution.ybus_to_polar(),
    }


# -----------------------------------------------------------------------------
# Main API functions for GUI
# -----------------------------------------------------------------------------

def run_power_flow(
    bus_data: Optional[Mapping[BusID, Mapping[str, Any]]] = None,
    line_lengths: Optional[Mapping[str, float]] = None,
    bus_types: Optional[Mapping[BusID, str]] = None,
    max_iterations: int = 10,
    tolerance: float = 1e-4,
    suppress_output: bool = True,
    include_matrices: bool = True,
) -> PowerFlowResult:
    """
    Build the default 7-bus case, run Newton-Raphson power flow, and return UI tables.

    This is the primary function a desktop GUI should call.
    """
    log_stream = io.StringIO()
    circuit, metadata = build_default_case(bus_data=bus_data, line_lengths=line_lengths, bus_types=bus_types)
    slack_bus = metadata["slack_bus"]
    pq_buses = metadata["PQ_buses"]
    pv_buses = metadata["PV_buses"]

    solver: Optional[PowerFlow] = None
    converged = False
    message = ""
    line_current_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    line_flow_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    power_losses_pu: Dict[Tuple[int, int], complex] = {}
    power_losses_mva: Dict[Tuple[int, int], complex] = {}
    total_losses_pu: complex = 0 + 0j

    try:
        solver = PowerFlow(
            circuit,
            metadata["ampacity_pu"],
            metadata["base_current_L"],
            metadata["base_current_H1"],
            metadata["base_current_H2"],
            slack_bus,
            pv_buses,
        )

        ctx = contextlib.redirect_stdout(log_stream) if suppress_output else contextlib.nullcontext()
        with ctx:
            converged = bool(
                solver.solve(
                    max_iterations=int(max_iterations),
                    tolerance=float(tolerance),
                    PQ_buses=pq_buses,
                    PV_buses=pv_buses,
                )
            )
            if converged:
                (
                    line_current_info,
                    line_flow_info,
                    power_losses_pu,
                    power_losses_mva,
                    total_losses_pu,
                ) = solver.calculate_line_currents()

        message = "Power flow converged." if converged else "Power flow did not converge."

    except Exception as exc:  # Return structured error for GUI instead of crashing.
        message = f"Power flow failed: {exc}"
        if not suppress_output:
            raise

    bus_table = extract_bus_voltage_table(circuit, metadata["bus_types"])
    line_current_table = line_current_info_to_dataframe(line_current_info)
    line_flow_table = line_flow_info_to_dataframe(line_flow_info)
    losses_table_pu = losses_to_dataframe(power_losses_pu, "Loss_pu")
    losses_table_mva = losses_to_dataframe(power_losses_mva, "Loss_MVA")

    total_losses_mva = total_losses_pu * metadata["base_power_MVA"]
    summary = {
        "converged": converged,
        "message": message,
        "max_iterations_requested": int(max_iterations),
        "tolerance": float(tolerance),
        "total_loss_pu_real": float(np.real(total_losses_pu)),
        "total_loss_pu_imag": float(np.imag(total_losses_pu)),
        "total_loss_MVA_real": float(np.real(total_losses_mva)),
        "total_loss_MVA_imag": float(np.imag(total_losses_mva)),
        "number_of_buses": len(circuit.buses),
        "number_of_current_records": len(line_current_info),
        "number_of_loss_records": len(power_losses_pu),
        "base_power_MVA": metadata["base_power_MVA"],
    }

    matrices = build_solution_matrices(circuit) if include_matrices else {}

    return PowerFlowResult(
        converged=converged,
        message=message,
        circuit=circuit,
        solver=solver,
        bus_table=bus_table,
        line_current_table=line_current_table,
        line_flow_table=line_flow_table,
        losses_table_pu=losses_table_pu,
        losses_table_mva=losses_table_mva,
        summary=summary,
        matrices=matrices,
        raw={
            "line_current_info": line_current_info,
            "line_flow_info": line_flow_info,
            "power_losses_pu": power_losses_pu,
            "power_losses_MVA": power_losses_mva,
            "total_losses_pu": total_losses_pu,
            "metadata": metadata,
        },
        log=log_stream.getvalue(),
    )


def run_fault_study(
    fault_type: str,
    faulted_bus: int = 4,
    prefault_voltage: float = 1.0,
    fault_impedance: Union[float, complex] = 0.0,
    bus_data: Optional[Mapping[BusID, Mapping[str, Any]]] = None,
    line_lengths: Optional[Mapping[str, float]] = None,
    bus_types: Optional[Mapping[BusID, str]] = None,
    solve_prefault_power_flow: bool = False,
    suppress_output: bool = True,
) -> FaultResult:
    """
    Run a fault study and return structured tables.

    fault_type options:
        "balanced", "3phase", "three_phase", "LG", "SLG", "LL", "LLG"
    """
    log_stream = io.StringIO()
    fault_key = fault_type.strip().lower().replace("-", "_").replace(" ", "_")

    circuit, metadata = build_default_case(bus_data=bus_data, line_lengths=line_lengths, bus_types=bus_types)

    if solve_prefault_power_flow:
        # This updates circuit bus voltages before fault calculation.
        pf_result = run_power_flow(
            bus_data=bus_data,
            line_lengths=line_lengths,
            bus_types=bus_types,
            suppress_output=suppress_output,
            include_matrices=False,
        )
        circuit = pf_result.circuit
        if str(faulted_bus) in circuit.buses:
            prefault_voltage = float(circuit.buses[str(faulted_bus)].v)

    try:
        matrices = build_solution_matrices(circuit)
        ybus_ps = matrices["ybus_positive_sequence"]
        ybus_ns = matrices["ybus_negative_sequence"]
        ybus_zs = matrices["ybus_zero_sequence"]
        Zf = complex(fault_impedance)
        Vf = float(prefault_voltage)

        ctx = contextlib.redirect_stdout(log_stream) if suppress_output else contextlib.nullcontext()
        with ctx:
            if fault_key in {"balanced", "three_phase", "3phase", "3_phase", "threephase"}:
                fault = BalancedFault(circuit, ybus_ps)
                fault_current = fault.fault_current(int(faulted_bus), Vf, Zf)
                post_fault_voltages = fault.post_fault_voltages()
                current_parts = _complex_parts(fault_current)
                current_table = pd.DataFrame([
                    {
                        "Faulted_Bus": int(faulted_bus),
                        "Fault_Type": "Balanced three-phase",
                        "I_fault_real_pu": current_parts["real"],
                        "I_fault_imag_pu": current_parts["imag"],
                        "I_fault_mag_pu": current_parts["magnitude"],
                        "I_fault_angle_deg": current_parts["angle_deg"],
                    }
                ])
                voltage_table = _complex_vector_to_dataframe(post_fault_voltages, value_name="Voltage")
                return FaultResult(
                    fault_type="balanced",
                    faulted_bus=int(faulted_bus),
                    message="Balanced three-phase fault completed.",
                    current_table=current_table,
                    voltage_table=voltage_table,
                    raw={"fault_object": fault, "post_fault_voltages": post_fault_voltages},
                    log=log_stream.getvalue(),
                )

            if fault_key in {"lg", "slg", "line_ground", "line_to_ground", "single_line_to_ground"}:
                fault = LGFault(circuit, ybus_ps, ybus_ns, ybus_zs)
                current_012, current_abc = fault.LGfault_current(int(faulted_bus), Vf, Zf)
                voltage_012, voltage_abc = fault.post_fault_voltages()
                current_table = _polar_nested_to_dataframe(current_abc, ["A", "B", "C"], "Phase")
                voltage_table = _polar_nested_to_dataframe(voltage_abc, ["A", "B", "C"], "Phase")
                sequence_current_table = _polar_nested_to_dataframe(current_012, ["0", "1", "2"], "Sequence")
                sequence_voltage_table = _polar_nested_to_dataframe(voltage_012, ["0", "1", "2"], "Sequence")
                return FaultResult(
                    fault_type="LG",
                    faulted_bus=int(faulted_bus),
                    message="Single line-to-ground fault completed.",
                    current_table=current_table,
                    voltage_table=voltage_table,
                    sequence_current_table=sequence_current_table,
                    sequence_voltage_table=sequence_voltage_table,
                    raw={"fault_object": fault, "current_abc": current_abc, "voltage_abc": voltage_abc},
                    log=log_stream.getvalue(),
                )

            if fault_key in {"ll", "line_line", "line_to_line"}:
                fault = LLFault(circuit, ybus_ps, ybus_ns, ybus_zs)
                current_012, current_abc = fault.LLfault_current(int(faulted_bus), Vf, Zf)
                voltage_012, voltage_abc = fault.post_fault_voltages()
                current_table = _polar_nested_to_dataframe(current_abc, ["A", "B", "C"], "Phase")
                voltage_table = _polar_nested_to_dataframe(voltage_abc, ["A", "B", "C"], "Phase")
                sequence_current_table = _polar_nested_to_dataframe(current_012, ["0", "1", "2"], "Sequence")
                sequence_voltage_table = _polar_nested_to_dataframe(voltage_012, ["0", "1", "2"], "Sequence")
                return FaultResult(
                    fault_type="LL",
                    faulted_bus=int(faulted_bus),
                    message="Line-to-line fault completed.",
                    current_table=current_table,
                    voltage_table=voltage_table,
                    sequence_current_table=sequence_current_table,
                    sequence_voltage_table=sequence_voltage_table,
                    raw={"fault_object": fault, "current_abc": current_abc, "voltage_abc": voltage_abc},
                    log=log_stream.getvalue(),
                )

            if fault_key in {"llg", "line_line_ground", "line_to_line_to_ground"}:
                fault = LLGFault(circuit, ybus_ps, ybus_ns, ybus_zs)
                current_012, current_abc = fault.LLGfault_current(int(faulted_bus), Vf, Zf)
                voltage_012, voltage_abc = fault.post_fault_voltages()
                current_table = _polar_nested_to_dataframe(current_abc, ["A", "B", "C"], "Phase")
                voltage_table = _polar_nested_to_dataframe(voltage_abc, ["A", "B", "C"], "Phase")
                sequence_current_table = _polar_nested_to_dataframe(current_012, ["0", "1", "2"], "Sequence")
                sequence_voltage_table = _polar_nested_to_dataframe(voltage_012, ["0", "1", "2"], "Sequence")
                return FaultResult(
                    fault_type="LLG",
                    faulted_bus=int(faulted_bus),
                    message="Line-to-line-to-ground fault completed.",
                    current_table=current_table,
                    voltage_table=voltage_table,
                    sequence_current_table=sequence_current_table,
                    sequence_voltage_table=sequence_voltage_table,
                    raw={"fault_object": fault, "current_abc": current_abc, "voltage_abc": voltage_abc},
                    log=log_stream.getvalue(),
                )

        raise ValueError(
            "Unsupported fault_type. Use one of: 'balanced', 'LG', 'LL', or 'LLG'."
        )

    except Exception as exc:
        if not suppress_output:
            raise
        return FaultResult(
            fault_type=fault_type,
            faulted_bus=int(faulted_bus),
            message=f"Fault study failed: {exc}",
            current_table=pd.DataFrame(),
            voltage_table=pd.DataFrame(),
            raw={"metadata": metadata},
            log=log_stream.getvalue(),
        )


# -----------------------------------------------------------------------------
# Convenience function for quick manual test
# -----------------------------------------------------------------------------

def smoke_test() -> Dict[str, Any]:
    """Run a quick power-flow and balanced-fault test."""
    pf = run_power_flow(suppress_output=True, include_matrices=False)
    fault = run_fault_study("balanced", suppress_output=True)
    return {
        "power_flow_message": pf.message,
        "bus_table_shape": pf.bus_table.shape,
        "line_current_table_shape": pf.line_current_table.shape,
        "balanced_fault_message": fault.message,
        "fault_voltage_table_shape": fault.voltage_table.shape,
    }


if __name__ == "__main__":
    print(smoke_test())
