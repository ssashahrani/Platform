"""
adapter_project_generic.py

Generic project-driven adapter for the Python-Based Power System Simulator UI.

This version removes the fixed 7-bus case dependency for project-driven studies.
It runs power flow from the buses/branches entered in the UI, not from Main.py.

Current scope:
- Generic Newton-Raphson power flow for any numeric bus IDs, one Slack bus, any
  number of PQ/PV buses.
- Generic branch model using per-unit R, X, B values from the UI.
- Generic balanced, LG, LL, and LLG fault studies using sequence Y-bus matrices
  assembled from project branches and sequence-source shunts.
- Returns pandas DataFrames ready for CustomTkinter tables and Matplotlib plots.

Place this file beside project_model.py and the UI file.
"""

from __future__ import annotations

import cmath
import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from project_model import PowerSystemProject
except Exception:  # pragma: no cover
    PowerSystemProject = Any  # type: ignore


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class PowerFlowResult:
    converged: bool
    message: str
    circuit: Any
    solver: Any
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
# Validation
# -----------------------------------------------------------------------------

def _bus_sort_key(bus_id: str) -> int:
    return int(str(bus_id).strip())


def validate_project_generic(project: PowerSystemProject) -> Tuple[bool, List[str], List[str]]:
    """
    Validate user-entered project data without assuming a fixed 7-bus network.

    Still requires numeric unique bus IDs because the current UI fault-bus input
    and engineering tables are numeric. The IDs do not need to be sequential.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not getattr(project, "buses", None):
        return False, ["At least one bus is required."], warnings

    bus_ids = [str(b.bus_id).strip() for b in project.buses]
    if any(not x for x in bus_ids):
        errors.append("Bus IDs cannot be blank.")
    if len(bus_ids) != len(set(bus_ids)):
        errors.append("Duplicate bus IDs are not allowed.")

    numeric_bus_ids: List[int] = []
    for x in bus_ids:
        try:
            numeric_bus_ids.append(int(x))
        except ValueError:
            errors.append(f"Bus ID {x!r} must be numeric, for example 1, 2, 3.")

    type_counts = {"slack": 0, "pq": 0, "pv": 0}
    for bus in project.buses:
        t = str(bus.bus_type).strip().lower()
        if t not in type_counts:
            errors.append(f"Bus {bus.bus_id}: bus_type must be Slack, PQ, or PV.")
        else:
            type_counts[t] += 1
        try:
            if float(bus.base_kv) <= 0:
                errors.append(f"Bus {bus.bus_id}: base_kv must be positive for physical current conversion.")
            if float(bus.voltage_pu) <= 0:
                errors.append(f"Bus {bus.bus_id}: voltage_pu must be positive.")
            float(bus.angle_deg)
            float(bus.p_load_pu)
            float(bus.q_load_pu)
            float(bus.p_gen_pu)
        except Exception:
            errors.append(f"Bus {bus.bus_id}: voltage, angle, load, and generation fields must be numeric.")

    if type_counts["slack"] != 1:
        errors.append("Exactly one Slack bus is required.")

    if not getattr(project, "branches", None):
        errors.append("At least one branch/line/transformer-equivalent is required.")

    valid_bus_set = set(bus_ids)
    branch_names = []
    for branch in project.branches:
        name = str(branch.name).strip() or "<blank>"
        branch_names.append(name)
        fb = str(branch.from_bus).strip()
        tb = str(branch.to_bus).strip()
        if fb not in valid_bus_set:
            errors.append(f"Branch {name}: from_bus {fb} does not exist in the Bus table.")
        if tb not in valid_bus_set:
            errors.append(f"Branch {name}: to_bus {tb} does not exist in the Bus table.")
        if fb == tb:
            errors.append(f"Branch {name}: from_bus and to_bus cannot be the same.")
        try:
            r = float(branch.r_pu)
            x = float(branch.x_pu)
            float(branch.b_pu)
            amp = float(branch.ampacity_pu)
            if abs(r) < 1e-12 and abs(x) < 1e-12:
                errors.append(f"Branch {name}: R and X cannot both be zero.")
            if amp <= 0:
                errors.append(f"Branch {name}: ampacity_pu must be positive.")
        except Exception:
            errors.append(f"Branch {name}: R, X, B, and ampacity must be numeric.")

    if any(not bool(getattr(br, "ampacity_verified", False)) for br in project.branches):
        warnings.append(
            "Thermal loading NOT EVALUATED for branches without verified ampacity ratings. "
            "Specify physically established ratings and set ampacity_verified=true before interpreting overload results."
        )

    if len(branch_names) != len(set(branch_names)):
        warnings.append("Duplicate branch names were found. Results are still computed, but naming may be confusing.")

    for src in getattr(project, "sequence_sources", []):
        name = str(src.name).strip() or "<blank>"
        bus = str(src.bus).strip()
        if bus not in valid_bus_set:
            errors.append(f"Sequence source {name}: bus {bus} does not exist in the Bus table.")
        try:
            values = [float(src.r1_pu), float(src.x1_pu), float(src.r2_pu), float(src.x2_pu), float(src.r0_pu), float(src.x0_pu)]
            if abs(values[0]) < 1e-12 and abs(values[1]) < 1e-12:
                warnings.append(f"Sequence source {name}: positive-sequence impedance is zero; it will be treated as open for stability.")
        except Exception:
            errors.append(f"Sequence source {name}: sequence impedance values must be numeric.")

    if not getattr(project, "sequence_sources", []):
        warnings.append("No sequence sources are defined. Fault study may fail because the sequence Y-bus can be singular.")

    return len(errors) == 0, errors, warnings


# -----------------------------------------------------------------------------
# Internal project extraction
# -----------------------------------------------------------------------------

def _ordered_buses(project: PowerSystemProject) -> List[Any]:
    return sorted(project.buses, key=lambda b: _bus_sort_key(b.bus_id))


def _bus_ids(project: PowerSystemProject) -> List[str]:
    return [str(b.bus_id).strip() for b in _ordered_buses(project)]


def _bus_type_map(project: PowerSystemProject) -> Dict[str, str]:
    return {str(b.bus_id).strip(): str(b.bus_type).strip() for b in project.buses}


def _branch_lookup(project: PowerSystemProject) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, br in enumerate(project.branches, start=1):
        rows.append({
            "name": str(br.name).strip() or f"Branch{idx}",
            "from_bus": str(br.from_bus).strip(),
            "to_bus": str(br.to_bus).strip(),
            "r_pu": float(br.r_pu),
            "x_pu": float(br.x_pu),
            "b_pu": float(br.b_pu),
            "ampacity_pu": float(br.ampacity_pu),
            "ampacity_verified": bool(getattr(br, "ampacity_verified", False)),
            "element_type": str(getattr(br, "element_type", "Line")),
        })
    return rows


def _build_ybus(project: PowerSystemProject, sequence: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Build Y-bus from generic branch data and optional sequence sources."""
    ids = _bus_ids(project)
    idx = {bus_id: i for i, bus_id in enumerate(ids)}
    y = np.zeros((len(ids), len(ids)), dtype=complex)

    for br in _branch_lookup(project):
        i = idx[br["from_bus"]]
        j = idx[br["to_bus"]]
        z = complex(br["r_pu"], br["x_pu"])
        if abs(z) < 1e-12:
            raise ValueError(f"Branch {br['name']}: series impedance cannot be zero.")
        y_series = 1 / z
        y_sh = 0.5j * br["b_pu"]
        y[i, i] += y_series + y_sh
        y[j, j] += y_series + y_sh
        y[i, j] -= y_series
        y[j, i] -= y_series

    if sequence is not None:
        sequence = sequence.lower()
        for src in getattr(project, "sequence_sources", []):
            bus = str(src.bus).strip()
            if bus not in idx:
                continue
            if sequence in {"positive", "ps", "1"}:
                z = complex(float(src.r1_pu), float(src.x1_pu))
            elif sequence in {"negative", "ns", "2"}:
                z = complex(float(src.r2_pu), float(src.x2_pu))
            elif sequence in {"zero", "zs", "0"}:
                z = complex(float(src.r0_pu), float(src.x0_pu))
            else:
                raise ValueError(f"Unknown sequence: {sequence}")
            if abs(z) >= 1e-12:
                y[idx[bus], idx[bus]] += 1 / z

    return pd.DataFrame(y, index=ids, columns=ids), ids, idx


# -----------------------------------------------------------------------------
# Generic Newton-Raphson power flow
# -----------------------------------------------------------------------------

def _calc_power(v: np.ndarray, theta: np.ndarray, ybus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = ybus.real
    b = ybus.imag
    n = len(v)
    p = np.zeros(n)
    q = np.zeros(n)
    for i in range(n):
        for j in range(n):
            angle = theta[i] - theta[j]
            p[i] += v[i] * v[j] * (g[i, j] * np.cos(angle) + b[i, j] * np.sin(angle))
            q[i] += v[i] * v[j] * (g[i, j] * np.sin(angle) - b[i, j] * np.cos(angle))
    return p, q


def _state_to_arrays(project: PowerSystemProject, ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int], Dict[str, str]]:
    bus_by_id = {str(b.bus_id).strip(): b for b in project.buses}
    n = len(ids)
    v = np.ones(n)
    theta = np.zeros(n)
    p_spec = np.zeros(n)
    q_spec = np.zeros(n)
    slack: List[int] = []
    pv: List[int] = []
    pq: List[int] = []
    types: Dict[str, str] = {}
    for i, bus_id in enumerate(ids):
        b = bus_by_id[bus_id]
        t = str(b.bus_type).strip().lower()
        types[bus_id] = {"slack": "Slack", "pq": "PQ", "pv": "PV"}.get(t, str(b.bus_type))
        v[i] = float(b.voltage_pu)
        theta[i] = np.radians(float(b.angle_deg))
        p_spec[i] = float(b.p_gen_pu) - float(b.p_load_pu)
        # The original backend stores reactive power directly as bus.q.
        # Therefore q_load_pu is treated as the specified Q injection/sign used by the user.
        q_spec[i] = float(b.q_load_pu)
        if t == "slack":
            slack.append(i)
        elif t == "pv":
            pv.append(i)
        else:
            pq.append(i)
    return v, theta, p_spec, q_spec, slack, pv, pq, types


def _pack_state(theta: np.ndarray, v: np.ndarray, non_slack: List[int], pq: List[int]) -> np.ndarray:
    return np.r_[theta[non_slack], v[pq]]


def _unpack_state(x: np.ndarray, theta: np.ndarray, v: np.ndarray, non_slack: List[int], pq: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    theta_new = theta.copy()
    v_new = v.copy()
    n_ang = len(non_slack)
    theta_new[non_slack] = x[:n_ang]
    v_new[pq] = x[n_ang:]
    return theta_new, v_new


def _mismatch_from_state(x: np.ndarray, base_theta: np.ndarray, base_v: np.ndarray, non_slack: List[int], pq: List[int], ybus: np.ndarray, p_spec: np.ndarray, q_spec: np.ndarray) -> np.ndarray:
    theta, v = _unpack_state(x, base_theta, base_v, non_slack, pq)
    p_calc, q_calc = _calc_power(v, theta, ybus)
    return np.r_[p_spec[non_slack] - p_calc[non_slack], q_spec[pq] - q_calc[pq]]


def _numerical_jacobian(func, x: np.ndarray, f0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = len(f0)
    n = len(x)
    j = np.zeros((m, n))
    for k in range(n):
        step = eps * max(1.0, abs(x[k]))
        x1 = x.copy()
        x1[k] += step
        f1 = func(x1)
        j[:, k] = (f1 - f0) / step
    return j


def solve_power_flow_generic(project: PowerSystemProject, max_iterations: int = 20, tolerance: float = 1e-6) -> Dict[str, Any]:
    ybus_df, ids, idx = _build_ybus(project, sequence=None)
    ybus = ybus_df.to_numpy(dtype=complex)
    v, theta, p_spec, q_spec, slack, pv, pq, types = _state_to_arrays(project, ids)
    if len(slack) != 1:
        raise ValueError("Power flow requires exactly one Slack bus.")
    non_slack = [i for i in range(len(ids)) if i not in slack]

    x = _pack_state(theta, v, non_slack, pq)
    logs: List[str] = []
    converged = False
    last_mis = np.array([])

    for iteration in range(1, int(max_iterations) + 1):
        f = _mismatch_from_state(x, theta, v, non_slack, pq, ybus, p_spec, q_spec)
        last_mis = f
        max_mis = float(np.max(np.abs(f))) if f.size else 0.0
        logs.append(f"Iteration {iteration}: max mismatch = {max_mis:.8g}")
        if max_mis < float(tolerance):
            converged = True
            break
        j = _numerical_jacobian(lambda xx: _mismatch_from_state(xx, theta, v, non_slack, pq, ybus, p_spec, q_spec), x, f)
        try:
            dx = np.linalg.solve(j, f)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(j, f, rcond=None)[0]
            logs.append("Jacobian was singular/ill-conditioned; least-squares correction was used.")
        # f is mismatch; Newton update is x_new = x - J^{-1} f.
        x = x - dx
        _, v_candidate = _unpack_state(x, theta, v, non_slack, pq)
        if np.any(v_candidate[pq] <= 0):
            # Simple damping for bad voltage steps.
            x = x + 0.5 * dx
            logs.append("Damping applied because a PQ voltage magnitude became non-positive.")

    theta_final, v_final = _unpack_state(x, theta, v, non_slack, pq)
    p_calc, q_calc = _calc_power(v_final, theta_final, ybus)

    return {
        "converged": converged,
        "message": "Power flow converged." if converged else "Power flow did not converge.",
        "ids": ids,
        "idx": idx,
        "types": types,
        "v": v_final,
        "theta": theta_final,
        "p_spec": p_spec,
        "q_spec": q_spec,
        "p_calc": p_calc,
        "q_calc": q_calc,
        "ybus": ybus_df,
        "iterations": iteration if 'iteration' in locals() else 0,
        "max_mismatch": float(np.max(np.abs(last_mis))) if last_mis.size else 0.0,
        "log": "\n".join(logs),
        "slack_indices": slack,
        "pv_indices": pv,
        "pq_indices": pq,
    }


# -----------------------------------------------------------------------------
# Result extraction
# -----------------------------------------------------------------------------

def _complex_parts(value: complex) -> Dict[str, float]:
    z = complex(value)
    return {
        "real": float(z.real),
        "imag": float(z.imag),
        "magnitude": float(abs(z)),
        "angle_deg": float(np.degrees(cmath.phase(z))),
    }


def _make_bus_table(project: PowerSystemProject, pf: Dict[str, Any]) -> pd.DataFrame:
    bus_by_id = {str(b.bus_id).strip(): b for b in project.buses}
    rows: List[Dict[str, Any]] = []
    for i, bus_id in enumerate(pf["ids"]):
        b = bus_by_id[bus_id]
        rows.append({
            "Bus": int(bus_id),
            "Type": pf["types"].get(bus_id, str(b.bus_type)),
            "Voltage_pu": float(pf["v"][i]),
            "Angle_rad": float(pf["theta"][i]),
            "Angle_deg": float(np.degrees(pf["theta"][i])),
            "P_spec_pu": float(pf["p_spec"][i]),
            "Q_spec_pu": float(pf["q_spec"][i]),
            "P_calc_pu": float(pf["p_calc"][i]),
            "Q_calc_pu": float(pf["q_calc"][i]),
            "P_load_pu": float(b.p_load_pu),
            "Q_load_pu": float(b.q_load_pu),
            "P_gen_pu": float(b.p_gen_pu),
        })
    return pd.DataFrame(rows)


def _make_line_tables(project: PowerSystemProject, pf: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, complex]:
    ids = pf["ids"]
    idx = pf["idx"]
    v_complex = pf["v"] * np.exp(1j * pf["theta"])
    base_mva = float(project.settings.base_power_mva)
    # Three-phase physical base: S_base/(sqrt(3)*V_LL,base) at each bus.
    # The legacy default_base_current_a project setting is NOT a physics-based current base.
    bus_base_kv = {str(b.bus_id).strip(): float(b.base_kv) for b in project.buses}

    current_rows: List[Dict[str, Any]] = []
    flow_rows: List[Dict[str, Any]] = []
    loss_pu_rows: List[Dict[str, Any]] = []
    loss_mva_rows: List[Dict[str, Any]] = []
    total_loss_pu = 0 + 0j

    for br in _branch_lookup(project):
        i = idx[br["from_bus"]]
        j = idx[br["to_bus"]]
        z = complex(br["r_pu"], br["x_pu"])
        y = 1 / z
        y_sh = 0.5j * br["b_pu"]
        vi = v_complex[i]
        vj = v_complex[j]
        iij = (vi - vj) * y + vi * y_sh
        iji = (vj - vi) * y + vj * y_sh
        sij = vi * np.conj(iij)
        sji = vj * np.conj(iji)
        loss = sij + sji
        total_loss_pu += loss

        for from_bus, to_bus, current, flow in [(br["from_bus"], br["to_bus"], iij, sij), (br["to_bus"], br["from_bus"], iji, sji)]:
            parts_i = _complex_parts(current)
            parts_s = _complex_parts(flow)
            vbase_kv = bus_base_kv.get(str(from_bus), 0.0)
            current_base_a = base_mva * 1000.0 / (np.sqrt(3.0) * vbase_kv) if vbase_kv > 0 else float("nan")
            rating_verified = bool(br.get("ampacity_verified", False))
            current_rows.append({
                "Branch": br["name"],
                "From_Bus": int(from_bus),
                "To_Bus": int(to_bus),
                "I_pu_real": parts_i["real"],
                "I_pu_imag": parts_i["imag"],
                "I_pu_mag": parts_i["magnitude"],
                "I_pu_angle_deg": parts_i["angle_deg"],
                "Line_current_A": parts_i["magnitude"] * current_base_a,
                "Current_base_A": current_base_a,
                "Ampacity_pu": br["ampacity_pu"],
                "Ampacity_verified": rating_verified,
                "Overcurrent": rating_verified and parts_i["magnitude"] > br["ampacity_pu"],
            })
            flow_rows.append({
                "Branch": br["name"],
                "From_Bus": int(from_bus),
                "To_Bus": int(to_bus),
                "S_pu_real": parts_s["real"],
                "S_pu_imag": parts_s["imag"],
                "S_pu_mag": parts_s["magnitude"],
                "S_pu_angle_deg": parts_s["angle_deg"],
                "P_MW": parts_s["real"] * base_mva,
                "Q_MVAR": parts_s["imag"] * base_mva,
                "S_MVA_mag": parts_s["magnitude"] * base_mva,
            })
        loss_parts = _complex_parts(loss)
        loss_pu_rows.append({
            "Branch": br["name"],
            "From_Bus": int(br["from_bus"]),
            "To_Bus": int(br["to_bus"]),
            "Loss_pu_real": loss_parts["real"],
            "Loss_pu_imag": loss_parts["imag"],
            "Loss_pu_mag": loss_parts["magnitude"],
        })
        loss_mva_rows.append({
            "Branch": br["name"],
            "From_Bus": int(br["from_bus"]),
            "To_Bus": int(br["to_bus"]),
            "P_loss_MW": loss_parts["real"] * base_mva,
            "Q_loss_MVAR": loss_parts["imag"] * base_mva,
            "S_loss_MVA_mag": loss_parts["magnitude"] * base_mva,
        })

    return (
        pd.DataFrame(current_rows),
        pd.DataFrame(flow_rows),
        pd.DataFrame(loss_pu_rows),
        pd.DataFrame(loss_mva_rows),
        total_loss_pu,
    )


def _summary_dataframe(summary: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([dict(summary)])


# -----------------------------------------------------------------------------
# Public power-flow API
# -----------------------------------------------------------------------------

def run_power_flow_project(
    project: PowerSystemProject,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    suppress_output: bool = True,
    include_matrices: bool = True,
) -> PowerFlowResult:
    """Run a fully data-driven power flow from UI-entered project tables."""
    ok, errors, warnings = validate_project_generic(project)
    if not ok:
        raise ValueError("Project validation failed:\n" + "\n".join(errors))

    log_stream = io.StringIO()
    try:
        ctx = contextlib.redirect_stdout(log_stream) if suppress_output else contextlib.nullcontext()
        with ctx:
            pf = solve_power_flow_generic(project, max_iterations=max_iterations, tolerance=tolerance)
        bus_table = _make_bus_table(project, pf)
        line_current_table, line_flow_table, losses_pu, losses_mva, total_loss_pu = _make_line_tables(project, pf)
        base_mva = float(project.settings.base_power_mva)
        total_loss_mva = total_loss_pu * base_mva
        summary = {
            "converged": bool(pf["converged"]),
            "message": pf["message"],
            "iterations": int(pf["iterations"]),
            "max_mismatch": float(pf["max_mismatch"]),
            "tolerance": float(tolerance),
            "number_of_buses": len(pf["ids"]),
            "number_of_branches": len(project.branches),
            "number_of_current_records": len(line_current_table),
            "number_of_loss_records": len(losses_pu),
            "total_loss_pu_real": float(total_loss_pu.real),
            "total_loss_pu_imag": float(total_loss_pu.imag),
            "total_loss_MVA_real": float(total_loss_mva.real),
            "total_loss_MVA_imag": float(total_loss_mva.imag),
            "base_power_MVA": base_mva,
        }
        matrices = {"ybus": pf["ybus"]} if include_matrices else {}
        return PowerFlowResult(
            converged=bool(pf["converged"]),
            message=pf["message"],
            circuit=None,
            solver=None,
            bus_table=bus_table,
            line_current_table=line_current_table,
            line_flow_table=line_flow_table,
            losses_table_pu=losses_pu,
            losses_table_mva=losses_mva,
            summary=summary,
            matrices=matrices,
            raw={"power_flow": pf, "warnings": warnings},
            log=(log_stream.getvalue() + "\n" + pf["log"]).strip(),
        )
    except Exception as exc:
        if not suppress_output:
            raise
        return PowerFlowResult(
            converged=False,
            message=f"Power flow failed: {exc}",
            circuit=None,
            solver=None,
            bus_table=pd.DataFrame(),
            line_current_table=pd.DataFrame(),
            line_flow_table=pd.DataFrame(),
            losses_table_pu=pd.DataFrame(),
            losses_table_mva=pd.DataFrame(),
            summary={"converged": False, "message": f"Power flow failed: {exc}"},
            matrices={},
            raw={"warnings": warnings if 'warnings' in locals() else []},
            log=log_stream.getvalue(),
        )


# -----------------------------------------------------------------------------
# Generic fault study
# -----------------------------------------------------------------------------

def _a_matrix() -> np.ndarray:
    a = np.exp(1j * np.deg2rad(120))
    a2 = np.exp(1j * np.deg2rad(240))
    return np.array([[1, 1, 1], [1, a2, a], [1, a, a2]], dtype=complex)


def _polar_rows(bus_ids: List[str], vectors: List[np.ndarray], labels: List[str], label_col: str, value_prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bus_id, vec in zip(bus_ids, vectors):
        arr = np.asarray(vec).reshape(-1)
        for label, z in zip(labels, arr):
            p = _complex_parts(z)
            rows.append({
                "Bus": int(bus_id),
                label_col: label,
                f"{value_prefix}_real": p["real"],
                f"{value_prefix}_imag": p["imag"],
                f"{value_prefix}_mag_pu": p["magnitude"],
                f"{value_prefix}_angle_deg": p["angle_deg"],
            })
    return pd.DataFrame(rows)


def _vector_to_voltage_table(bus_ids: List[str], vec: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bus_id, z in zip(bus_ids, np.asarray(vec).reshape(-1)):
        p = _complex_parts(z)
        rows.append({
            "Bus": int(bus_id),
            "Voltage_real": p["real"],
            "Voltage_imag": p["imag"],
            "Voltage_mag_pu": p["magnitude"],
            "Voltage_angle_deg": p["angle_deg"],
        })
    return pd.DataFrame(rows)


def _sequence_matrices(project: PowerSystemProject) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], Dict[str, pd.DataFrame]]:
    y1, ids, idx = _build_ybus(project, "positive")
    y2, _, _ = _build_ybus(project, "negative")
    y0, _, _ = _build_ybus(project, "zero")
    mats = {
        "ybus_positive_sequence": y1,
        "ybus_negative_sequence": y2,
        "ybus_zero_sequence": y0,
    }
    return y1.to_numpy(dtype=complex), y2.to_numpy(dtype=complex), y0.to_numpy(dtype=complex), ids, idx, mats


def run_fault_study_project(
    project: PowerSystemProject,
    fault_type: str,
    faulted_bus: int = 4,
    prefault_voltage: float = 1.0,
    fault_impedance: Union[float, complex] = 0.0,
    solve_prefault_power_flow: bool = False,
    suppress_output: bool = True,
) -> FaultResult:
    """Run a generic sequence-network fault study from UI-entered project data."""
    ok, errors, warnings = validate_project_generic(project)
    if not ok:
        raise ValueError("Project validation failed:\n" + "\n".join(errors))

    fault_key = str(fault_type).strip().lower().replace("-", "_").replace(" ", "_")
    y1, y2, y0, ids, idx, mats = _sequence_matrices(project)
    fb = str(int(faulted_bus))
    if fb not in idx:
        raise ValueError(f"Faulted bus {faulted_bus} does not exist in the project.")
    k = idx[fb]
    vf = complex(float(prefault_voltage), 0.0)
    zf = complex(fault_impedance)
    A = _a_matrix()

    try:
        z1 = np.linalg.inv(y1)
        z2 = np.linalg.inv(y2)
        z0 = np.linalg.inv(y0)
    except np.linalg.LinAlgError as exc:
        return FaultResult(
            fault_type=fault_type,
            faulted_bus=int(faulted_bus),
            message=f"Fault study failed: a sequence Y-bus matrix is singular. Add/adjust sequence sources. Details: {exc}",
            current_table=pd.DataFrame(),
            voltage_table=pd.DataFrame(),
            raw={"matrices": mats, "warnings": warnings},
        )

    n = len(ids)
    zero = np.zeros((n, 1), dtype=complex)

    if fault_key in {"balanced", "three_phase", "3phase", "3_phase", "threephase"}:
        iff = vf / (z1[k, k] + zf)
        i_vec = zero.copy()
        i_vec[k, 0] = iff
        v_post = np.ones((n, 1), dtype=complex) - z1 @ i_vec
        p = _complex_parts(iff)
        current_table = pd.DataFrame([{
            "Faulted_Bus": int(faulted_bus),
            "Fault_Type": "Balanced three-phase",
            "I_fault_real_pu": p["real"],
            "I_fault_imag_pu": p["imag"],
            "I_fault_mag_pu": p["magnitude"],
            "I_fault_angle_deg": p["angle_deg"],
        }])
        return FaultResult(
            fault_type="balanced",
            faulted_bus=int(faulted_bus),
            message="Balanced three-phase fault completed.",
            current_table=current_table,
            voltage_table=_vector_to_voltage_table(ids, v_post),
            raw={"matrices": mats, "warnings": warnings},
        )

    I0 = zero.copy()
    I1 = zero.copy()
    I2 = zero.copy()

    if fault_key in {"lg", "slg", "line_ground", "line_to_ground"}:
        i = vf / (z1[k, k] + z2[k, k] + z0[k, k] + 3 * zf)
        I0[k, 0] = I1[k, 0] = I2[k, 0] = i
        label = "Line-to-ground fault completed."
        ftype = "LG"
    elif fault_key in {"ll", "line_line", "line_to_line"}:
        i = vf / (z1[k, k] + z2[k, k] + zf)
        I1[k, 0] = i
        I2[k, 0] = -i
        label = "Line-to-line fault completed."
        ftype = "LL"
    elif fault_key in {"llg", "line_line_ground", "double_line_ground"}:
        I1f = vf / (z1[k, k] + (z2[k, k] * (z0[k, k] + 3 * zf) / (z2[k, k] + z0[k, k] + 3 * zf)))
        I2f = -I1f * (z0[k, k] + 3 * zf) / (z0[k, k] + 3 * zf + z2[k, k])
        I0f = -I1f * z2[k, k] / (z0[k, k] + 3 * zf + z2[k, k])
        I0[k, 0] = I0f
        I1[k, 0] = I1f
        I2[k, 0] = I2f
        label = "Line-line-ground fault completed."
        ftype = "LLG"
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    V0 = 0 - z0 @ I0
    V1 = vf - z1 @ I1
    V2 = 0 - z2 @ I2

    current_012: List[np.ndarray] = []
    current_abc: List[np.ndarray] = []
    voltage_012: List[np.ndarray] = []
    voltage_abc: List[np.ndarray] = []
    for i in range(n):
        ci = np.array([[I0[i, 0]], [I1[i, 0]], [I2[i, 0]]], dtype=complex)
        vi = np.array([[V0[i, 0]], [V1[i, 0]], [V2[i, 0]]], dtype=complex)
        current_012.append(ci)
        current_abc.append(A @ ci)
        voltage_012.append(vi)
        voltage_abc.append(A @ vi)

    return FaultResult(
        fault_type=ftype,
        faulted_bus=int(faulted_bus),
        message=label,
        current_table=_polar_rows(ids, current_abc, ["Ia", "Ib", "Ic"], "Phase", "Current"),
        voltage_table=_polar_rows(ids, voltage_abc, ["Va", "Vb", "Vc"], "Phase", "Voltage"),
        sequence_current_table=_polar_rows(ids, current_012, ["I0", "I1", "I2"], "Sequence", "Current"),
        sequence_voltage_table=_polar_rows(ids, voltage_012, ["V0", "V1", "V2"], "Sequence", "Voltage"),
        raw={"matrices": mats, "warnings": warnings},
    )


# -----------------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------------

def smoke_test() -> Dict[str, Any]:
    from project_model import default_7_bus_project
    project = default_7_bus_project()
    pf = run_power_flow_project(project)
    fault = run_fault_study_project(project, "balanced", faulted_bus=4)
    return {
        "power_flow_message": pf.message,
        "bus_table_shape": pf.bus_table.shape,
        "line_current_table_shape": pf.line_current_table.shape,
        "line_flow_table_shape": pf.line_flow_table.shape,
        "losses_table_shape": pf.losses_table_pu.shape,
        "fault_message": fault.message,
        "fault_voltage_shape": fault.voltage_table.shape,
    }


if __name__ == "__main__":
    print(smoke_test())
