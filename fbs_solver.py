"""
fbs_solver.py

Forward/Backward Sweep (FBS/BFSW) solver for radial distribution networks
entered through the project_model.py data model.

Scope of this first implementation
----------------------------------
- Single-phase balanced per-unit equivalent.
- Radial networks only.
- One Slack bus and PQ buses.
- Constant-power bus injections/loads.
- Series branch impedances are used in the FBS iteration.
- Branch shunt susceptance is kept in the returned Y-bus and later result
  extraction, but is intentionally not used in the FBS voltage iteration.
  This matches the common distribution-feeder FBS simplification where line
  charging is negligible compared with load current.

Why separate from the Newton-Raphson backend?
---------------------------------------------
The existing simulator already has a Newton-Raphson path.  This module adds a
radial-distribution solver without modifying the original backend classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from project_model import PowerSystemProject
except Exception:  # pragma: no cover
    PowerSystemProject = Any  # type: ignore


@dataclass
class RadialTopology:
    root: str
    ids: List[str]
    idx: Dict[str, int]
    parent: Dict[str, str]
    children: Dict[str, List[str]]
    parent_branch: Dict[str, Dict[str, Any]]
    order: List[str]


def _bus_sort_key(bus_id: str) -> int:
    return int(str(bus_id).strip())


def _ordered_bus_ids(project: PowerSystemProject) -> List[str]:
    return [str(b.bus_id).strip() for b in sorted(project.buses, key=lambda b: _bus_sort_key(b.bus_id))]


def _branch_rows(project: PowerSystemProject) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, br in enumerate(project.branches, start=1):
        rows.append({
            "name": str(br.name).strip() or f"Branch{k}",
            "from_bus": str(br.from_bus).strip(),
            "to_bus": str(br.to_bus).strip(),
            "r_pu": float(br.r_pu),
            "x_pu": float(br.x_pu),
            "b_pu": float(br.b_pu),
            "ampacity_pu": float(br.ampacity_pu),
            "element_type": str(getattr(br, "element_type", "Line")),
        })
    return rows


def _bus_type_map(project: PowerSystemProject) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for bus in project.buses:
        t = str(bus.bus_type).strip().lower()
        mapping[str(bus.bus_id).strip()] = {"slack": "Slack", "pq": "PQ", "pv": "PV"}.get(t, str(bus.bus_type))
    return mapping


def _build_ybus(project: PowerSystemProject) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    ids = _ordered_bus_ids(project)
    idx = {bus_id: i for i, bus_id in enumerate(ids)}
    y = np.zeros((len(ids), len(ids)), dtype=complex)
    for br in _branch_rows(project):
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
    return pd.DataFrame(y, index=ids, columns=ids), ids, idx


def _calc_power(v: np.ndarray, theta: np.ndarray, ybus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Network power injection from final voltages and Y-bus."""
    vc = v * np.exp(1j * theta)
    current = ybus @ vc
    s = vc * np.conj(current)
    return s.real, s.imag


def validate_fbs_project(project: PowerSystemProject) -> Tuple[bool, List[str], List[str]]:
    """Checks specific to the FBS solver."""
    errors: List[str] = []
    warnings: List[str] = []

    ids = _ordered_bus_ids(project)
    bus_set = set(ids)
    if not ids:
        return False, ["FBS requires at least one bus."], warnings

    slack = [str(b.bus_id).strip() for b in project.buses if str(b.bus_type).strip().lower() == "slack"]
    if len(slack) != 1:
        errors.append("FBS requires exactly one Slack bus.")

    pv = [str(b.bus_id).strip() for b in project.buses if str(b.bus_type).strip().lower() == "pv"]
    if pv:
        errors.append(
            "FBS v1 supports Slack + PQ buses only. Convert PV buses to PQ or use Newton-Raphson. "
            f"PV buses found: {', '.join(pv)}."
        )

    branches = _branch_rows(project)
    if len(branches) != max(0, len(ids) - 1):
        errors.append(
            f"FBS requires a radial network with branches = buses - 1. "
            f"Found {len(branches)} branches and {len(ids)} buses."
        )

    seen_edges = set()
    adjacency: Dict[str, List[str]] = {x: [] for x in ids}
    for br in branches:
        a = br["from_bus"]
        b = br["to_bus"]
        if a not in bus_set or b not in bus_set:
            # Generic project validation will report the exact missing bus.
            continue
        edge = tuple(sorted((a, b), key=lambda x: int(x)))
        if edge in seen_edges:
            errors.append(f"FBS does not allow parallel duplicate branches between buses {edge[0]} and {edge[1]}.")
        seen_edges.add(edge)
        adjacency[a].append(b)
        adjacency[b].append(a)

    if slack:
        root = slack[0]
        visited = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nb in adjacency.get(node, []):
                if nb not in visited:
                    queue.append(nb)
        if len(visited) != len(ids):
            missing = sorted(bus_set - visited, key=lambda x: int(x))
            errors.append(f"FBS requires a connected radial network. Isolated/unreached buses: {', '.join(missing)}.")

    if any(abs(float(getattr(br, "b_pu", 0.0))) > 1e-9 for br in getattr(project, "branches", [])):
        warnings.append(
            "FBS v1 ignores branch shunt B during the voltage iteration; B is still included in Y-bus/results extraction."
        )

    return len(errors) == 0, errors, warnings


def _orient_radial_network(project: PowerSystemProject) -> RadialTopology:
    ids = _ordered_bus_ids(project)
    idx = {bus_id: i for i, bus_id in enumerate(ids)}
    slack = [str(b.bus_id).strip() for b in project.buses if str(b.bus_type).strip().lower() == "slack"]
    if len(slack) != 1:
        raise ValueError("FBS requires exactly one Slack bus.")
    root = slack[0]

    adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {x: [] for x in ids}
    for br in _branch_rows(project):
        adjacency[br["from_bus"]].append((br["to_bus"], br))
        adjacency[br["to_bus"]].append((br["from_bus"], br))

    parent: Dict[str, str] = {}
    children: Dict[str, List[str]] = {x: [] for x in ids}
    parent_branch: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nb, br in adjacency[node]:
            if nb in visited:
                continue
            visited.add(nb)
            parent[nb] = node
            children[node].append(nb)
            parent_branch[nb] = br
            queue.append(nb)

    if len(visited) != len(ids):
        raise ValueError("FBS network is not connected from the Slack bus.")

    return RadialTopology(root=root, ids=ids, idx=idx, parent=parent, children=children, parent_branch=parent_branch, order=order)


def solve_power_flow_fbs(project: PowerSystemProject, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Run FBS power flow on a radial Slack/PQ project.

    Returned dictionary intentionally mirrors adapter_project_generic.solve_power_flow_generic()
    so the existing table/plot extraction functions can be reused.
    """
    ok, errors, warnings = validate_fbs_project(project)
    if not ok:
        raise ValueError("FBS validation failed:\n" + "\n".join(errors))

    topo = _orient_radial_network(project)
    ybus_df, ids, idx = _build_ybus(project)
    ybus = ybus_df.to_numpy(dtype=complex)
    types = _bus_type_map(project)
    bus_by_id = {str(b.bus_id).strip(): b for b in project.buses}

    n = len(ids)
    v0 = np.ones(n, dtype=complex)
    p_spec = np.zeros(n)
    q_spec = np.zeros(n)
    for bus_id in ids:
        i = idx[bus_id]
        bus = bus_by_id[bus_id]
        vm = float(bus.voltage_pu)
        va = np.radians(float(bus.angle_deg))
        v0[i] = vm * np.exp(1j * va)
        p_spec[i] = float(bus.p_gen_pu) - float(bus.p_load_pu)
        # Keep the same sign convention used by the generic Newton-Raphson adapter.
        q_spec[i] = float(bus.q_load_pu)

    root_idx = idx[topo.root]
    v_slack = v0[root_idx]
    voltage = v0.copy()
    voltage[root_idx] = v_slack

    # Demand is the negative of specified net injection.
    # Example: P_load=1.1, P_gen=0, Q_input=-0.5 -> S_spec=-1.1-j0.5, S_demand=1.1+j0.5.
    s_spec = p_spec + 1j * q_spec
    s_demand = -s_spec
    s_demand[root_idx] = 0.0 + 0.0j

    logs: List[str] = []
    branch_current: Dict[Tuple[str, str], complex] = {}
    converged = False
    max_delta = np.inf

    for iteration in range(1, int(max_iterations) + 1):
        # Backward sweep: bus injection currents at current voltages.
        load_current = np.zeros(n, dtype=complex)
        for bus_id in ids:
            i = idx[bus_id]
            if i == root_idx:
                continue
            if abs(voltage[i]) < 1e-12:
                raise ValueError(f"Voltage at bus {bus_id} became numerically zero during FBS.")
            load_current[i] = np.conj(s_demand[i] / voltage[i])

        downstream_current: Dict[str, complex] = {bus_id: load_current[idx[bus_id]] for bus_id in ids}
        branch_current.clear()
        for bus_id in reversed(topo.order):
            if bus_id == topo.root:
                continue
            total = downstream_current[bus_id]
            p = topo.parent[bus_id]
            branch_current[(p, bus_id)] = total
            downstream_current[p] += total

        # Forward sweep: update child bus voltages from the root.
        new_voltage = voltage.copy()
        new_voltage[root_idx] = v_slack
        for p in topo.order:
            for c in topo.children[p]:
                br = topo.parent_branch[c]
                z = complex(br["r_pu"], br["x_pu"])
                new_voltage[idx[c]] = new_voltage[idx[p]] - z * branch_current[(p, c)]

        max_delta = float(np.max(np.abs(new_voltage - voltage)))
        logs.append(f"Iteration {iteration}: max |Delta V| = {max_delta:.8g}")
        voltage = new_voltage
        if max_delta < float(tolerance):
            converged = True
            break

    v_mag = np.abs(voltage)
    theta = np.angle(voltage)
    p_calc, q_calc = _calc_power(v_mag, theta, ybus)

    return {
        "converged": converged,
        "message": "FBS power flow converged." if converged else "FBS power flow did not converge.",
        "ids": ids,
        "idx": idx,
        "types": types,
        "v": v_mag,
        "theta": theta,
        "p_spec": p_spec,
        "q_spec": q_spec,
        "p_calc": p_calc,
        "q_calc": q_calc,
        "ybus": ybus_df,
        "iterations": iteration if "iteration" in locals() else 0,
        "max_mismatch": max_delta,
        "log": "\n".join(logs),
        "slack_indices": [root_idx],
        "pv_indices": [],
        "pq_indices": [idx[x] for x in ids if x != topo.root],
        "method": "Forward/Backward Sweep",
        "warnings": warnings,
    }


def is_radial_fbs_eligible(project: PowerSystemProject) -> Tuple[bool, List[str], List[str]]:
    """Convenience function for Auto solver selection."""
    return validate_fbs_project(project)
