"""
adapter_project_fbs.py

Solver-dispatch adapter for the project-driven UI.

Adds a second power-flow solver:
- Newton-Raphson: generic numerical Newton solver from adapter_project_generic.py
- Forward/Backward Sweep: radial Slack/PQ solver from fbs_solver.py
- Auto: use FBS when the project is radial and Slack/PQ only; otherwise use Newton-Raphson

Fault studies are still delegated to adapter_project_generic.py.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List

import pandas as pd

from adapter_project_generic import (  # reuse proven project adapter utilities
    FaultResult,
    PowerFlowResult,
    _make_bus_table,
    _make_line_tables,
    run_fault_study_project,
    run_power_flow_project as run_power_flow_project_newton,
    validate_project_generic,
)
from fbs_solver import is_radial_fbs_eligible, solve_power_flow_fbs, validate_fbs_project

try:
    from project_model import PowerSystemProject
except Exception:  # pragma: no cover
    PowerSystemProject = Any  # type: ignore


def _normalise_method(method: str | None) -> str:
    raw = (method or "Newton-Raphson").strip().lower().replace("_", "-")
    if raw in {"nr", "newton", "newton-raphson", "newton raphson"}:
        return "Newton-Raphson"
    if raw in {"fbs", "bfs", "bfsw", "forward/backward sweep", "forward-backward sweep", "forward backward sweep"}:
        return "Forward/Backward Sweep"
    if raw in {"auto", "automatic"}:
        return "Auto"
    raise ValueError(f"Unknown solver method: {method!r}")


def choose_solver_method(project: PowerSystemProject, requested: str | None = "Auto") -> tuple[str, list[str]]:
    """Return selected solver method and explanatory warnings."""
    method = _normalise_method(requested)
    warnings: list[str] = []
    if method != "Auto":
        return method, warnings

    eligible, fbs_errors, fbs_warnings = is_radial_fbs_eligible(project)
    if eligible:
        warnings.extend(["Auto selected Forward/Backward Sweep because the network is radial and has Slack/PQ buses."])
        warnings.extend(fbs_warnings)
        return "Forward/Backward Sweep", warnings

    warnings.append("Auto selected Newton-Raphson because FBS eligibility was not satisfied.")
    warnings.extend([f"FBS not used: {msg}" for msg in fbs_errors[:4]])
    return "Newton-Raphson", warnings


def run_power_flow_project(
    project: PowerSystemProject,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    suppress_output: bool = True,
    include_matrices: bool = True,
    solver_method: str | None = "Newton-Raphson",
) -> PowerFlowResult:
    """Run power flow using Newton-Raphson, FBS, or Auto selection."""
    ok, errors, warnings = validate_project_generic(project)
    if not ok:
        raise ValueError("Project validation failed:\n" + "\n".join(errors))

    method, method_warnings = choose_solver_method(project, solver_method)
    warnings = list(warnings) + method_warnings

    if method == "Newton-Raphson":
        result = run_power_flow_project_newton(
            project,
            max_iterations=max_iterations,
            tolerance=tolerance,
            suppress_output=suppress_output,
            include_matrices=include_matrices,
        )
        result.summary["solver_method"] = "Newton-Raphson"
        result.message = f"Newton-Raphson: {result.message}"
        result.raw.setdefault("warnings", [])
        result.raw["warnings"] = list(result.raw.get("warnings", [])) + warnings
        if warnings:
            result.log = (result.log + "\n" + "\n".join(warnings)).strip()
        return result

    if method == "Forward/Backward Sweep":
        fbs_ok, fbs_errors, fbs_warnings = validate_fbs_project(project)
        if not fbs_ok:
            raise ValueError("Forward/Backward Sweep cannot run on this project:\n" + "\n".join(fbs_errors))
        warnings.extend(fbs_warnings)
        log_stream = io.StringIO()
        try:
            ctx = contextlib.redirect_stdout(log_stream) if suppress_output else contextlib.nullcontext()
            with ctx:
                pf = solve_power_flow_fbs(project, max_iterations=max_iterations, tolerance=tolerance)
            bus_table = _make_bus_table(project, pf)
            line_current_table, line_flow_table, losses_pu, losses_mva, total_loss_pu = _make_line_tables(project, pf)
            base_mva = float(project.settings.base_power_mva)
            total_loss_mva = total_loss_pu * base_mva
            summary: Dict[str, Any] = {
                "converged": bool(pf["converged"]),
                "message": pf["message"],
                "solver_method": "Forward/Backward Sweep",
                "iterations": int(pf["iterations"]),
                "max_voltage_change": float(pf["max_mismatch"]),
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
                raw={"power_flow": pf, "warnings": warnings, "solver_method": "Forward/Backward Sweep"},
                log=(log_stream.getvalue() + "\n" + pf["log"] + "\n" + "\n".join(warnings)).strip(),
            )
        except Exception as exc:
            if not suppress_output:
                raise
            return PowerFlowResult(
                converged=False,
                message=f"Forward/Backward Sweep failed: {exc}",
                circuit=None,
                solver=None,
                bus_table=pd.DataFrame(),
                line_current_table=pd.DataFrame(),
                line_flow_table=pd.DataFrame(),
                losses_table_pu=pd.DataFrame(),
                losses_table_mva=pd.DataFrame(),
                summary={"converged": False, "message": f"Forward/Backward Sweep failed: {exc}", "solver_method": "Forward/Backward Sweep"},
                matrices={},
                raw={"warnings": warnings, "solver_method": "Forward/Backward Sweep"},
                log=log_stream.getvalue(),
            )

    raise ValueError(f"Unsupported solver method: {method}")


def available_solver_methods() -> list[str]:
    return ["Auto", "Newton-Raphson", "Forward/Backward Sweep"]


def smoke_test() -> Dict[str, Any]:
    from project_model import BusRecord, BranchRecord, PowerSystemProject, ProjectSettings, default_7_bus_project

    # Auto should choose Newton-Raphson for the meshed/PV default 7-bus project.
    default_project = default_7_bus_project()
    nr = run_power_flow_project(default_project, max_iterations=10, tolerance=1e-4, solver_method="Auto")

    # A simple radial Slack/PQ case should use FBS.
    radial = PowerSystemProject(
        settings=ProjectSettings(name="3-bus radial", base_power_mva=100.0),
        buses=[
            BusRecord("1", "Slack", 1.0, 0.0, 0.0, 0.0, 0.0),
            BusRecord("2", "PQ", 1.0, 0.0, 0.5, -0.2, 0.0),
            BusRecord("3", "PQ", 1.0, 0.0, 0.4, -0.15, 0.0),
        ],
        branches=[
            BranchRecord("L12", "1", "2", 0.01, 0.05, 0.0, 10.0),
            BranchRecord("L23", "2", "3", 0.015, 0.045, 0.0, 10.0),
        ],
        sequence_sources=[],
    )
    fbs = run_power_flow_project(radial, max_iterations=100, tolerance=1e-8, solver_method="Forward/Backward Sweep")

    return {
        "default_auto_solver": nr.summary.get("solver_method"),
        "default_message": nr.message,
        "default_bus_table_shape": tuple(nr.bus_table.shape),
        "radial_solver": fbs.summary.get("solver_method"),
        "radial_message": fbs.message,
        "radial_bus_table_shape": tuple(fbs.bus_table.shape),
        "radial_min_voltage": float(fbs.bus_table["Voltage_pu"].min()),
    }


if __name__ == "__main__":
    print(smoke_test())
