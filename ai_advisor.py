"""
ai_advisor.py

Rule-based AI Engineering Advisor for the power-system simulator.

This module does not replace the numerical power-flow solvers. It reads the
validated solver outputs and produces diagnostic engineering insights,
findings, severity labels, and recommended corrective actions.

Intended first AI layer:
- No training data required.
- Deterministic and traceable.
- Safe for non-programming engineers.
- Can later be paired with an LLM to rewrite the report, while keeping these
  rules as the source of technical truth.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


@dataclass
class AdvisorThresholds:
    """Editable engineering thresholds used by the AI Advisor."""

    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    voltage_critical_min_pu: float = 0.90
    high_loading_percent: float = 80.0
    overload_percent: float = 100.0
    top_loss_branches: int = 5

    def validate(self) -> None:
        if not (0.0 < self.voltage_critical_min_pu <= self.voltage_min_pu < self.voltage_max_pu):
            raise ValueError(
                "Voltage thresholds must satisfy: 0 < critical low <= lower limit < upper limit."
            )
        if self.high_loading_percent <= 0:
            raise ValueError("High loading threshold must be positive.")
        if self.overload_percent <= 0:
            raise ValueError("Overload threshold must be positive.")
        if self.high_loading_percent > self.overload_percent:
            raise ValueError("High loading threshold cannot exceed overload threshold.")
        if int(self.top_loss_branches) <= 0:
            raise ValueError("Top loss branches must be a positive integer.")


@dataclass
class AIInsightReport:
    """Structured output of the AI Advisor."""

    report_text: str
    findings_table: pd.DataFrame
    actions_table: pd.DataFrame
    summary: Dict[str, Any]
    thresholds: AdvisorThresholds

    def threshold_dict(self) -> Dict[str, Any]:
        return asdict(self.thresholds)


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _to_numeric_series(df: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if column is None or df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _format_bus(bus: Any) -> str:
    if pd.isna(bus):
        return "N/A"
    try:
        v = float(bus)
        if v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return str(bus)


def _add_finding(rows: list[dict[str, Any]], severity: str, category: str, item: str, value: Any, message: str) -> None:
    rows.append(
        {
            "Severity": severity,
            "Category": category,
            "Item": item,
            "Value": value,
            "Message": message,
        }
    )


def _add_action(rows: list[dict[str, Any]], priority: str, target: str, action: str, rationale: str) -> None:
    rows.append(
        {
            "Priority": priority,
            "Target": target,
            "Recommended Action": action,
            "Rationale": rationale,
        }
    )


def _summarize_voltage(bus_table: pd.DataFrame, thresholds: AdvisorThresholds) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "min_voltage_pu": None,
        "min_voltage_bus": None,
        "max_voltage_pu": None,
        "max_voltage_bus": None,
        "low_voltage_count": 0,
        "critical_low_voltage_count": 0,
        "high_voltage_count": 0,
    }

    if bus_table is None or bus_table.empty:
        _add_finding(findings, "Warning", "Voltage", "Bus table", "N/A", "No bus-voltage table is available.")
        return summary, findings, actions

    bus_col = _first_existing_column(bus_table, ["Bus", "bus_id", "Bus_ID"])
    v_col = _first_existing_column(bus_table, ["Voltage_pu", "Voltage_mag_pu", "V_pu", "Voltage"])
    if v_col is None:
        _add_finding(findings, "Warning", "Voltage", "Voltage column", "N/A", "No voltage-magnitude column was found.")
        return summary, findings, actions

    tmp = bus_table.copy()
    tmp["__v__"] = pd.to_numeric(tmp[v_col], errors="coerce")
    tmp = tmp.dropna(subset=["__v__"])
    if tmp.empty:
        _add_finding(findings, "Warning", "Voltage", "Voltage values", "N/A", "Voltage values could not be parsed as numbers.")
        return summary, findings, actions

    min_row = tmp.loc[tmp["__v__"].idxmin()]
    max_row = tmp.loc[tmp["__v__"].idxmax()]
    min_bus = _format_bus(min_row[bus_col]) if bus_col else "N/A"
    max_bus = _format_bus(max_row[bus_col]) if bus_col else "N/A"
    vmin = float(min_row["__v__"])
    vmax = float(max_row["__v__"])
    low = tmp[tmp["__v__"] < thresholds.voltage_min_pu]
    critical = tmp[tmp["__v__"] < thresholds.voltage_critical_min_pu]
    high = tmp[tmp["__v__"] > thresholds.voltage_max_pu]

    summary.update(
        {
            "min_voltage_pu": vmin,
            "min_voltage_bus": min_bus,
            "max_voltage_pu": vmax,
            "max_voltage_bus": max_bus,
            "low_voltage_count": int(len(low)),
            "critical_low_voltage_count": int(len(critical)),
            "high_voltage_count": int(len(high)),
        }
    )

    if len(critical) > 0:
        _add_finding(
            findings,
            "Critical",
            "Voltage",
            f"Bus {min_bus}",
            f"{vmin:.5f} pu",
            f"Minimum voltage is below the critical threshold of {thresholds.voltage_critical_min_pu:.3f} pu.",
        )
        _add_action(
            actions,
            "High",
            f"Low-voltage area near Bus {min_bus}",
            "Test reactive-power support, capacitor placement, local DG support, or feeder reinforcement near the weakest downstream buses.",
            "Critical undervoltage usually indicates insufficient voltage support, high downstream loading, or high feeder impedance.",
        )
    elif len(low) > 0:
        _add_finding(
            findings,
            "Warning",
            "Voltage",
            f"Bus {min_bus}",
            f"{vmin:.5f} pu",
            f"Minimum voltage is below the lower threshold of {thresholds.voltage_min_pu:.3f} pu.",
        )
        _add_action(
            actions,
            "High",
            f"Bus {min_bus}",
            "Evaluate capacitor placement, local DG voltage support, tap setting adjustment, or load transfer.",
            "The lowest-voltage bus identifies the first area to investigate for voltage improvement.",
        )
    else:
        _add_finding(
            findings,
            "OK",
            "Voltage",
            "All buses",
            f"min={vmin:.5f} pu, max={vmax:.5f} pu",
            "All bus voltages are within the configured voltage band.",
        )

    if len(high) > 0:
        _add_finding(
            findings,
            "Warning",
            "Voltage",
            f"Bus {max_bus}",
            f"{vmax:.5f} pu",
            f"Maximum voltage exceeds the upper threshold of {thresholds.voltage_max_pu:.3f} pu.",
        )
        _add_action(
            actions,
            "Medium",
            f"High-voltage area near Bus {max_bus}",
            "Review voltage-control settings, DG reactive-power behavior, tap positions, and lightly loaded feeder sections.",
            "High voltage can occur under light loading, excessive reactive support, or aggressive voltage regulation.",
        )

    return summary, findings, actions


def _summarize_loading(line_current_table: pd.DataFrame, thresholds: AdvisorThresholds) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "max_loading_percent": None,
        "max_loading_branch": None,
        "high_loading_count": 0,
        "overload_count": 0,
    }

    if line_current_table is None or line_current_table.empty:
        _add_finding(findings, "Info", "Line loading", "Line current table", "N/A", "No line-current table is available.")
        return summary, findings, actions

    i_col = _first_existing_column(line_current_table, ["I_pu_mag", "Current_pu_mag", "I_mag_pu"])
    amp_col = _first_existing_column(line_current_table, ["Ampacity_pu", "ampacity_pu"])
    branch_col = _first_existing_column(line_current_table, ["Branch", "branch", "Line"])
    from_col = _first_existing_column(line_current_table, ["From_Bus", "from_bus", "From"])
    to_col = _first_existing_column(line_current_table, ["To_Bus", "to_bus", "To"])

    if i_col is None or amp_col is None:
        _add_finding(findings, "Info", "Line loading", "Loading percentage", "N/A", "Line loading percentage could not be computed because I_pu_mag or Ampacity_pu is missing.")
        return summary, findings, actions

    if "Ampacity_verified" not in line_current_table.columns or not line_current_table["Ampacity_verified"].fillna(False).astype(bool).all():
        _add_finding(findings, "Info", "Line loading", "Thermal ratings", "Not evaluated",
                     "At least one branch lacks an independently verified ampacity rating; no thermal pass/overload claim is made.")
        return summary, findings, actions

    tmp = line_current_table.copy()
    tmp["__i__"] = pd.to_numeric(tmp[i_col], errors="coerce")
    tmp["__amp__"] = pd.to_numeric(tmp[amp_col], errors="coerce")
    tmp = tmp[(tmp["__amp__"] > 0) & tmp["__i__"].notna()]
    if tmp.empty:
        _add_finding(findings, "Info", "Line loading", "Loading values", "N/A", "Line loading values could not be parsed.")
        return summary, findings, actions

    tmp["__loading_pct__"] = 100.0 * tmp["__i__"].abs() / tmp["__amp__"].abs()
    max_row = tmp.loc[tmp["__loading_pct__"].idxmax()]
    max_loading = float(max_row["__loading_pct__"])
    branch = str(max_row[branch_col]) if branch_col else "N/A"
    if from_col and to_col:
        element = f"{branch} ({_format_bus(max_row[from_col])} → {_format_bus(max_row[to_col])})"
    else:
        element = branch

    high = tmp[tmp["__loading_pct__"] >= thresholds.high_loading_percent]
    overload = tmp[tmp["__loading_pct__"] >= thresholds.overload_percent]
    summary.update(
        {
            "max_loading_percent": max_loading,
            "max_loading_branch": element,
            "high_loading_count": int(len(high)),
            "overload_count": int(len(overload)),
        }
    )

    if len(overload) > 0:
        _add_finding(
            findings,
            "Critical",
            "Line loading",
            element,
            f"{max_loading:.2f}%",
            f"At least one branch exceeds the overload threshold of {thresholds.overload_percent:.1f}%.",
        )
        _add_action(
            actions,
            "High",
            element,
            "Check branch rating, consider network reconfiguration, load transfer, conductor upgrade, or local generation support downstream.",
            "Overloaded branches are immediate thermal/security concerns.",
        )
    elif len(high) > 0:
        _add_finding(
            findings,
            "Warning",
            "Line loading",
            element,
            f"{max_loading:.2f}%",
            f"At least one branch exceeds the high-loading warning threshold of {thresholds.high_loading_percent:.1f}%.",
        )
        _add_action(
            actions,
            "Medium",
            element,
            "Monitor loading margin and test load transfer, capacitor/DG support, or branch reinforcement scenarios.",
            "High loading may become an overload under peak demand or contingency conditions.",
        )
    else:
        _add_finding(findings, "OK", "Line loading", element, f"{max_loading:.2f}%", "No high-loading or overload condition was detected using the configured thresholds.")

    return summary, findings, actions


def _summarize_losses(losses_table_mva: pd.DataFrame, thresholds: AdvisorThresholds) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"total_p_loss_mw": None, "total_q_loss_mvar": None, "top_loss_branch": None}

    if losses_table_mva is None or losses_table_mva.empty:
        _add_finding(findings, "Info", "Losses", "Loss table", "N/A", "No loss table is available.")
        return summary, findings, actions

    p_col = _first_existing_column(losses_table_mva, ["P_loss_MW", "P_loss", "Loss_MW"])
    q_col = _first_existing_column(losses_table_mva, ["Q_loss_MVAR", "Q_loss_MVAr", "Q_loss", "Loss_MVAR"])
    branch_col = _first_existing_column(losses_table_mva, ["Branch", "branch", "Line"])
    from_col = _first_existing_column(losses_table_mva, ["From_Bus", "from_bus", "From"])
    to_col = _first_existing_column(losses_table_mva, ["To_Bus", "to_bus", "To"])
    if p_col is None:
        _add_finding(findings, "Info", "Losses", "P loss", "N/A", "P-loss column was not found.")
        return summary, findings, actions

    tmp = losses_table_mva.copy()
    tmp["__p__"] = pd.to_numeric(tmp[p_col], errors="coerce").fillna(0.0)
    if q_col:
        tmp["__q__"] = pd.to_numeric(tmp[q_col], errors="coerce").fillna(0.0)
    else:
        tmp["__q__"] = 0.0
    total_p = float(tmp["__p__"].sum())
    total_q = float(tmp["__q__"].sum())
    if tmp["__p__"].abs().sum() > 0:
        top = tmp.loc[tmp["__p__"].abs().idxmax()]
        branch = str(top[branch_col]) if branch_col else "N/A"
        if from_col and to_col:
            top_label = f"{branch} ({_format_bus(top[from_col])} → {_format_bus(top[to_col])})"
        else:
            top_label = branch
        top_p = float(top["__p__"])
    else:
        top_label = "N/A"
        top_p = 0.0

    summary.update({"total_p_loss_mw": total_p, "total_q_loss_mvar": total_q, "top_loss_branch": top_label})
    _add_finding(
        findings,
        "Info",
        "Losses",
        "System total",
        f"P={total_p:.5f} MW, Q={total_q:.5f} MVAr",
        f"Highest branch real-power loss is on {top_label} with {top_p:.5f} MW.",
    )
    if top_label != "N/A":
        _add_action(
            actions,
            "Medium",
            top_label,
            "Use this branch as the first candidate for loss-reduction scenarios: downstream DG, capacitor placement, reconfiguration, or conductor reinforcement.",
            "High branch losses usually indicate high current flow and/or large branch impedance.",
        )
    return summary, findings, actions


def generate_ai_insights(
    power_flow_result: Any,
    project: Any = None,
    thresholds: AdvisorThresholds | None = None,
) -> AIInsightReport:
    """
    Generate engineering diagnosis from the latest power-flow result.

    Parameters
    ----------
    power_flow_result:
        Result object returned by adapter_project_fbs.run_power_flow_project().
    project:
        Optional PowerSystemProject, used only for project metadata.
    thresholds:
        AdvisorThresholds with UI-editable limits.
    """
    if power_flow_result is None:
        raise ValueError("Run power flow before generating AI insights.")

    thresholds = thresholds or AdvisorThresholds()
    thresholds.validate()

    bus_table = getattr(power_flow_result, "bus_table", pd.DataFrame())
    line_current_table = getattr(power_flow_result, "line_current_table", pd.DataFrame())
    losses_table_mva = getattr(power_flow_result, "losses_table_mva", pd.DataFrame())
    pf_summary = dict(getattr(power_flow_result, "summary", {}) or {})

    findings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "project_name": getattr(getattr(project, "settings", None), "name", "N/A"),
        "solver_method": pf_summary.get("solver_method", getattr(power_flow_result, "raw", {}).get("solver_method", "N/A")),
        "converged": bool(getattr(power_flow_result, "converged", pf_summary.get("converged", False))),
        "message": str(getattr(power_flow_result, "message", pf_summary.get("message", ""))),
        "iterations": pf_summary.get("iterations", "N/A"),
        "number_of_buses": pf_summary.get("number_of_buses", len(getattr(project, "buses", []) or [])),
        "number_of_branches": pf_summary.get("number_of_branches", len(getattr(project, "branches", []) or [])),
    }

    if not summary["converged"]:
        _add_finding(
            findings,
            "Critical",
            "Convergence",
            "Power flow",
            "Not converged",
            "The numerical result should not be used for engineering conclusions until convergence is achieved.",
        )
        _add_action(
            actions,
            "High",
            "Solver setup / project data",
            "Check input data, reduce unrealistic loading, verify branch impedances, try another solver method, or relax tolerance temporarily for diagnosis.",
            "Non-convergence often indicates poor initial data, infeasible loading, wrong sign convention, or a solver/network mismatch.",
        )
    else:
        _add_finding(findings, "OK", "Convergence", "Power flow", "Converged", "The solver converged; result tables can be interpreted.")

    voltage_summary, voltage_findings, voltage_actions = _summarize_voltage(bus_table, thresholds)
    loading_summary, loading_findings, loading_actions = _summarize_loading(line_current_table, thresholds)
    loss_summary, loss_findings, loss_actions = _summarize_losses(losses_table_mva, thresholds)

    summary.update(voltage_summary)
    summary.update(loading_summary)
    summary.update(loss_summary)
    findings.extend(voltage_findings)
    findings.extend(loading_findings)
    findings.extend(loss_findings)
    actions.extend(voltage_actions)
    actions.extend(loading_actions)
    actions.extend(loss_actions)

    if not actions:
        _add_action(
            actions,
            "Low",
            "System",
            "No urgent corrective action is suggested by the configured thresholds. Continue with sensitivity studies or contingency checks.",
            "The base-case power-flow indicators are within the selected engineering limits.",
        )

    findings_df = pd.DataFrame(findings)
    actions_df = pd.DataFrame(actions)

    # Compose readable report.
    vmin = summary.get("min_voltage_pu")
    vmax = summary.get("max_voltage_pu")
    max_loading = summary.get("max_loading_percent")
    total_p_loss = summary.get("total_p_loss_mw")
    total_q_loss = summary.get("total_q_loss_mvar")

    lines: list[str] = []
    lines.append("AI Engineering Advisor")
    lines.append("=" * 25)
    lines.append(f"Project: {summary.get('project_name', 'N/A')}")
    lines.append(f"Solver: {summary.get('solver_method', 'N/A')}")
    lines.append(f"Converged: {'Yes' if summary.get('converged') else 'No'}")
    lines.append(f"Iterations: {summary.get('iterations', 'N/A')}")
    lines.append(f"Buses: {summary.get('number_of_buses', 'N/A')}")
    lines.append(f"Branches: {summary.get('number_of_branches', 'N/A')}")
    lines.append("")
    lines.append("Configured Thresholds")
    lines.append(f"- Voltage lower limit: {thresholds.voltage_min_pu:.3f} pu")
    lines.append(f"- Voltage upper limit: {thresholds.voltage_max_pu:.3f} pu")
    lines.append(f"- Critical low-voltage limit: {thresholds.voltage_critical_min_pu:.3f} pu")
    lines.append(f"- High-loading warning: {thresholds.high_loading_percent:.1f}%")
    lines.append(f"- Overload limit: {thresholds.overload_percent:.1f}%")
    lines.append("")
    lines.append("System Diagnosis")
    if vmin is not None:
        lines.append(f"- Minimum voltage: {float(vmin):.5f} pu at Bus {summary.get('min_voltage_bus', 'N/A')}")
    else:
        lines.append("- Minimum voltage: N/A")
    if vmax is not None:
        lines.append(f"- Maximum voltage: {float(vmax):.5f} pu at Bus {summary.get('max_voltage_bus', 'N/A')}")
    else:
        lines.append("- Maximum voltage: N/A")
    if max_loading is not None:
        lines.append(f"- Maximum branch loading: {float(max_loading):.2f}% at {summary.get('max_loading_branch', 'N/A')}")
    else:
        lines.append("- Maximum branch loading: N/A")
    if total_p_loss is not None:
        lines.append(f"- Total real loss: {float(total_p_loss):.5f} MW")
    else:
        lines.append("- Total real loss: N/A")
    if total_q_loss is not None:
        lines.append(f"- Total reactive loss: {float(total_q_loss):.5f} MVAr")
    else:
        lines.append("- Total reactive loss: N/A")
    lines.append("")

    # High-level diagnosis paragraph.
    critical_count = int((findings_df["Severity"] == "Critical").sum()) if not findings_df.empty else 0
    warning_count = int((findings_df["Severity"] == "Warning").sum()) if not findings_df.empty else 0
    if critical_count:
        lines.append("Overall Assessment: Critical issues were detected. Review the high-priority actions before using this case for planning conclusions.")
    elif warning_count:
        lines.append("Overall Assessment: The case is solvable, but warning-level issues should be investigated before final engineering decisions.")
    else:
        lines.append("Overall Assessment: No critical or warning-level issue was detected using the configured thresholds.")
    lines.append("")
    lines.append("Top Findings")
    for _, row in findings_df.head(8).iterrows():
        lines.append(f"- [{row['Severity']}] {row['Category']} / {row['Item']}: {row['Message']}")
    lines.append("")
    lines.append("Recommended Actions")
    for _, row in actions_df.head(8).iterrows():
        lines.append(f"- [{row['Priority']}] {row['Target']}: {row['Recommended Action']}")
    lines.append("")
    lines.append("Note: This advisor is a rule-based decision-support layer. The numerical power-flow solver remains the source of truth.")

    return AIInsightReport(
        report_text="\n".join(lines),
        findings_table=findings_df,
        actions_table=actions_df,
        summary=summary,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    print("ai_advisor.py loaded. Import generate_ai_insights() from the UI.")
