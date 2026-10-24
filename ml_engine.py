"""
ml_engine.py

Gradient-boosted-tree machine-learning assistant for the Power System Simulator UI.

Purpose
-------
This module does not replace Newton-Raphson or FBS power flow.  It creates a
case/topology-specific surrogate model by repeatedly perturbing the current
project load data, solving each scenario with the deterministic solver, and
training scikit-learn GradientBoosting models to predict screening KPIs:

- Minimum bus voltage
- Bus where minimum voltage occurs
- Total real power loss
- Voltage-violation flag
- Overload flag

Recommended first scope
-----------------------
Fixed-topology radial distribution feeders, especially the IEEE 33-bus feeder,
using the Forward/Backward Sweep solver as the trusted source of labels.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from project_model import PowerSystemProject
except Exception:  # pragma: no cover
    PowerSystemProject = Any  # type: ignore


# -----------------------------------------------------------------------------
# Lightweight fallback estimators for constant targets
# -----------------------------------------------------------------------------

class ConstantRegressor:
    """Pickle-safe constant regressor used when a target has no variation."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


class ConstantClassifier:
    """Pickle-safe constant classifier used when only one class is present."""

    def __init__(self, value: Any):
        self.value = value
        self.classes_ = np.array([value])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray([self.value] * len(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


@dataclass
class MLModelBundle:
    """Container for trained ML surrogate models and metadata."""

    feature_columns: List[str]
    bus_ids: List[str]
    models: Dict[str, Any]
    metrics_table: pd.DataFrame
    training_summary: Dict[str, Any]
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    scenario_config: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Project and feature helpers
# -----------------------------------------------------------------------------

def _sorted_bus_ids(project: PowerSystemProject) -> List[str]:
    ids = [str(b.bus_id).strip() for b in project.buses]
    try:
        return sorted(ids, key=lambda x: int(x))
    except Exception:
        return sorted(ids)


def _project_signature(project: PowerSystemProject) -> Dict[str, Any]:
    """Return topology metadata that should match when a model is reused."""
    bus_ids = _sorted_bus_ids(project)
    edges = sorted((str(br.from_bus).strip(), str(br.to_bus).strip()) for br in project.branches)
    return {
        "bus_ids": bus_ids,
        "number_of_buses": len(bus_ids),
        "number_of_branches": len(project.branches),
        "edges": edges,
        "base_power_mva": float(project.settings.base_power_mva),
    }


def clone_project(project: PowerSystemProject) -> PowerSystemProject:
    """Deep-copy a project without relying on UI state."""
    if hasattr(project, "to_dict") and hasattr(project.__class__, "from_dict"):
        return project.__class__.from_dict(project.to_dict())
    return copy.deepcopy(project)


def build_feature_row(project: PowerSystemProject) -> Dict[str, float]:
    """
    Build one tabular feature row from the project.

    Features are deliberately transparent and topology-specific.  A model trained
    on one bus set must not be blindly used on another topology.
    """
    bus_by_id = {str(b.bus_id).strip(): b for b in project.buses}
    bus_ids = _sorted_bus_ids(project)
    row: Dict[str, float] = {}

    total_p_load = 0.0
    total_q_load = 0.0
    total_abs_q_load = 0.0
    total_p_gen = 0.0
    max_p_load = 0.0
    weighted_p_distance = 0.0

    for bus_id in bus_ids:
        bus = bus_by_id[bus_id]
        p = float(bus.p_load_pu)
        q = float(bus.q_load_pu)
        pg = float(bus.p_gen_pu)
        row[f"p_load_bus_{bus_id}"] = p
        row[f"q_load_bus_{bus_id}"] = q
        row[f"p_gen_bus_{bus_id}"] = pg
        total_p_load += p
        total_q_load += q
        total_abs_q_load += abs(q)
        total_p_gen += pg
        max_p_load = max(max_p_load, p)
        try:
            weighted_p_distance += int(bus_id) * p
        except Exception:
            weighted_p_distance += p

    row["total_p_load_pu"] = total_p_load
    row["total_q_load_pu"] = total_q_load
    row["total_abs_q_load_pu"] = total_abs_q_load
    row["total_p_gen_pu"] = total_p_gen
    row["net_p_load_pu"] = total_p_load - total_p_gen
    row["max_p_load_pu"] = max_p_load
    row["p_load_center_index"] = weighted_p_distance / total_p_load if abs(total_p_load) > 1e-12 else 0.0
    row["number_of_buses"] = float(len(project.buses))
    row["number_of_branches"] = float(len(project.branches))
    return row


def _feature_columns_from_dataset(dataset: pd.DataFrame) -> List[str]:
    non_feature_prefixes = ("target_",)
    non_feature_columns = {
        "scenario_id",
        "global_load_multiplier",
        "solver_method",
        "converged",
        "iterations",
        "message",
        "status",
        "error",
    }
    features: List[str] = []
    for col in dataset.columns:
        if col in non_feature_columns:
            continue
        if any(str(col).startswith(prefix) for prefix in non_feature_prefixes):
            continue
        # Keep numeric feature columns only.
        if pd.api.types.is_numeric_dtype(dataset[col]):
            features.append(str(col))
    return features


def _extract_power_flow_targets(
    result: Any,
    voltage_min_pu: float,
    voltage_max_pu: float,
) -> Dict[str, Any]:
    bus_table = getattr(result, "bus_table", pd.DataFrame())
    line_current_table = getattr(result, "line_current_table", pd.DataFrame())
    losses_mva = getattr(result, "losses_table_mva", pd.DataFrame())
    summary = dict(getattr(result, "summary", {}) or {})

    if bus_table is None or bus_table.empty or "Voltage_pu" not in bus_table.columns:
        raise ValueError("Power-flow result does not contain a usable Voltage_pu bus table.")

    tmp = bus_table.copy()
    tmp["__v__"] = pd.to_numeric(tmp["Voltage_pu"], errors="coerce")
    tmp = tmp.dropna(subset=["__v__"])
    if tmp.empty:
        raise ValueError("Power-flow result voltage table contains no numeric voltages.")

    min_row = tmp.loc[tmp["__v__"].idxmin()]
    max_row = tmp.loc[tmp["__v__"].idxmax()]
    vmin = float(min_row["__v__"])
    vmax = float(max_row["__v__"])
    vmin_bus = str(min_row.get("Bus", ""))
    vmax_bus = str(max_row.get("Bus", ""))

    p_loss_mw = summary.get("total_loss_MVA_real", None)
    q_loss_mvar = summary.get("total_loss_MVA_imag", None)
    if (p_loss_mw is None or q_loss_mvar is None) and losses_mva is not None and not losses_mva.empty:
        if "P_loss_MW" in losses_mva.columns:
            p_loss_mw = float(pd.to_numeric(losses_mva["P_loss_MW"], errors="coerce").fillna(0).sum())
        if "Q_loss_MVAR" in losses_mva.columns:
            q_loss_mvar = float(pd.to_numeric(losses_mva["Q_loss_MVAR"], errors="coerce").fillna(0).sum())

    overload = False
    ratings_verified = bool(line_current_table is not None and not line_current_table.empty and
                            "Ampacity_verified" in line_current_table.columns and
                            line_current_table["Ampacity_verified"].fillna(False).astype(bool).all())
    if ratings_verified:
        if "Overcurrent" in line_current_table.columns:
            overload = bool(line_current_table["Overcurrent"].astype(bool).any())
        elif {"I_pu_mag", "Ampacity_pu"}.issubset(line_current_table.columns):
            i_mag = pd.to_numeric(line_current_table["I_pu_mag"], errors="coerce")
            amp = pd.to_numeric(line_current_table["Ampacity_pu"], errors="coerce")
            overload = bool((i_mag > amp).fillna(False).any())

    return {
        "target_vmin_pu": vmin,
        "target_vmax_pu": vmax,
        "target_vmin_bus": vmin_bus,
        "target_vmax_bus": vmax_bus,
        "target_total_p_loss_mw": float(p_loss_mw) if p_loss_mw is not None else np.nan,
        "target_total_q_loss_mvar": float(q_loss_mvar) if q_loss_mvar is not None else np.nan,
        "target_voltage_violation": int(vmin < voltage_min_pu or vmax > voltage_max_pu),
        "target_overload": int(overload) if ratings_verified else np.nan,
    }


def _perturb_loads(
    project: PowerSystemProject,
    rng: np.random.Generator,
    global_load_min: float,
    global_load_max: float,
    local_variation_percent: float,
) -> Tuple[PowerSystemProject, float]:
    scenario = clone_project(project)
    global_multiplier = float(rng.uniform(global_load_min, global_load_max))
    local_span = max(0.0, float(local_variation_percent)) / 100.0

    for bus in scenario.buses:
        bus_type = str(bus.bus_type).strip().lower()
        if bus_type == "slack":
            continue
        local_multiplier = float(rng.uniform(1.0 - local_span, 1.0 + local_span)) if local_span > 0 else 1.0
        scale = global_multiplier * local_multiplier
        bus.p_load_pu = float(bus.p_load_pu) * scale
        bus.q_load_pu = float(bus.q_load_pu) * scale

    return scenario, global_multiplier


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------

def generate_training_dataset(
    base_project: PowerSystemProject,
    n_scenarios: int = 500,
    solver_method: str = "Forward/Backward Sweep",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    global_load_min: float = 0.5,
    global_load_max: float = 1.5,
    local_variation_percent: float = 20.0,
    voltage_min_pu: float = 0.95,
    voltage_max_pu: float = 1.05,
    random_seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a supervised ML dataset by solving randomized operating scenarios.

    Returns (dataset, summary). Failed/non-convergent scenarios are skipped but
    counted in the summary.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")
    if global_load_min <= 0 or global_load_max <= 0 or global_load_max <= global_load_min:
        raise ValueError("Load multiplier range must satisfy 0 < min < max.")
    if voltage_min_pu <= 0 or voltage_max_pu <= voltage_min_pu:
        raise ValueError("Voltage limits must satisfy 0 < min < max.")

    from adapter_project_fbs import run_power_flow_project

    rng = np.random.default_rng(int(random_seed))
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for i in range(1, int(n_scenarios) + 1):
        scenario_project, global_multiplier = _perturb_loads(
            base_project,
            rng=rng,
            global_load_min=global_load_min,
            global_load_max=global_load_max,
            local_variation_percent=local_variation_percent,
        )
        try:
            result = run_power_flow_project(
                scenario_project,
                max_iterations=max_iterations,
                tolerance=tolerance,
                suppress_output=True,
                solver_method=solver_method,
            )
            if not bool(getattr(result, "converged", False)):
                failures.append(f"Scenario {i}: solver did not converge; {getattr(result, 'message', '')}")
                continue
            row = build_feature_row(scenario_project)
            row.update(_extract_power_flow_targets(result, voltage_min_pu, voltage_max_pu))
            row.update(
                {
                    "scenario_id": i,
                    "global_load_multiplier": global_multiplier,
                    "solver_method": result.summary.get("solver_method", solver_method),
                    "converged": True,
                    "iterations": int(result.summary.get("iterations", 0) or 0),
                    "message": str(result.message),
                }
            )
            rows.append(row)
        except Exception as exc:
            failures.append(f"Scenario {i}: {exc}")
        if progress_callback:
            progress_callback(i, int(n_scenarios))

    dataset = pd.DataFrame(rows)
    summary = {
        "requested_scenarios": int(n_scenarios),
        "successful_scenarios": int(len(dataset)),
        "failed_scenarios": int(len(failures)),
        "failure_examples": failures[:10],
        "solver_method": solver_method,
        "voltage_min_pu": float(voltage_min_pu),
        "voltage_max_pu": float(voltage_max_pu),
        "global_load_min": float(global_load_min),
        "global_load_max": float(global_load_max),
        "local_variation_percent": float(local_variation_percent),
        "random_seed": int(random_seed),
        "project_signature": _project_signature(base_project),
    }
    if dataset.empty:
        raise ValueError("No successful scenarios were generated. Check solver eligibility and project data.")
    return dataset, summary


# -----------------------------------------------------------------------------
# Model training and prediction
# -----------------------------------------------------------------------------

def _train_regressor(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int) -> Any:
    if pd.to_numeric(y_train, errors="coerce").nunique(dropna=True) <= 1:
        return ConstantRegressor(float(pd.to_numeric(y_train, errors="coerce").dropna().iloc[0]))
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=int(random_seed),
    ).fit(X_train, y_train)


def _train_classifier(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int) -> Any:
    if y_train.nunique(dropna=True) <= 1:
        return ConstantClassifier(y_train.dropna().iloc[0])
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=int(random_seed),
    ).fit(X_train, y_train)


def train_gradient_boosting_models(
    dataset: pd.DataFrame,
    voltage_min_pu: float = 0.95,
    voltage_max_pu: float = 1.05,
    test_size: float = 0.2,
    random_seed: int = 42,
    scenario_summary: Optional[Dict[str, Any]] = None,
) -> MLModelBundle:
    """Train gradient-boosted decision-tree models from a generated dataset."""
    if dataset is None or dataset.empty:
        raise ValueError("Dataset is empty. Generate a dataset before training the ML model.")

    required_targets = [
        "target_vmin_pu",
        "target_total_p_loss_mw",
        "target_voltage_violation",
        "target_vmin_bus",
    ]
    missing = [c for c in required_targets if c not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required target columns: {missing}")

    data = dataset.dropna(subset=["target_vmin_pu", "target_total_p_loss_mw"]).copy()
    if len(data) < 30:
        raise ValueError("At least 30 successful scenarios are recommended for training. Generate more scenarios.")

    feature_columns = _feature_columns_from_dataset(data)
    if not feature_columns:
        raise ValueError("No numeric feature columns found in the dataset.")

    X = data[feature_columns].astype(float)
    bus_ids = []
    if scenario_summary and "project_signature" in scenario_summary:
        bus_ids = list(scenario_summary["project_signature"].get("bus_ids", []))
    if not bus_ids:
        bus_ids = sorted([c.replace("p_load_bus_", "") for c in feature_columns if c.startswith("p_load_bus_")], key=lambda x: int(x) if x.isdigit() else x)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

    # One split reused for all targets to keep comparisons consistent.
    indices = np.arange(len(data))
    if len(data) >= 50:
        train_idx, test_idx = train_test_split(indices, test_size=float(test_size), random_state=int(random_seed))
    else:
        train_idx, test_idx = train_test_split(indices, test_size=max(1, int(0.2 * len(data))), random_state=int(random_seed))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    models: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []

    def add_regression_model(target_col: str, model_key: str, label: str, unit: str, ok_mae: float, warn_mae: float) -> None:
        y = pd.to_numeric(data[target_col], errors="coerce")
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = _train_regressor(X_train, y_train, random_seed)
        if not isinstance(model, ConstantRegressor):
            # _train_regressor already fits non-constant models.
            pass
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        try:
            r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else np.nan
        except Exception:
            r2 = np.nan
        status = "OK" if mae <= ok_mae else "Warning" if mae <= warn_mae else "Critical"
        models[model_key] = model
        metric_rows.append({"Target": label, "Model": type(model).__name__, "Metric": "MAE", "Value": mae, "Unit": unit, "Status": status})
        metric_rows.append({"Target": label, "Model": type(model).__name__, "Metric": "R2", "Value": r2, "Unit": "", "Status": "Info"})

    def add_classifier_model(target_col: str, model_key: str, label: str) -> None:
        y = data[target_col].astype(str if target_col.endswith("_bus") else int)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = _train_classifier(X_train, y_train, random_seed)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        status = "OK" if acc >= 0.95 else "Warning" if acc >= 0.85 else "Critical"
        models[model_key] = model
        metric_rows.append({"Target": label, "Model": type(model).__name__, "Metric": "Accuracy", "Value": acc, "Unit": "fraction", "Status": status})
        classes = getattr(model, "classes_", [])
        metric_rows.append({"Target": label, "Model": type(model).__name__, "Metric": "Classes", "Value": ", ".join(map(str, classes)), "Unit": "", "Status": "Info"})

    add_regression_model("target_vmin_pu", "vmin_regressor", "Minimum voltage", "pu", ok_mae=0.003, warn_mae=0.01)
    add_regression_model("target_total_p_loss_mw", "p_loss_regressor", "Total real loss", "MW", ok_mae=0.01, warn_mae=0.05)
    if "target_total_q_loss_mvar" in data.columns and pd.to_numeric(data["target_total_q_loss_mvar"], errors="coerce").notna().sum() > 0:
        add_regression_model("target_total_q_loss_mvar", "q_loss_regressor", "Total reactive loss", "MVAr", ok_mae=0.01, warn_mae=0.05)
    add_classifier_model("target_voltage_violation", "voltage_violation_classifier", "Voltage violation")
    if "target_overload" in data.columns and data["target_overload"].notna().all():
        add_classifier_model("target_overload", "overload_classifier", "Overload risk")
    else:
        metric_rows.append({"Target": "Overload risk", "Model": "Not trained", "Metric": "Thermal status",
                            "Value": "Not evaluated: unverified ampacity", "Unit": "", "Status": "Info"})
    add_classifier_model("target_vmin_bus", "weakest_bus_classifier", "Weakest-voltage bus")

    metrics_table = pd.DataFrame(metric_rows)
    training_summary = {
        "training_rows": int(len(data)),
        "test_rows": int(len(test_idx)),
        "feature_count": int(len(feature_columns)),
        "model_family": "GradientBoosting",
        "voltage_min_pu": float(voltage_min_pu),
        "voltage_max_pu": float(voltage_max_pu),
        "scenario_summary": scenario_summary or {},
    }
    return MLModelBundle(
        feature_columns=feature_columns,
        bus_ids=bus_ids,
        models=models,
        metrics_table=metrics_table,
        training_summary=training_summary,
        voltage_min_pu=float(voltage_min_pu),
        voltage_max_pu=float(voltage_max_pu),
        scenario_config=scenario_summary or {},
    )


def _probability_for_positive(model: Any, X: pd.DataFrame, positive_label: Any = 1) -> Optional[float]:
    try:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        if positive_label in classes:
            return float(proba[0, classes.index(positive_label)])
        if str(positive_label) in classes:
            return float(proba[0, classes.index(str(positive_label))])
        # Constant negative classifier.
        return 0.0
    except Exception:
        return None


def predict_current_case(bundle: MLModelBundle, project: PowerSystemProject) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict screening KPIs for the current UI project."""
    current_bus_ids = _sorted_bus_ids(project)
    if list(bundle.bus_ids) and list(bundle.bus_ids) != current_bus_ids:
        raise ValueError(
            "This ML model was trained for a different bus set. "
            f"Model buses={bundle.bus_ids}; current project buses={current_bus_ids}."
        )

    feature_row = build_feature_row(project)
    missing = [col for col in bundle.feature_columns if col not in feature_row]
    if missing:
        raise ValueError(f"Current project is missing ML feature columns required by the model: {missing[:8]}")
    X = pd.DataFrame([{col: feature_row[col] for col in bundle.feature_columns}]).astype(float)

    vmin = float(bundle.models["vmin_regressor"].predict(X)[0])
    p_loss = float(bundle.models["p_loss_regressor"].predict(X)[0])
    q_loss = None
    if "q_loss_regressor" in bundle.models:
        q_loss = float(bundle.models["q_loss_regressor"].predict(X)[0])

    voltage_flag = int(bundle.models["voltage_violation_classifier"].predict(X)[0])
    ratings_verified = all(bool(getattr(b, "ampacity_verified", False)) for b in project.branches)
    overload_flag = (int(bundle.models["overload_classifier"].predict(X)[0])
                     if ratings_verified and "overload_classifier" in bundle.models else None)
    weakest_bus = str(bundle.models["weakest_bus_classifier"].predict(X)[0])
    voltage_prob = _probability_for_positive(bundle.models["voltage_violation_classifier"], X, 1)
    overload_prob = (_probability_for_positive(bundle.models["overload_classifier"], X, 1)
                     if ratings_verified and "overload_classifier" in bundle.models else None)

    rows: List[Dict[str, Any]] = [
        {"Item": "Predicted minimum voltage", "Value": vmin, "Unit": "pu", "Status": "Warning" if vmin < bundle.voltage_min_pu else "OK"},
        {"Item": "Predicted weakest-voltage bus", "Value": weakest_bus, "Unit": "bus", "Status": "Info"},
        {"Item": "Predicted total real loss", "Value": p_loss, "Unit": "MW", "Status": "Info"},
        {"Item": "Predicted voltage violation", "Value": "Yes" if voltage_flag else "No", "Unit": "", "Status": "Warning" if voltage_flag else "OK"},
        {"Item": "Predicted overload risk", "Value": "Not evaluated" if overload_flag is None else ("Yes" if overload_flag else "No"), "Unit": "", "Status": "Info" if overload_flag is None else ("Critical" if overload_flag else "OK")},
    ]
    if q_loss is not None:
        rows.insert(3, {"Item": "Predicted total reactive loss", "Value": q_loss, "Unit": "MVAr", "Status": "Info"})
    if voltage_prob is not None:
        rows.append({"Item": "Voltage-violation probability", "Value": voltage_prob, "Unit": "fraction", "Status": "Warning" if voltage_prob >= 0.5 else "OK"})
    if overload_prob is not None:
        rows.append({"Item": "Overload probability", "Value": overload_prob, "Unit": "fraction", "Status": "Critical" if overload_prob >= 0.5 else "OK"})

    prediction = {
        "vmin_pu": vmin,
        "p_loss_mw": p_loss,
        "q_loss_mvar": q_loss,
        "voltage_violation": voltage_flag,
        "overload": overload_flag,
        "weakest_bus": weakest_bus,
        "voltage_violation_probability": voltage_prob,
        "overload_probability": overload_prob,
    }
    return pd.DataFrame(rows), prediction


def compare_prediction_with_solver(prediction: Dict[str, Any], power_flow_result: Any) -> pd.DataFrame:
    """Build a comparison table between ML prediction and the latest deterministic solver result."""
    if power_flow_result is None:
        return pd.DataFrame({"Status": ["No solver result available. Run power flow for comparison."]})

    actual = _extract_power_flow_targets(
        power_flow_result,
        voltage_min_pu=float("inf"),  # not used for numeric comparison here
        voltage_max_pu=float("inf"),
    )
    rows: List[Dict[str, Any]] = []

    def add_numeric(name: str, pred_key: str, actual_key: str, unit: str, ok_abs: float, warn_abs: float) -> None:
        pred = prediction.get(pred_key, None)
        act = actual.get(actual_key, None)
        if pred is None or act is None or pd.isna(pred) or pd.isna(act):
            return
        err = float(pred) - float(act)
        abs_err = abs(err)
        status = "OK" if abs_err <= ok_abs else "Warning" if abs_err <= warn_abs else "Critical"
        rows.append({"Metric": name, "ML prediction": pred, "Solver result": act, "Error": err, "Absolute error": abs_err, "Unit": unit, "Status": status})

    add_numeric("Minimum voltage", "vmin_pu", "target_vmin_pu", "pu", ok_abs=0.003, warn_abs=0.01)
    add_numeric("Total real loss", "p_loss_mw", "target_total_p_loss_mw", "MW", ok_abs=0.01, warn_abs=0.05)
    if prediction.get("q_loss_mvar") is not None:
        add_numeric("Total reactive loss", "q_loss_mvar", "target_total_q_loss_mvar", "MVAr", ok_abs=0.01, warn_abs=0.05)

    rows.append(
        {
            "Metric": "Weakest-voltage bus",
            "ML prediction": prediction.get("weakest_bus"),
            "Solver result": actual.get("target_vmin_bus"),
            "Error": "",
            "Absolute error": "",
            "Unit": "bus",
            "Status": "OK" if str(prediction.get("weakest_bus")) == str(actual.get("target_vmin_bus")) else "Warning",
        }
    )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

def save_model_bundle(bundle: MLModelBundle, path: str | Path) -> None:
    import joblib

    joblib.dump(bundle, Path(path))


def load_model_bundle(path: str | Path) -> MLModelBundle:
    import joblib

    obj = joblib.load(Path(path))
    if not isinstance(obj, MLModelBundle):
        raise ValueError("The selected file is not a valid MLModelBundle.")
    return obj
