"""
load_forecasting.py

Load-forecasting module for the Power System Simulator UI.

Scope v1
--------
1. Forecast total feeder real/reactive load from historical CSV data.
2. Convert total feeder forecast into bus-level loads using current bus-load shares.
3. Run deterministic power flow for each forecasted hour.
4. Summarize voltage/loss risk over time.
5. Later extension: bus-level forecasting.

This module deliberately keeps the deterministic power-flow solver as the
source of truth. The forecasting model only supplies future loading scenarios.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency. Run: pip install scikit-learn") from exc

from adapter_project_fbs import run_power_flow_project
from project_model import PowerSystemProject


@dataclass
class LoadForecastBundle:
    """Trained total-load forecasting bundle."""

    p_model: Any
    q_model: Optional[Any]
    feature_columns: List[str]
    history: pd.DataFrame
    frequency_hours: int
    q_to_p_ratio: float
    metrics: pd.DataFrame
    timestamp_column: str = "timestamp"
    p_column: str = "total_p_mw"
    q_column: str = "total_q_mvar"
    temperature_column: str = "temperature_c"


# -----------------------------------------------------------------------------
# CSV preparation and feature engineering
# -----------------------------------------------------------------------------


def load_history_csv(path: str) -> pd.DataFrame:
    """Load and validate historical total feeder load CSV."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Historical load CSV must contain a 'timestamp' column.")
    if "total_p_mw" not in df.columns:
        raise ValueError("Historical load CSV must contain a 'total_p_mw' column.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["total_p_mw"] = pd.to_numeric(out["total_p_mw"], errors="coerce")
    if "total_q_mvar" in out.columns:
        out["total_q_mvar"] = pd.to_numeric(out["total_q_mvar"], errors="coerce")
    if "temperature_c" in out.columns:
        out["temperature_c"] = pd.to_numeric(out["temperature_c"], errors="coerce")

    out = out.dropna(subset=["timestamp", "total_p_mw"]).sort_values("timestamp").reset_index(drop=True)
    # 168 lagged rows are consumed before fitting; at least 30 eligible examples
    # are required by _make_features(). Reject short histories at admission.
    if len(out) < 198:
        raise ValueError("At least 198 consecutive hourly observations are required (168 lag rows + 30 model rows).")
    if out["timestamp"].duplicated().any():
        raise ValueError("Duplicate historical timestamps detected. Resolve duplicates before training.")
    if not out["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("The forecasting model requires consecutive hourly timestamps. Repair gaps/irregular sampling before import.")
    if out["total_p_mw"].le(0).any():
        raise ValueError("total_p_mw must be positive for all historical records.")
    return out


def _infer_frequency_hours(history: pd.DataFrame) -> int:
    times = pd.to_datetime(history["timestamp"]).sort_values()
    diffs = times.diff().dropna()
    if diffs.empty:
        return 1
    median_hours = diffs.median().total_seconds() / 3600.0
    if median_hours <= 0:
        return 1
    return max(1, int(round(median_hours)))


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    return out


def _make_features(history: pd.DataFrame, target_col: str, include_temperature: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Create lag/time features for a target column."""
    out = _add_time_features(history)
    for lag in [1, 2, 3, 24, 48, 168]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    out["rolling_24_mean"] = out[target_col].shift(1).rolling(24).mean()
    out["rolling_24_std"] = out[target_col].shift(1).rolling(24).std()
    out["rolling_168_mean"] = out[target_col].shift(1).rolling(168).mean()

    feature_cols = [
        "hour", "dayofweek", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
        "rolling_24_mean", "rolling_24_std", "rolling_168_mean",
    ]
    if include_temperature and "temperature_c" in out.columns:
        out["temperature_c"] = out["temperature_c"].ffill().bfill()
        feature_cols.append("temperature_c")

    model_df = out.dropna(subset=feature_cols + [target_col]).copy()
    if len(model_df) < 30:
        raise ValueError(
            "Not enough rows after creating lag features. Use a longer hourly history, ideally > 200 records."
        )
    return model_df, feature_cols


def _metrics_frame(target_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "Target": target_name,
        "Rows": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def train_total_load_forecaster(
    history: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 300,
    learning_rate: float = 0.05,
) -> LoadForecastBundle:
    """Train GradientBoostingRegressor models for feeder P and optionally Q."""
    if not (0.05 <= float(test_size) <= 0.5):
        raise ValueError("test_size must be between 0.05 and 0.5.")

    hist = history.copy().sort_values("timestamp").reset_index(drop=True)
    include_temp = "temperature_c" in hist.columns
    p_df, feature_cols = _make_features(hist, "total_p_mw", include_temperature=include_temp)

    split = max(1, int(len(p_df) * (1 - test_size)))
    train_p = p_df.iloc[:split]
    test_p = p_df.iloc[split:]
    if len(test_p) < 5:
        raise ValueError("Test set is too small. Use more historical records or reduce test_size.")

    p_model = GradientBoostingRegressor(
        n_estimators=max_iter,
        learning_rate=learning_rate,
        max_depth=3,
        random_state=random_state,
    )
    p_model.fit(train_p[feature_cols], train_p["total_p_mw"])
    p_pred = p_model.predict(test_p[feature_cols])
    metric_rows = [_metrics_frame("total_p_mw", test_p["total_p_mw"].to_numpy(), p_pred)]

    q_model = None
    q_to_p_ratio = 0.0
    if "total_q_mvar" in hist.columns and hist["total_q_mvar"].notna().sum() >= 48:
        q_hist = hist.dropna(subset=["total_q_mvar"]).copy()
        # Signed Q is allowed. Ratio preserves the user's sign convention.
        valid_ratio = q_hist["total_p_mw"].abs() > 1e-9
        if valid_ratio.any():
            q_to_p_ratio = float((q_hist.loc[valid_ratio, "total_q_mvar"] / q_hist.loc[valid_ratio, "total_p_mw"]).median())
        q_df, q_feature_cols = _make_features(q_hist, "total_q_mvar", include_temperature=include_temp)
        # Use the same feature columns if possible. This keeps forecasting logic simple.
        missing = [c for c in feature_cols if c not in q_df.columns]
        if not missing and len(q_df) >= 40:
            split_q = max(1, int(len(q_df) * (1 - test_size)))
            train_q = q_df.iloc[:split_q]
            test_q = q_df.iloc[split_q:]
            if len(test_q) >= 5:
                q_model = GradientBoostingRegressor(
                    n_estimators=max_iter,
                    learning_rate=learning_rate,
                    max_depth=3,
                    random_state=random_state,
                )
                q_model.fit(train_q[feature_cols], train_q["total_q_mvar"])
                q_pred = q_model.predict(test_q[feature_cols])
                metric_rows.append(_metrics_frame("total_q_mvar", test_q["total_q_mvar"].to_numpy(), q_pred))
    else:
        q_to_p_ratio = 0.0

    if q_model is None and "total_q_mvar" in hist.columns:
        valid_ratio = hist["total_p_mw"].abs() > 1e-9
        if valid_ratio.any():
            q_to_p_ratio = float((hist.loc[valid_ratio, "total_q_mvar"] / hist.loc[valid_ratio, "total_p_mw"]).median())

    frequency_hours = _infer_frequency_hours(hist)
    metrics = pd.DataFrame(metric_rows)
    return LoadForecastBundle(
        p_model=p_model,
        q_model=q_model,
        feature_columns=feature_cols,
        history=hist,
        frequency_hours=frequency_hours,
        q_to_p_ratio=q_to_p_ratio,
        metrics=metrics,
    )


# -----------------------------------------------------------------------------
# Forecasting
# -----------------------------------------------------------------------------


def _feature_row_from_history(history: pd.DataFrame, target_col: str, timestamp: pd.Timestamp, feature_cols: List[str]) -> pd.DataFrame:
    """Build one feature row for recursive forecasting."""
    hist = history.sort_values("timestamp").reset_index(drop=True)
    values = hist[target_col].astype(float).to_numpy()
    if len(values) < 168:
        raise ValueError("Need at least 168 hourly records for lag_168 forecasting feature.")

    row: Dict[str, Any] = {
        "timestamp": timestamp,
        "hour": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        "month": timestamp.month,
        "is_weekend": int(timestamp.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * timestamp.dayofweek / 7.0),
        "dow_cos": np.cos(2 * np.pi * timestamp.dayofweek / 7.0),
        "lag_1": values[-1],
        "lag_2": values[-2],
        "lag_3": values[-3],
        "lag_24": values[-24],
        "lag_48": values[-48],
        "lag_168": values[-168],
        "rolling_24_mean": float(np.mean(values[-24:])),
        "rolling_24_std": float(np.std(values[-24:], ddof=1)) if len(values[-24:]) > 1 else 0.0,
        "rolling_168_mean": float(np.mean(values[-168:])),
    }
    if "temperature_c" in feature_cols:
        if "temperature_c" in hist.columns and hist["temperature_c"].notna().any():
            row["temperature_c"] = float(hist["temperature_c"].ffill().iloc[-1])
        else:
            row["temperature_c"] = 25.0
    return pd.DataFrame([{c: row[c] for c in feature_cols}])


def forecast_total_load(bundle: LoadForecastBundle, horizon_hours: int = 24) -> pd.DataFrame:
    """Forecast total feeder P/Q recursively for the requested horizon."""
    horizon_hours = int(horizon_hours)
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")
    if horizon_hours > 24 * 14:
        raise ValueError("v1 supports up to 14 days of hourly forecast to keep computation reasonable.")

    p_hist = bundle.history.copy().sort_values("timestamp").reset_index(drop=True)
    q_hist = p_hist.copy() if "total_q_mvar" in p_hist.columns else None
    last_ts = pd.to_datetime(p_hist["timestamp"].iloc[-1])
    freq = pd.Timedelta(hours=bundle.frequency_hours)

    rows: List[Dict[str, Any]] = []
    for step in range(1, horizon_hours + 1):
        ts = last_ts + step * freq
        p_x = _feature_row_from_history(p_hist, "total_p_mw", ts, bundle.feature_columns)
        p_pred = float(bundle.p_model.predict(p_x)[0])
        p_pred = max(0.0, p_pred)

        if bundle.q_model is not None and q_hist is not None and "total_q_mvar" in q_hist.columns:
            q_x = _feature_row_from_history(q_hist, "total_q_mvar", ts, bundle.feature_columns)
            q_pred = float(bundle.q_model.predict(q_x)[0])
        else:
            q_pred = float(p_pred * bundle.q_to_p_ratio)

        rows.append({
            "timestamp": ts,
            "forecast_step": step,
            "forecast_total_p_mw": p_pred,
            "forecast_total_q_mvar": q_pred,
        })

        # Recursive update.
        new_row = {"timestamp": ts, "total_p_mw": p_pred}
        if "temperature_c" in p_hist.columns:
            new_row["temperature_c"] = float(p_hist["temperature_c"].ffill().iloc[-1])
        p_hist = pd.concat([p_hist, pd.DataFrame([new_row])], ignore_index=True)

        if q_hist is not None and "total_q_mvar" in q_hist.columns:
            q_new = {"timestamp": ts, "total_q_mvar": q_pred}
            if "temperature_c" in q_hist.columns:
                q_new["temperature_c"] = float(q_hist["temperature_c"].ffill().iloc[-1])
            q_hist = pd.concat([q_hist, pd.DataFrame([q_new])], ignore_index=True)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Forecasted power-flow simulation
# -----------------------------------------------------------------------------


def apply_total_load_forecast_to_project(
    project: PowerSystemProject,
    forecast_total_p_mw: float,
    forecast_total_q_mvar: Optional[float] = None,
) -> PowerSystemProject:
    """Scale bus loads according to current bus-load shares."""
    out = copy.deepcopy(project)
    base_mva = float(out.settings.base_power_mva)
    if base_mva <= 0:
        raise ValueError("Project base_power_mva must be positive.")

    load_buses = [b for b in out.buses if str(b.bus_type).strip().lower() != "slack"]
    current_p_sum = float(sum(float(b.p_load_pu) for b in load_buses))
    current_q_sum = float(sum(float(b.q_load_pu) for b in load_buses))
    if abs(current_p_sum) < 1e-12:
        raise ValueError("Current project has zero total non-slack P load; bus-load shares cannot be computed.")

    target_p_pu = float(forecast_total_p_mw) / base_mva
    p_scale = target_p_pu / current_p_sum

    if forecast_total_q_mvar is None:
        q_scale = p_scale
    else:
        if abs(current_q_sum) < 1e-12:
            if abs(float(forecast_total_q_mvar)) > 1e-9:
                raise ValueError("Nonzero forecast Q cannot be allocated: baseline project Q totals zero.")
            q_scale = 0.0
        else:
            target_q_pu = float(forecast_total_q_mvar) / base_mva
            q_scale = target_q_pu / current_q_sum

    for bus in out.buses:
        if str(bus.bus_type).strip().lower() == "slack":
            continue
        bus.p_load_pu = float(bus.p_load_pu) * p_scale
        bus.q_load_pu = float(bus.q_load_pu) * q_scale
    return out


def _extract_power_flow_risk(
    result: Any,
    timestamp: Any,
    p_mw: float,
    q_mvar: float,
    vmin_limit: float,
    vmax_limit: float,
) -> Dict[str, Any]:
    bus_table = getattr(result, "bus_table", pd.DataFrame())
    losses_mva = getattr(result, "losses_table_mva", pd.DataFrame())
    current_table = getattr(result, "line_current_table", pd.DataFrame())
    summary = dict(getattr(result, "summary", {}) or {})

    vmin = np.nan
    vmax = np.nan
    vmin_bus = "N/A"
    vmax_bus = "N/A"
    if bus_table is not None and not bus_table.empty:
        v_col = next((c for c in ["Voltage_pu", "Voltage_mag_pu", "V_pu", "Voltage"] if c in bus_table.columns), None)
        b_col = next((c for c in ["Bus", "bus_id", "Bus_ID"] if c in bus_table.columns), None)
        if v_col is not None:
            tmp = bus_table.copy()
            tmp["__v__"] = pd.to_numeric(tmp[v_col], errors="coerce")
            tmp = tmp.dropna(subset=["__v__"])
            if not tmp.empty:
                min_row = tmp.loc[tmp["__v__"].idxmin()]
                max_row = tmp.loc[tmp["__v__"].idxmax()]
                vmin = float(min_row["__v__"])
                vmax = float(max_row["__v__"])
                vmin_bus = str(min_row[b_col]) if b_col else "N/A"
                vmax_bus = str(max_row[b_col]) if b_col else "N/A"

    p_loss = summary.get("total_loss_MVA_real", np.nan)
    q_loss = summary.get("total_loss_MVA_imag", np.nan)
    if (pd.isna(p_loss) or pd.isna(q_loss)) and losses_mva is not None and not losses_mva.empty:
        if "P_loss_MW" in losses_mva.columns:
            p_loss = pd.to_numeric(losses_mva["P_loss_MW"], errors="coerce").sum()
        if "Q_loss_MVAR" in losses_mva.columns:
            q_loss = pd.to_numeric(losses_mva["Q_loss_MVAR"], errors="coerce").sum()

    max_loading_pct = np.nan
    overload = False
    thermal_status = "Not evaluated"
    if current_table is not None and not current_table.empty and "Ampacity_verified" in current_table.columns:
        rating_ok = current_table["Ampacity_verified"].fillna(False).astype(bool)
    else:
        rating_ok = pd.Series(dtype=bool)
    if current_table is not None and not current_table.empty and len(rating_ok) and rating_ok.all():
        thermal_status = "Evaluated"
        if "I_pu_mag" in current_table.columns and "Ampacity_pu" in current_table.columns:
            amp = pd.to_numeric(current_table["Ampacity_pu"], errors="coerce").replace(0, np.nan)
            cur = pd.to_numeric(current_table["I_pu_mag"], errors="coerce")
            loading = (cur / amp) * 100.0
            if loading.notna().any():
                max_loading_pct = float(loading.max())
                overload = bool((loading > 100.0).any())
        elif "Overcurrent" in current_table.columns:
            overload = bool(current_table["Overcurrent"].astype(bool).any())

    voltage_violation = bool((not pd.isna(vmin) and vmin < vmin_limit) or (not pd.isna(vmax) and vmax > vmax_limit))
    severity = "OK"
    if not bool(getattr(result, "converged", False)) or not np.isfinite(vmin) or not np.isfinite(vmax):
        severity = "Error"
    elif overload or (not pd.isna(vmin) and vmin < 0.90):
        severity = "Critical"
    elif voltage_violation:
        severity = "Warning"

    return {
        "timestamp": timestamp,
        "Severity": severity,
        "forecast_total_p_mw": float(p_mw),
        "forecast_total_q_mvar": float(q_mvar),
        "converged": bool(getattr(result, "converged", False)),
        "solver_method": summary.get("solver_method", "N/A"),
        "vmin_pu": vmin,
        "vmin_bus": vmin_bus,
        "vmax_pu": vmax,
        "vmax_bus": vmax_bus,
        "voltage_violation": voltage_violation,
        "total_p_loss_mw": p_loss,
        "total_q_loss_mvar": q_loss,
        "max_loading_percent": max_loading_pct,
        "overload": overload if thermal_status == "Evaluated" else None,
        "thermal_status": thermal_status,
        "message": str(getattr(result, "message", "")),
    }


def run_forecasted_power_flows(
    bundle: LoadForecastBundle,
    project: PowerSystemProject,
    horizon_hours: int = 24,
    solver_method: str = "Forward/Backward Sweep",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    vmin_limit: float = 0.95,
    vmax_limit: float = 1.05,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    allow_opposite_q_sign: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast load and run one power flow for each forecasted hour."""
    forecast_df = forecast_total_load(bundle, horizon_hours=horizon_hours)
    ref_buses = [b for b in project.buses if str(b.bus_type).lower() != "slack"]
    base_p = sum(float(b.p_load_pu) for b in ref_buses)
    base_q = sum(float(b.q_load_pu) for b in ref_buses)
    if abs(base_p) < 1e-12:
        raise ValueError("Cannot run forecasted PF with zero reference P demand.")
    # No historical Q: preserve the reference network's signed aggregate Q/P.
    if "total_q_mvar" not in bundle.history.columns or not bundle.history["total_q_mvar"].notna().any():
        forecast_df["forecast_total_q_mvar"] = forecast_df["forecast_total_p_mw"] * base_q / base_p
        forecast_df["Q_forecast_method"] = "project_baseline_q_over_p"
    else:
        forecast_df["Q_forecast_method"] = "trained_or_historical_ratio"
        q_ratio = float(np.nanmedian(pd.to_numeric(forecast_df["forecast_total_q_mvar"], errors="coerce") /
                                     pd.to_numeric(forecast_df["forecast_total_p_mw"], errors="coerce")))
        if (abs(base_q) > 1e-9 and abs(q_ratio) > 1e-9 and base_q * q_ratio < 0 and not allow_opposite_q_sign):
            raise ValueError(
                "Historical reactive-power sign differs from the project's reference Q sign. "
                "For the bundled IEEE 33-bus case, negative Q denotes inductive demand. "
                "Check metering conventions or explicitly approve capacitive operation."
            )
    risk_rows: List[Dict[str, Any]] = []
    n = len(forecast_df)
    for i, row in forecast_df.iterrows():
        if progress_callback is not None:
            progress_callback(int(i + 1), n)
        scenario_project = apply_total_load_forecast_to_project(
            project,
            forecast_total_p_mw=float(row["forecast_total_p_mw"]),
            forecast_total_q_mvar=float(row["forecast_total_q_mvar"]),
        )
        result = run_power_flow_project(
            scenario_project,
            max_iterations=max_iterations,
            tolerance=tolerance,
            suppress_output=True,
            solver_method=solver_method,
        )
        risk_entry = _extract_power_flow_risk(
            result,
            timestamp=row["timestamp"],
            p_mw=float(row["forecast_total_p_mw"]),
            q_mvar=float(row["forecast_total_q_mvar"]),
            vmin_limit=vmin_limit,
            vmax_limit=vmax_limit,
        )
        risk_entry["Q_forecast_method"] = row["Q_forecast_method"]
        risk_rows.append(risk_entry)
    return forecast_df, pd.DataFrame(risk_rows)


def summarize_forecast_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact summary KPIs for forecasted power-flow results."""
    if risk_df is None or risk_df.empty:
        return pd.DataFrame({"Metric": ["Status"], "Value": ["No forecast-risk results"]})
    rows = []
    rows.append({"Metric": "Forecast hours", "Value": int(len(risk_df))})
    rows.append({"Metric": "Voltage violation hours", "Value": int(pd.Series(risk_df["voltage_violation"]).astype(bool).sum()) if "voltage_violation" in risk_df else "N/A"})
    thermal_known = "thermal_status" in risk_df.columns and risk_df["thermal_status"].eq("Evaluated").all()
    rows.append({"Metric": "Thermal assessment", "Value": "Evaluated" if thermal_known else "Not evaluated: unverified ratings"})
    rows.append({"Metric": "Overload hours", "Value": int(pd.Series(risk_df["overload"]).fillna(False).astype(bool).sum()) if thermal_known else "Not evaluated"})
    if "vmin_pu" in risk_df.columns:
        v = pd.to_numeric(risk_df["vmin_pu"], errors="coerce")
        if v.notna().any():
            idx = v.idxmin()
            rows.append({"Metric": "Worst minimum voltage", "Value": f"{float(v.loc[idx]):.5f} pu at {risk_df.loc[idx, 'timestamp']}"})
            rows.append({"Metric": "Worst-voltage bus", "Value": str(risk_df.loc[idx, "vmin_bus"])})
    if "total_p_loss_mw" in risk_df.columns:
        loss = pd.to_numeric(risk_df["total_p_loss_mw"], errors="coerce")
        if loss.notna().any():
            idx = loss.idxmax()
            rows.append({"Metric": "Maximum real loss", "Value": f"{float(loss.loc[idx]):.5f} MW at {risk_df.loc[idx, 'timestamp']}"})
    return pd.DataFrame(rows)
