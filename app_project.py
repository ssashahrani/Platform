"""Python-Based Power System Simulation and Analysis Platform.

Launch: python app_project.py

The earlier UI sources v10/v15/v16/v20 are fully consolidated in this file.
The following non-UI modules must still be present in the project:
    project_model.py, adapter_project_fbs.py, ai_advisor.py,
    ml_engine.py, load_forecasting.py
Their numerical and ML implementations are not duplicated here.

UI dependencies: customtkinter, pandas, matplotlib, numpy, openpyxl,
scikit-learn, and those required by the engineering backend.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional
from pathlib import Path
import hashlib
import json
from PIL import Image

try:
    import customtkinter as ctk
except ModuleNotFoundError as exc:
    raise SystemExit("CustomTkinter is not installed. Run: pip install customtkinter") from exc

try:
    import pandas as pd
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ModuleNotFoundError as exc:
    raise SystemExit("Missing UI dependency. Run: pip install pandas matplotlib numpy") from exc

from project_model import (
    BranchRecord, BusRecord, PowerSystemProject, ProjectSettings,
    SequenceSourceRecord, default_7_bus_project,
)
from adapter_project_fbs import (
    available_solver_methods, run_fault_study_project,
    run_power_flow_project, validate_project_generic,
)
from ai_advisor import AdvisorThresholds, AIInsightReport, generate_ai_insights as build_ai_insights
from ml_engine import (
    MLModelBundle, compare_prediction_with_solver, generate_training_dataset,
    load_model_bundle, predict_current_case, save_model_bundle,
    train_gradient_boosting_models,
)
from load_forecasting import (
    LoadForecastBundle, forecast_total_load, load_history_csv,
    run_forecasted_power_flows, summarize_forecast_risk,
    train_total_load_forecaster,
)

APP_TITLE = "Power System Simulation and Analysis Platform"
WINDOW_SIZE = "1720x980"

class DataFrameTable(ctk.CTkFrame):
    """Display a pandas DataFrame using ttk.Treeview, with severity-based row colors."""

    SEVERITY_TAG_STYLES = {
        "ok": {"background": "#d8f3dc", "foreground": "#0b3d1e"},
        "pass": {"background": "#d8f3dc", "foreground": "#0b3d1e"},
        "normal": {"background": "#d8f3dc", "foreground": "#0b3d1e"},
        "info": {"background": "#dbeafe", "foreground": "#1e3a8a"},
        "warning": {"background": "#fff3cd", "foreground": "#664d03"},
        "error": {"background": "#f8d7da", "foreground": "#842029"},
        "critical": {"background": "#f8d7da", "foreground": "#842029"},
        "fail": {"background": "#f8d7da", "foreground": "#842029"},
        "high": {"background": "#f8d7da", "foreground": "#842029"},
        "medium": {"background": "#fff3cd", "foreground": "#664d03"},
        "low": {"background": "#dbeafe", "foreground": "#1e3a8a"},
    }

    def __init__(self, master: Any, title: str = "", **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        if title:
            ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=15, weight="bold"), anchor="w").grid(
                row=0, column=0, sticky="ew", padx=10, pady=(8, 4)
            )

        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self._configure_color_tags()
        self.set_dataframe(pd.DataFrame())

    def _configure_color_tags(self) -> None:
        for tag, style in self.SEVERITY_TAG_STYLES.items():
            self.tree.tag_configure(tag, **style)

    @staticmethod
    def _fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, complex):
            sign = "+" if value.imag >= 0 else "-"
            return f"{value.real:.6g} {sign} j{abs(value.imag):.6g}"
        return str(value)

    @staticmethod
    def _row_tag(row: pd.Series) -> str:
        """Return a Treeview tag based on Severity/Priority/Status fields."""
        for key in ("Severity", "Priority", "Status", "Result"):
            if key in row.index:
                value = str(row.get(key, "")).strip().lower()
                if value in DataFrameTable.SEVERITY_TAG_STYLES:
                    return value
                if "critical" in value:
                    return "critical"
                if "error" in value or "fail" in value or "overload" in value:
                    return "error"
                if "warning" in value or "warn" in value or "medium" in value:
                    return "warning"
                if "ok" in value or "pass" in value or "success" in value:
                    return "ok"
                if "info" in value:
                    return "info"
        return ""

    def set_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        df = pd.DataFrame() if df is None else df.copy()
        self.tree.delete(*self.tree.get_children())
        if df.empty:
            self.tree["columns"] = ["Status"]
            self.tree.heading("Status", text="Status")
            self.tree.column("Status", width=400, anchor="center")
            self.tree.insert("", "end", values=("No data to display",), tags=("info",))
            return

        columns = [str(c) for c in df.columns]
        self.tree["columns"] = columns
        # Size columns using displayed cell values, not only header length.
        # Priority-action descriptions were previously clipped at 230 px.
        sample = df.head(250)
        for col in columns:
            self.tree.heading(col, text=col)
            max_chars = max(
                len(col),
                *(len(self._fmt(value)) for value in sample[col].tolist()),
            )
            width = max(105, min(700, max_chars * 8 + 32))
            self.tree.column(col, width=width, minwidth=90,
                             anchor="w" if max_chars > 20 else "center",
                             stretch=False)
        for _, row in df.iterrows():
            tag = self._row_tag(row)
            tags = (tag,) if tag else ()
            self.tree.insert("", "end", values=[self._fmt(row[col]) for col in df.columns], tags=tags)


class AIInsightsPanel(ctk.CTkFrame):
    """AI Advisor tab with editable thresholds, report, findings, actions, and color-coded status."""

    def __init__(self, master: Any, generate_callback: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.generate_callback = generate_callback
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        controls = ctk.CTkFrame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for c in range(7):
            controls.grid_columnconfigure(c, weight=1)

        self.threshold_entries: Dict[str, ctk.CTkEntry] = {}
        fields = [
            ("Voltage min pu", "voltage_min_pu", "0.95"),
            ("Voltage max pu", "voltage_max_pu", "1.05"),
            ("Critical min pu", "voltage_critical_min_pu", "0.90"),
            ("High loading %", "high_loading_percent", "80"),
            ("Overload %", "overload_percent", "100"),
            ("Top loss branches", "top_loss_branches", "5"),
        ]
        for col, (label, key, default) in enumerate(fields):
            ctk.CTkLabel(controls, text=label, anchor="w").grid(row=0, column=col, sticky="ew", padx=4, pady=(4, 0))
            entry = ctk.CTkEntry(controls, width=110)
            entry.grid(row=1, column=col, sticky="ew", padx=4, pady=(0, 4))
            entry.insert(0, default)
            self.threshold_entries[key] = entry

        ctk.CTkButton(controls, text="Generate AI Insights", command=self.generate_callback).grid(
            row=1, column=6, sticky="ew", padx=8, pady=(0, 4)
        )

        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_columnconfigure(1, weight=3)
        self.health_badge = ctk.CTkLabel(
            status_frame,
            text="AI status: Not run",
            fg_color="#6c757d",
            text_color="white",
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
            padx=14,
            pady=8,
        )
        self.health_badge.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)
        self.health_message = ctk.CTkLabel(
            status_frame,
            text="Green = pass/OK, yellow = warning, red = critical/error.",
            anchor="w",
            font=ctk.CTkFont(size=13),
        )
        self.health_message.grid(row=0, column=1, sticky="ew", padx=(6, 8), pady=8)

        self.ai_tabs = ctk.CTkTabview(self)
        self.ai_tabs.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        for name in ["Report", "Findings", "Recommended Actions"]:
            self.ai_tabs.add(name)
            self.ai_tabs.tab(name).grid_columnconfigure(0, weight=1)
            self.ai_tabs.tab(name).grid_rowconfigure(0, weight=1)

        self.report_box = ctk.CTkTextbox(
            self.ai_tabs.tab("Report"),
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=13),
            corner_radius=12,
            border_width=1,
        )
        self.report_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.findings_table = DataFrameTable(self.ai_tabs.tab("Findings"), "AI Findings")
        self.findings_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.actions_table = DataFrameTable(self.ai_tabs.tab("Recommended Actions"), "AI Recommended Actions")
        self.actions_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.clear()

    def _set_health_badge(self, level: str, message: str) -> None:
        level_norm = level.strip().lower()
        styles = {
            "pass": ("#198754", "AI status: PASS"),
            "ok": ("#198754", "AI status: PASS"),
            "warning": ("#ffc107", "AI status: WARNING"),
            "critical": ("#dc3545", "AI status: CRITICAL"),
            "error": ("#dc3545", "AI status: ERROR"),
            "info": ("#0d6efd", "AI status: INFO"),
            "not_run": ("#6c757d", "AI status: Not run"),
        }
        color, label = styles.get(level_norm, styles["info"])
        text_color = "black" if level_norm == "warning" else "white"
        self.health_badge.configure(text=label, fg_color=color, text_color=text_color)
        self.health_message.configure(text=message)

    @staticmethod
    def _overall_status(findings: pd.DataFrame) -> tuple[str, str]:
        if findings is None or findings.empty or "Severity" not in findings.columns:
            return "info", "AI findings are not available."
        severities = findings["Severity"].astype(str).str.lower().tolist()
        critical_count = sum(1 for s in severities if "critical" in s or "error" in s or "fail" in s)
        warning_count = sum(1 for s in severities if "warning" in s or "warn" in s)
        ok_count = sum(1 for s in severities if s in {"ok", "pass", "normal"})
        if critical_count:
            return "critical", f"{critical_count} critical/error finding(s) and {warning_count} warning(s). Check red rows first."
        if warning_count:
            return "warning", f"{warning_count} warning finding(s). Review yellow rows and recommended actions."
        return "pass", f"No warnings or critical findings detected. {ok_count} pass/OK finding(s)."

    def get_thresholds(self) -> AdvisorThresholds:
        def f(key: str) -> float:
            return float(self.threshold_entries[key].get().strip())
        thresholds = AdvisorThresholds(
            voltage_min_pu=f("voltage_min_pu"),
            voltage_max_pu=f("voltage_max_pu"),
            voltage_critical_min_pu=f("voltage_critical_min_pu"),
            high_loading_percent=f("high_loading_percent"),
            overload_percent=f("overload_percent"),
            top_loss_branches=int(float(self.threshold_entries["top_loss_branches"].get().strip())),
        )
        thresholds.validate()
        return thresholds

    def clear(self) -> None:
        self._set_health_badge("not_run", "Green = pass/OK, yellow = warning, red = critical/error.")
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert(
            "1.0",
            "Run power flow, then click Generate AI Insights.\n\n"
            "The AI Advisor uses rule-based engineering thresholds. It does not replace the power-flow solver.",
        )
        self.report_box.configure(state="disabled")
        self.findings_table.set_dataframe(pd.DataFrame())
        self.actions_table.set_dataframe(pd.DataFrame())

    def set_error(self, message: str) -> None:
        self._set_health_badge("error", "AI Advisor failed. Review the message below.")
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", f"AI Advisor could not generate insights:\n\n{message}")
        self.report_box.configure(state="disabled")
        self.findings_table.set_dataframe(pd.DataFrame())
        self.actions_table.set_dataframe(pd.DataFrame())

    def set_report(self, report: AIInsightReport) -> None:
        level, message = self._overall_status(report.findings_table)
        self._set_health_badge(level, message)
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", report.report_text)
        self.report_box.configure(state="disabled")
        self.findings_table.set_dataframe(report.findings_table)
        self.actions_table.set_dataframe(report.actions_table)
        self.ai_tabs.set("Report")


class MachineLearningPanel(ctk.CTkFrame):
    """Machine-learning tab for gradient-boosted-tree surrogate studies."""

    def __init__(
        self,
        master: Any,
        generate_dataset_callback: Any,
        train_model_callback: Any,
        predict_callback: Any,
        compare_callback: Any,
        save_model_callback: Any,
        load_model_callback: Any,
        export_dataset_callback: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.generate_dataset_callback = generate_dataset_callback
        self.train_model_callback = train_model_callback
        self.predict_callback = predict_callback
        self.compare_callback = compare_callback
        self.save_model_callback = save_model_callback
        self.load_model_callback = load_model_callback
        self.export_dataset_callback = export_dataset_callback

        self.grid_columnconfigure(0, weight=1)
        # Rows 0-3 size to their contents; only the result tabs absorb
        # the remaining height. The status label must never be stretched
        # or clipped by the notebook's vertical allocation.
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=1)

        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Machine Learning: Gradient-Boosted Decision Trees",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        ctk.CTkLabel(
            header,
            text="Topology-specific screening model trained from your own FBS/NR simulation results. The deterministic solver remains the source of truth.",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        config = ctk.CTkFrame(self)
        config.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        for c in range(8):
            config.grid_columnconfigure(c, weight=1)

        self.config_entries: Dict[str, ctk.CTkEntry] = {}
        fields = [
            ("Scenarios", "n_scenarios", "500"),
            ("Load min", "global_load_min", "0.5"),
            ("Load max", "global_load_max", "1.5"),
            ("Local variation %", "local_variation_percent", "20"),
            ("V min pu", "voltage_min_pu", "0.95"),
            ("V max pu", "voltage_max_pu", "1.05"),
            ("Seed", "random_seed", "42"),
            ("Test size", "test_size", "0.2"),
        ]
        for col, (label, key, default) in enumerate(fields):
            ctk.CTkLabel(config, text=label, anchor="w").grid(row=0, column=col, sticky="ew", padx=4, pady=(4, 0))
            entry = ctk.CTkEntry(config, width=110)
            entry.grid(row=1, column=col, sticky="ew", padx=4, pady=(0, 6))
            entry.insert(0, default)
            self.config_entries[key] = entry

        buttons = ctk.CTkFrame(self)
        buttons.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        for c in range(7):
            buttons.grid_columnconfigure(c, weight=1)
        ctk.CTkButton(buttons, text="Generate Dataset", command=self.generate_dataset_callback).grid(row=0, column=0, sticky="ew", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Train Model", command=self.train_model_callback).grid(row=0, column=1, sticky="ew", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Predict Current Case", command=self.predict_callback).grid(row=0, column=2, sticky="ew", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Compare with Solver", command=self.compare_callback).grid(row=0, column=3, sticky="ew", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Save Model", fg_color="#596b7a", command=self.save_model_callback).grid(row=0, column=4, sticky="ew", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Load Model", fg_color="#596b7a", command=self.load_model_callback).grid(row=0, column=5, sticky="ew", padx=4, pady=6)
        ctk.CTkButton(buttons, text="Export Dataset CSV", fg_color="#2e7d32", command=self.export_dataset_callback).grid(row=0, column=6, sticky="ew", padx=4, pady=6)

        status_row = ctk.CTkFrame(self)
        status_row.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        status_row.grid_columnconfigure(1, weight=1)
        self.ml_badge = ctk.CTkLabel(
            status_row,
            text="ML status: Not trained",
            fg_color="#6c757d",
            text_color="white",
            corner_radius=14,
            font=ctk.CTkFont(size=14, weight="bold"),
            padx=14,
            pady=8,
        )
        self.ml_badge.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)
        self.ml_message = ctk.CTkLabel(
            status_row,
            text="Generate a dataset, train the model, then predict the current case.",
            anchor="w", justify="left", wraplength=730,
        )
        self.ml_message.grid(row=0, column=1, sticky="ew", padx=(6, 8), pady=8)

        self.ml_tabs = ctk.CTkTabview(self)
        self.ml_tabs.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.grid_rowconfigure(4, weight=1)
        for name in ["Dataset Preview", "Model Metrics", "Prediction", "Comparison", "Log"]:
            self.ml_tabs.add(name)
            self.ml_tabs.tab(name).grid_columnconfigure(0, weight=1)
            self.ml_tabs.tab(name).grid_rowconfigure(0, weight=1)

        self.dataset_table = DataFrameTable(self.ml_tabs.tab("Dataset Preview"), "ML Dataset Preview")
        self.dataset_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.metrics_table = DataFrameTable(self.ml_tabs.tab("Model Metrics"), "Model Accuracy Metrics")
        self.metrics_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.prediction_table = DataFrameTable(self.ml_tabs.tab("Prediction"), "Current-Case ML Prediction")
        self.prediction_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.comparison_table = DataFrameTable(self.ml_tabs.tab("Comparison"), "ML Prediction vs Deterministic Solver")
        self.comparison_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.log_box = ctk.CTkTextbox(self.ml_tabs.tab("Log"), wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.clear()

    def get_config(self) -> Dict[str, Any]:
        def f(key: str) -> float:
            return float(self.config_entries[key].get().strip())
        n_scenarios = int(float(self.config_entries["n_scenarios"].get().strip()))
        if n_scenarios < 30:
            raise ValueError("Use at least 30 scenarios. For IEEE 33-bus, 500 to 2000 scenarios is better.")
        cfg = {
            "n_scenarios": n_scenarios,
            "global_load_min": f("global_load_min"),
            "global_load_max": f("global_load_max"),
            "local_variation_percent": f("local_variation_percent"),
            "voltage_min_pu": f("voltage_min_pu"),
            "voltage_max_pu": f("voltage_max_pu"),
            "random_seed": int(float(self.config_entries["random_seed"].get().strip())),
            "test_size": f("test_size"),
        }
        if cfg["global_load_min"] <= 0 or cfg["global_load_max"] <= cfg["global_load_min"]:
            raise ValueError("Load range must satisfy 0 < Load min < Load max.")
        if cfg["voltage_min_pu"] <= 0 or cfg["voltage_max_pu"] <= cfg["voltage_min_pu"]:
            raise ValueError("Voltage range must satisfy 0 < V min < V max.")
        if not (0.05 <= cfg["test_size"] <= 0.5):
            raise ValueError("Test size should be between 0.05 and 0.5.")
        return cfg

    def set_status(self, level: str, message: str) -> None:
        styles = {
            "ok": ("#198754", "ML status: READY"),
            "training": ("#0d6efd", "ML status: RUNNING"),
            "warning": ("#ffc107", "ML status: WARNING"),
            "error": ("#dc3545", "ML status: ERROR"),
            "idle": ("#6c757d", "ML status: Not trained"),
        }
        color, label = styles.get(level, styles["idle"])
        text_color = "black" if level == "warning" else "white"
        self.ml_badge.configure(text=label, fg_color=color, text_color=text_color)
        self.ml_message.configure(text=message)

    def clear(self) -> None:
        self.set_status("idle", "Generate a dataset, train the model, then predict the current case.")
        self.dataset_table.set_dataframe(pd.DataFrame())
        self.metrics_table.set_dataframe(pd.DataFrame())
        self.prediction_table.set_dataframe(pd.DataFrame())
        self.comparison_table.set_dataframe(pd.DataFrame())
        self.set_log(
            "Machine Learning workflow:\n"
            "1. Load or define a fixed-topology project, such as IEEE 33-bus.\n"
            "2. Generate randomized operating scenarios.\n"
            "3. Train gradient-boosted decision-tree models.\n"
            "4. Predict the current case.\n"
            "5. Verify important decisions using the deterministic solver.\n"
        )

    def set_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", text)
        self.log_box.configure(state="disabled")

    def append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_dataset(self, dataset: pd.DataFrame, summary: Dict[str, Any]) -> None:
        preview_cols = [
            c for c in [
                "scenario_id", "global_load_multiplier", "target_vmin_pu", "target_vmin_bus",
                "target_total_p_loss_mw", "target_voltage_violation", "target_overload", "iterations",
            ] if c in dataset.columns
        ]
        preview = dataset[preview_cols].head(100) if preview_cols else dataset.head(100)
        self.dataset_table.set_dataframe(preview)
        self.set_status(
            "ok",
            f"Dataset ready: {summary.get('successful_scenarios', len(dataset))} successful scenarios; "
            f"{summary.get('failed_scenarios', 0)} failed/skipped.",
        )
        self.append_log(
            f"Dataset generated: {summary.get('successful_scenarios', len(dataset))} rows, "
            f"{len(dataset.columns)} columns, solver={summary.get('solver_method')}."
        )
        failures = summary.get("failure_examples", []) or []
        if failures:
            self.append_log("Failure examples:")
            for item in failures[:5]:
                self.append_log(f"- {item}")
        self.ml_tabs.set("Dataset Preview")

    def set_metrics(self, metrics: pd.DataFrame, summary: Dict[str, Any]) -> None:
        self.metrics_table.set_dataframe(metrics)
        rows = summary.get("training_rows", "N/A")
        features = summary.get("feature_count", "N/A")
        self.set_status("ok", f"Model trained: {rows} rows, {features} features. Review Model Metrics.")
        self.append_log(f"Model trained: rows={rows}, features={features}, family={summary.get('model_family')}")
        self.ml_tabs.set("Model Metrics")

    def set_prediction(self, prediction: pd.DataFrame) -> None:
        self.prediction_table.set_dataframe(prediction)
        self.set_status("ok", "Current-case ML prediction generated. Run/compare with solver before making engineering decisions.")
        self.ml_tabs.set("Prediction")

    def set_comparison(self, comparison: pd.DataFrame) -> None:
        self.comparison_table.set_dataframe(comparison)
        self.ml_tabs.set("Comparison")

    def set_error(self, message: str) -> None:
        self.set_status("error", message)
        self.append_log("ERROR: " + message)


class EditableTable(ctk.CTkFrame):
    """
    Simple table editor: form fields + Add/Update + Delete.

    This avoids making users edit Python code while keeping the UI simple.
    """

    def __init__(self, master: Any, title: str, columns: List[str], defaults: Dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.columns = columns
        self.defaults = defaults or {}
        self.entries: Dict[str, ctk.CTkEntry] = {}
        self.selected_item: Optional[str] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=16, weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=10, pady=(8, 4)
        )

        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=max(95, min(180, len(col) * 10 + 45)), anchor="center", stretch=False)

        form = ctk.CTkFrame(self)
        form.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        max_cols = 4
        for idx, col in enumerate(columns):
            r = (idx // max_cols) * 2
            c = idx % max_cols
            form.grid_columnconfigure(c, weight=1)
            ctk.CTkLabel(form, text=col, anchor="w").grid(row=r, column=c, sticky="ew", padx=6, pady=(6, 0))
            entry = ctk.CTkEntry(form)
            entry.grid(row=r + 1, column=c, sticky="ew", padx=6, pady=(0, 6))
            entry.insert(0, str(self.defaults.get(col, "")))
            self.entries[col] = entry

        buttons = ctk.CTkFrame(self)
        buttons.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        buttons.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)
        ctk.CTkButton(buttons, text="Add Row", command=self.add_row).grid(row=0, column=0, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Update Selected", command=self.update_selected).grid(row=0, column=1, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Duplicate", fg_color="#596b7a", command=self.duplicate_selected).grid(row=0, column=2, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Move Up", fg_color="#596b7a", command=lambda: self.move_selected(-1)).grid(row=0, column=3, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Move Down", fg_color="#596b7a", command=lambda: self.move_selected(1)).grid(row=0, column=4, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Delete Selected", fg_color="#6b6b6b", command=self.delete_selected).grid(row=0, column=5, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Clear Form", fg_color="#6b6b6b", command=self.clear_form).grid(row=0, column=6, sticky="ew", padx=4, pady=5)
        ctk.CTkButton(buttons, text="Clear Table", fg_color="#8a3d3d", command=self.confirm_clear_rows).grid(row=0, column=7, sticky="ew", padx=4, pady=5)

    def _on_select(self, _event: Any = None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        self.selected_item = selected[0]
        values = self.tree.item(self.selected_item, "values")
        for col, value in zip(self.columns, values):
            self.entries[col].delete(0, "end")
            self.entries[col].insert(0, str(value))

    def _form_values(self) -> List[str]:
        return [self.entries[col].get().strip() for col in self.columns]

    def add_row(self, values: Optional[List[Any]] = None) -> None:
        row_values = [str(v) for v in (values if values is not None else self._form_values())]
        if len(row_values) != len(self.columns):
            messagebox.showerror("Invalid row", "Wrong number of row values.")
            return
        self.tree.insert("", "end", values=row_values)

    def update_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No row selected", "Select a row to update.")
            return
        self.tree.item(selected[0], values=self._form_values())

    def duplicate_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No row selected", "Select a row to duplicate.")
            return
        values = list(self.tree.item(selected[0], "values"))
        new_item = self.tree.insert("", "end", values=values)
        self.tree.selection_set(new_item)
        self.tree.focus(new_item)

    def move_selected(self, direction: int) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No row selected", "Select a row to move.")
            return
        item = selected[0]
        children = list(self.tree.get_children())
        index = children.index(item)
        new_index = max(0, min(len(children) - 1, index + direction))
        if new_index == index:
            return
        self.tree.move(item, "", new_index)
        self.tree.selection_set(item)
        self.tree.focus(item)

    def delete_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No row selected", "Select a row to delete.")
            return
        for item in selected:
            self.tree.delete(item)
        self.selected_item = None

    def clear_form(self) -> None:
        for col, entry in self.entries.items():
            entry.delete(0, "end")
            entry.insert(0, str(self.defaults.get(col, "")))

    def clear_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.selected_item = None

    def confirm_clear_rows(self) -> None:
        if messagebox.askyesno("Clear table", "Delete all rows in this table?"):
            self.clear_rows()

    def get_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            rows.append({col: str(value) for col, value in zip(self.columns, values)})
        return rows

    def set_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.clear_rows()
        for row in rows:
            self.add_row([row.get(col, "") for col in self.columns])


class CleanMetricCard(ctk.CTkFrame):
    """Compact KPI card with controlled color states."""

    COLORS = {
        "neutral": ("#ffffff", "#e2e8f0", "#0f172a", "#64748b"),
        "info": ("#eff6ff", "#60a5fa", "#1e3a8a", "#475569"),
        "ok": ("#ecfdf5", "#22c55e", "#14532d", "#475569"),
        "warning": ("#fffbeb", "#f59e0b", "#78350f", "#475569"),
        "critical": ("#fef2f2", "#ef4444", "#7f1d1d", "#475569"),
        "error": ("#fef2f2", "#ef4444", "#7f1d1d", "#475569"),
        "purple": ("#f5f3ff", "#8b5cf6", "#4c1d95", "#475569"),
    }

    def __init__(self, master: Any, title: str, **kwargs: Any) -> None:
        super().__init__(master, corner_radius=18, border_width=1, **kwargs)
        self.title = title
        self.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))

        self.value_label = ctk.CTkLabel(
            self,
            text="—",
            anchor="w",
            font=ctk.CTkFont(size=23, weight="bold"),
        )
        self.value_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 0))

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="Waiting",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11),
            wraplength=180,
        )
        self.subtitle_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.set("—", "Waiting", "neutral")

    def set(self, value: str, subtitle: str, state: str = "neutral") -> None:
        bg, border, value_color, sub_color = self.COLORS.get(state, self.COLORS["neutral"])
        self.configure(fg_color=bg, border_color=border)
        self.title_label.configure(text_color=sub_color)
        self.value_label.configure(text=value, text_color=value_color)
        self.subtitle_label.configure(text=subtitle, text_color=sub_color)


class DashboardVoltagePlot(ctk.CTkFrame):
    """Voltage profile card for the Dashboard landing page."""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, corner_radius=18, border_width=1, fg_color="#ffffff", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Voltage Profile",
            font=ctk.CTkFont(size=17, weight="bold"),
            anchor="w",
            text_color="#0f172a",
        ).grid(row=0, column=0, sticky="ew")
        self.badge = ctk.CTkLabel(
            header,
            text="Not run",
            fg_color="#64748b",
            text_color="white",
            corner_radius=12,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=10,
            pady=5,
        )
        self.badge.grid(row=0, column=1, sticky="e")

        self.figure = Figure(figsize=(7.0, 4.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.clear()

    def clear(self) -> None:
        self.badge.configure(text="Not run", fg_color="#64748b")
        self.ax.clear()
        self.ax.set_title("Run power flow to display voltage profile")
        self.ax.set_xlabel("Bus")
        self.ax.set_ylabel("Voltage (pu)")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def set_voltage_profile(self, bus_table: pd.DataFrame, vmin_limit: float = 0.95, vmax_limit: float = 1.05) -> None:
        if bus_table is None or bus_table.empty:
            self.clear()
            return

        b_col = self._column(bus_table, ["Bus", "bus_id", "Bus_ID"])
        v_col = self._column(bus_table, ["Voltage_pu", "Voltage_mag_pu", "V_pu", "Voltage"])
        if b_col is None or v_col is None:
            self.clear()
            self.ax.set_title("Bus or voltage column not found")
            self.canvas.draw_idle()
            return

        df = bus_table[[b_col, v_col]].copy()
        df["__bus__"] = pd.to_numeric(df[b_col], errors="coerce")
        df["__v__"] = pd.to_numeric(df[v_col], errors="coerce")
        df = df.dropna(subset=["__bus__", "__v__"]).sort_values("__bus__")
        if df.empty:
            self.clear()
            return

        x = df["__bus__"].to_list()
        y = df["__v__"].to_list()
        vmin = min(y)
        state = "PASS" if vmin >= vmin_limit else ("WARNING" if vmin >= 0.90 else "CRITICAL")
        color = "#15803d" if state == "PASS" else ("#d97706" if state == "WARNING" else "#dc2626")
        self.badge.configure(text=state, fg_color=color)

        self.ax.clear()
        self.ax.plot(x, y, marker="o", linewidth=2)
        self.ax.axhline(vmin_limit, linestyle="--", linewidth=1)
        self.ax.axhline(vmax_limit, linestyle="--", linewidth=1)
        self.ax.set_title("Bus Voltage Magnitude")
        self.ax.set_xlabel("Bus")
        self.ax.set_ylabel("Voltage (pu)")
        self.ax.grid(True, alpha=0.3)

        # Avoid overcrowded x labels for large feeders.
        if len(x) <= 35:
            self.ax.set_xticks(x)
        else:
            step = max(1, len(x) // 15)
            self.ax.set_xticks(x[::step])

        ymin = min(min(y), vmin_limit) - 0.015
        ymax = max(max(y), vmax_limit) + 0.015
        self.ax.set_ylim(ymin, ymax)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_plot(self, bus_table: pd.DataFrame) -> None:
        """
        Compatibility alias.

        ProjectEditorApp.run_power_flow() expects the voltage plot widget to expose
        update_plot(bus_table). The polished dashboard widget originally used
        update(bus_table), which caused:
            'DashboardVoltagePlot' object has no attribute 'update_plot'
        """
        self.set_voltage_profile(bus_table)


class DashboardMetricCard(CleanMetricCard):
    """Responsive KPI: keep complete values visible, including long project/solver names.

    Unlike a single-line CTkLabel, value text wraps within a bounded card.
    Resize-driven wrap lengths prevent large text from expanding neighboring columns.
    """

    def __init__(self, master: Any, title: str, **kwargs: Any) -> None:
        super().__init__(master, title, **kwargs)
        self.configure(height=132)
        self.grid_propagate(False)
        # Fixed available height and a bounded wrap width keep widgets from
        # determining the entire dashboard's minimum horizontal size.
        self.value_label.configure(
            width=1, height=62,
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w", justify="left", wraplength=260,
        )
        self.subtitle_label.configure(
            width=1, height=22, anchor="w", justify="left", wraplength=260,
        )
        self.bind("<Configure>", self._resize_text, add="+")

    def _resize_text(self, event: Any) -> None:
        width = max(90, int(event.width) - 30)
        # Adapt text size at narrow widths, without cutting off names.
        size = 20 if width >= 380 else (18 if width >= 250 else 15)
        self.value_label.configure(
            wraplength=width, font=ctk.CTkFont(size=size, weight="bold")
        )
        self.subtitle_label.configure(wraplength=width)


class DashboardHome(ctk.CTkFrame):
    """Engineering dashboard with readable KPIs and full-height result sections.

    The original single-row KPI strip truncated long solver/project names, and
    stacking multiple Treeviews caused Priority Actions to collapse. The new
    layout assigns two KPI rows and one full-size content area with four views.
    """

    def __init__(self, master: Any, app: Any, **kwargs: Any) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        kpis = ctk.CTkFrame(self, fg_color="transparent")
        kpis.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 7))
        for col in range(12):
            kpis.grid_columnconfigure(col, weight=1, uniform="dashboard_kpis")

        # Wider first row: full project and solver identifiers can wrap naturally.
        self.project_card = DashboardMetricCard(kpis, "Project")
        self.project_card.grid(row=0, column=0, columnspan=6, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self.solver_card = DashboardMetricCard(kpis, "Solver")
        self.solver_card.grid(row=0, column=6, columnspan=6, sticky="nsew", padx=(6, 0), pady=(0, 6))

        self.status_card = DashboardMetricCard(kpis, "Power Flow")
        self.status_card.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=(0, 6))
        self.vmin_card = DashboardMetricCard(kpis, "Minimum Voltage")
        self.vmin_card.grid(row=1, column=3, columnspan=3, sticky="nsew", padx=6)
        self.loss_card = DashboardMetricCard(kpis, "Real Loss")
        self.loss_card.grid(row=1, column=6, columnspan=3, sticky="nsew", padx=6)
        self.ml_card = DashboardMetricCard(kpis, "ML Model")
        self.ml_card.grid(row=1, column=9, columnspan=3, sticky="nsew", padx=(6, 0))

        # One content view at a time avoids permanently undersized Treeviews.
        # Keep the original public attributes for the existing backend callbacks.
        self.sections = ctk.CTkTabview(self, corner_radius=14)
        self.sections.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        for name in ("System Health", "Priority Actions", "Voltage Profile", "Study Notes"):
            self.sections.add(name)
            page = self.sections.tab(name)
            page.grid_columnconfigure(0, weight=1)
            page.grid_rowconfigure(0, weight=1)

        self.health_panel = DataFrameTable(self.sections.tab("System Health"), "System Health")
        self.health_panel.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.action_panel = DataFrameTable(self.sections.tab("Priority Actions"), "Priority Actions")
        self.action_panel.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.voltage_chart = DashboardVoltagePlot(self.sections.tab("Voltage Profile"))
        self.voltage_chart.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.notes = ctk.CTkTextbox(
            self.sections.tab("Study Notes"), wrap="word",
            font=ctk.CTkFont(family="Consolas", size=13),
        )
        self.notes.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sections.set("System Health")
        self.clear()

    def clear(self) -> None:
        self.project_card.set("No project", "Load or define a case", "info")
        self.solver_card.set("Auto", "Solver selection", "info")
        self.status_card.set("Not run", "Run a study", "neutral")
        self.vmin_card.set("—", "No voltage result", "neutral")
        self.loss_card.set("—", "No loss result", "neutral")
        self.ml_card.set("Not trained", "Use ML tab", "purple")
        self.voltage_chart.clear()
        self.health_panel.set_dataframe(pd.DataFrame({"Status": ["Run power flow to populate system health."]}))
        self.action_panel.set_dataframe(pd.DataFrame({"Status": ["No priority actions yet."]}))
        self.set_notes(
            "Dashboard workflow:\n"
            "1. Load or define a project.\n"
            "2. Validate project data.\n"
            "3. Run power flow.\n"
            "4. Review voltage, losses, AI findings, and ML screening.\n"
        )

    def set_notes(self, text: str) -> None:
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", text)
        self.notes.configure(state="disabled")


class VisualSummaryPanel(ctk.CTkFrame):
    """Polished visual Summary tab compatible with ProjectEditorApp.run_power_flow()."""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        for c in range(4):
            cards.grid_columnconfigure(c, weight=1)

        self.solver_card = CleanMetricCard(cards, "Solver")
        self.solver_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.iter_card = CleanMetricCard(cards, "Iterations")
        self.iter_card.grid(row=0, column=1, sticky="ew", padx=6)
        self.vmin_card = CleanMetricCard(cards, "Minimum Voltage")
        self.vmin_card.grid(row=0, column=2, sticky="ew", padx=6)
        self.loss_card = CleanMetricCard(cards, "Losses")
        self.loss_card.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self.summary_text = ctk.CTkTextbox(body, wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
        self.summary_text.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.health_table = DataFrameTable(body, "Study Health Indicators")
        self.health_table.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.clear()

    def clear(self) -> None:
        self.solver_card.set("—", "No result", "info")
        self.iter_card.set("—", "No result", "info")
        self.vmin_card.set("—", "No result", "neutral")
        self.loss_card.set("—", "No result", "neutral")
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "Run power flow to display a polished study summary.")
        self.summary_text.configure(state="disabled")
        self.health_table.set_dataframe(pd.DataFrame({"Status": ["No study has been run."]}))

    @staticmethod
    def _col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def update_summary(self, result: Any, project: Any, requested_solver: str = "") -> None:
        summary: Dict[str, Any] = dict(getattr(result, "summary", {}) or {})
        bus_table = getattr(result, "bus_table", pd.DataFrame())
        losses_mva = getattr(result, "losses_table_mva", pd.DataFrame())

        solver = str(summary.get("solver_method", requested_solver or "N/A"))
        converged = bool(getattr(result, "converged", False))
        iterations = summary.get("iterations", "N/A")
        tolerance = summary.get("tolerance", "N/A")
        message = str(getattr(result, "message", "")).strip()

        self.solver_card.set(solver, f"{len(project.buses)} buses, {len(project.branches)} branches", "info")
        self.iter_card.set(str(iterations), f"Tolerance: {tolerance}", "ok" if converged else "error")

        min_voltage = None
        max_voltage = None
        min_bus = "N/A"
        max_bus = "N/A"
        if bus_table is not None and not bus_table.empty:
            v_col = self._col(bus_table, ["Voltage_pu", "Voltage_mag_pu", "V_pu", "Voltage"])
            b_col = self._col(bus_table, ["Bus", "bus_id", "Bus_ID"])
            if v_col is not None:
                tmp = bus_table.copy()
                tmp["__v__"] = pd.to_numeric(tmp[v_col], errors="coerce")
                tmp = tmp.dropna(subset=["__v__"])
                if not tmp.empty:
                    min_row = tmp.loc[tmp["__v__"].idxmin()]
                    max_row = tmp.loc[tmp["__v__"].idxmax()]
                    min_voltage = float(min_row["__v__"])
                    max_voltage = float(max_row["__v__"])
                    min_bus = str(min_row[b_col]) if b_col else "N/A"
                    max_bus = str(max_row[b_col]) if b_col else "N/A"

        if min_voltage is not None:
            state = "ok" if min_voltage >= 0.95 else ("warning" if min_voltage >= 0.90 else "critical")
            self.vmin_card.set(f"{min_voltage:.5f} pu", f"Bus {min_bus}", state)

        p_loss = summary.get("total_loss_MVA_real", None)
        q_loss = summary.get("total_loss_MVA_imag", None)
        if p_loss is None and losses_mva is not None and not losses_mva.empty:
            if "P_loss_MW" in losses_mva.columns:
                p_loss = pd.to_numeric(losses_mva["P_loss_MW"], errors="coerce").sum()
            if "Q_loss_MVAR" in losses_mva.columns:
                q_loss = pd.to_numeric(losses_mva["Q_loss_MVAR"], errors="coerce").sum()

        try:
            p_loss_float = float(p_loss)
        except Exception:
            p_loss_float = None

        if p_loss_float is not None:
            state = "ok" if p_loss_float < 0.5 else ("warning" if p_loss_float < 2 else "critical")
            self.loss_card.set(f"{p_loss_float:.5f} MW", f"Q: {float(q_loss or 0):.5f} MVAr", state)

        rows = [
            {
                "Severity": "OK" if converged else "Error",
                "Indicator": "Convergence",
                "Value": "Yes" if converged else "No",
                "Comment": message or "No solver message",
            },
            {
                "Severity": "OK" if (min_voltage is not None and min_voltage >= 0.95) else "Warning",
                "Indicator": "Minimum voltage",
                "Value": f"{min_voltage:.5f} pu at Bus {min_bus}" if min_voltage is not None else "N/A",
                "Comment": "Within limit" if min_voltage is not None and min_voltage >= 0.95 else "Below typical 0.95 pu limit",
            },
            {
                "Severity": "OK" if (max_voltage is not None and max_voltage <= 1.05) else "Warning",
                "Indicator": "Maximum voltage",
                "Value": f"{max_voltage:.5f} pu at Bus {max_bus}" if max_voltage is not None else "N/A",
                "Comment": "Within limit" if max_voltage is not None and max_voltage <= 1.05 else "Above typical 1.05 pu limit",
            },
            {
                "Severity": "Info",
                "Indicator": "Losses",
                "Value": f"{p_loss_float:.5f} MW" if p_loss_float is not None else "N/A",
                "Comment": "Review Losses tab for branch-level details",
            },
        ]
        self.health_table.set_dataframe(pd.DataFrame(rows))

        lines = [
            "POWER FLOW SUMMARY",
            "",
            f"Project: {project.settings.name}",
            f"Effective solver: {solver}",
            f"Converged: {'Yes' if converged else 'No'}",
            f"Iterations: {iterations}",
            f"Tolerance: {tolerance}",
            f"Network size: {len(project.buses)} buses, {len(project.branches)} branches",
            "",
            f"Minimum voltage: {min_voltage:.5f} pu at Bus {min_bus}" if min_voltage is not None else "Minimum voltage: N/A",
            f"Maximum voltage: {max_voltage:.5f} pu at Bus {max_bus}" if max_voltage is not None else "Maximum voltage: N/A",
            f"Total real loss: {p_loss_float:.5f} MW" if p_loss_float is not None else "Total real loss: N/A",
            f"Total reactive loss: {float(q_loss or 0):.5f} MVAr" if q_loss is not None else "Total reactive loss: N/A",
            "",
            "Recommended review:",
            "- Inspect warning rows first.",
            "- Use Voltage Profile for weak-bus location.",
            "- Use AI Insights for diagnosis and candidate corrective actions.",
            "- Use ML only as a fast screening assistant; verify final results with the solver.",
        ]
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")


class ForecastPlot(ctk.CTkFrame):
    """Compact plot card for the load-forecasting tab."""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, corner_radius=18, border_width=1, fg_color="#ffffff", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 2))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Forecasted Total Feeder Load",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        self.badge = ctk.CTkLabel(
            header,
            text="Waiting",
            fg_color="#64748b",
            text_color="white",
            corner_radius=12,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=10,
            pady=5,
        )
        self.badge.grid(row=0, column=1, sticky="e")

        self.figure = Figure(figsize=(7, 3.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))
        self.clear()

    def clear(self) -> None:
        self.badge.configure(text="Waiting", fg_color="#64748b")
        self.ax.clear()
        self.ax.set_title("Forecast will appear after training the model")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("MW")
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def plot_forecast(self, forecast_df: pd.DataFrame, risk_df: Optional[pd.DataFrame] = None) -> None:
        if forecast_df is None or forecast_df.empty:
            self.clear()
            return

        self.ax.clear()
        x = pd.to_datetime(forecast_df["timestamp"])
        y = pd.to_numeric(forecast_df["forecast_total_p_mw"], errors="coerce")
        self.ax.plot(x, y, marker="o", linewidth=2.2, label="Forecast P")

        badge_text = "FORECAST READY"
        badge_color = "#0891b2"

        if risk_df is not None and not risk_df.empty and "Severity" in risk_df.columns:
            df = risk_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            sev = df["Severity"].astype(str).str.lower()

            ok = sev.eq("ok")
            warn = sev.eq("warning")
            crit = sev.isin(["critical", "error"])

            if ok.any():
                self.ax.scatter(df.loc[ok, "timestamp"], df.loc[ok, "forecast_total_p_mw"], s=28, label="PF OK")
            if warn.any():
                self.ax.scatter(df.loc[warn, "timestamp"], df.loc[warn, "forecast_total_p_mw"], s=45, label="PF warning")
                badge_text, badge_color = "PF WARNING", "#d97706"
            if crit.any():
                self.ax.scatter(df.loc[crit, "timestamp"], df.loc[crit, "forecast_total_p_mw"], s=60, label="PF critical")
                badge_text, badge_color = "PF CRITICAL", "#dc2626"

        self.badge.configure(text=badge_text, fg_color=badge_color)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("MW")
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(axis="x", rotation=25)
        self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()


class LoadForecastingPanel(ctk.CTkFrame):
    """
    Tab-based Load Forecasting workspace.

    Internal workflow tabs:
    Overview | Data | Model Training | Forecast | Forecasted PF Risk | Export / Log
    """

    def __init__(
        self,
        master: Any,
        app: Any,
        load_csv_callback: Any,
        train_callback: Any,
        forecast_callback: Any,
        run_pf_callback: Any,
        export_callback: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self.load_csv_callback = load_csv_callback
        self.train_callback = train_callback
        self.forecast_callback = forecast_callback
        self.run_pf_callback = run_pf_callback
        self.export_callback = export_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self.clear()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=18, border_width=1, fg_color="#ffffff")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Load Forecasting + Forecasted Power-Flow Risk",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))

        ctk.CTkLabel(
            header,
            text="Data → Model Training → Forecast → Forecasted PF Risk → Export / Log",
            font=ctk.CTkFont(size=13),
            text_color="#64748b",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))

        self.status_badge = ctk.CTkLabel(
            header,
            text="NOT READY",
            fg_color="#64748b",
            text_color="white",
            corner_radius=14,
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=12,
            pady=7,
        )
        self.status_badge.grid(row=0, column=1, rowspan=2, sticky="e", padx=16, pady=14)

    def _build_tabs(self) -> None:
        self.workflow_tabs = ctk.CTkTabview(self, corner_radius=18)
        self.workflow_tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        tab_names = ["Overview", "Data", "Model Training", "Forecast", "Forecasted PF Risk", "Export / Log"]
        for name in tab_names:
            self.workflow_tabs.add(name)
            page = self.workflow_tabs.tab(name)
            page.grid_columnconfigure(0, weight=1)
            # Keep the tab header/command controls at their requested height;
            # only the data/results area beneath them may expand or contract.
            page.grid_rowconfigure(0, weight=0)
            page.grid_rowconfigure(1, weight=1)

        self._build_overview_tab()
        self._build_data_tab()
        self._build_model_training_tab()
        self._build_forecast_tab()
        self._build_risk_tab()
        self._build_export_log_tab()

    def _build_overview_tab(self) -> None:
        tab = self.workflow_tabs.tab("Overview")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        cards = ctk.CTkFrame(tab, fg_color="transparent")
        cards.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        for c in range(4):
            cards.grid_columnconfigure(c, weight=1)

        self.data_card = CleanMetricCard(cards, "Historical Data")
        self.data_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.model_card = CleanMetricCard(cards, "Forecast Model")
        self.model_card.grid(row=0, column=1, sticky="ew", padx=6)
        self.forecast_card = CleanMetricCard(cards, "Forecast")
        self.forecast_card.grid(row=0, column=2, sticky="ew", padx=6)
        self.risk_card = CleanMetricCard(cards, "Risk Level")
        self.risk_card.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self.overview_plot = ForecastPlot(body)
        self.overview_plot.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=2)
        right.grid_rowconfigure(1, weight=1)

        self.overview_summary_table = DataFrameTable(right, "Compact Risk Summary")
        self.overview_summary_table.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        note_frame = ctk.CTkFrame(right, corner_radius=18, border_width=1, fg_color="#ffffff")
        note_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        note_frame.grid_columnconfigure(0, weight=1)
        note_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            note_frame,
            text="Workflow Notes",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))

        self.overview_notes = ctk.CTkTextbox(note_frame, wrap="word", font=ctk.CTkFont(size=12))
        self.overview_notes.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))

    def _build_data_tab(self) -> None:
        tab = self.workflow_tabs.tab("Data")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(tab, corner_radius=18, border_width=1, fg_color="#ffffff")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top,
            text="Historical Total Feeder Load Data",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))

        ctk.CTkLabel(
            top,
            text="Required: timestamp, total_p_mw. Optional: total_q_mvar, temperature_c.",
            text_color="#64748b",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))

        ctk.CTkButton(
            top,
            text="Upload Load CSV",
            command=self.load_csv_callback,
            height=38,
            fg_color="#475569",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=16, pady=14)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.history_table = DataFrameTable(body, "Historical Load Preview")
        self.history_table.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        side = ctk.CTkFrame(body, corner_radius=18, border_width=1, fg_color="#ffffff")
        side.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            side,
            text="Data Status",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        self.data_status_box = ctk.CTkTextbox(side, wrap="word", font=ctk.CTkFont(family="Consolas", size=12))
        self.data_status_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

    def _build_model_training_tab(self) -> None:
        tab = self.workflow_tabs.tab("Model Training")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(tab, corner_radius=18, border_width=1, fg_color="#ffffff")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top,
            text="Forecast Model Training",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))

        ctk.CTkLabel(
            top,
            text="Train gradient-boosting models for total P and optional total Q.",
            text_color="#64748b",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        for c in range(4):
            row.grid_columnconfigure(c, weight=1)

        self.entries: Dict[str, ctk.CTkEntry] = {}

        fields = [
            ("Test size", "test_size", "0.2"),
            ("Max boosting iter", "max_iter", "300"),
            ("Learning rate", "learning_rate", "0.05"),
        ]
        for col, (label, key, default) in enumerate(fields):
            ctk.CTkLabel(row, text=label, text_color="#475569", anchor="w").grid(row=0, column=col, sticky="ew", padx=5)
            entry = ctk.CTkEntry(row, height=32)
            entry.grid(row=1, column=col, sticky="ew", padx=5)
            entry.insert(0, default)
            self.entries[key] = entry

        ctk.CTkButton(
            row,
            text="Train Forecast Model",
            command=self.train_callback,
            height=36,
            fg_color="#2563eb",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=1, column=3, sticky="ew", padx=5)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.metrics_table = DataFrameTable(body, "Forecast Model Metrics")
        self.metrics_table.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.model_notes = ctk.CTkTextbox(body, wrap="word", font=ctk.CTkFont(family="Consolas", size=12))
        self.model_notes.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    def _build_forecast_tab(self) -> None:
        tab = self.workflow_tabs.tab("Forecast")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(tab, corner_radius=18, border_width=1, fg_color="#ffffff")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top,
            text="Generate Total Feeder Load Forecast",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))

        ctk.CTkLabel(
            top,
            text="Forecast total feeder load for the selected horizon. Bus-level forecasting remains a later extension.",
            text_color="#64748b",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, weight=3)

        ctk.CTkLabel(row, text="Forecast horizon h", text_color="#475569", anchor="w").grid(row=0, column=0, sticky="ew", padx=5)
        self.entries["horizon_hours"] = ctk.CTkEntry(row, height=32)
        self.entries["horizon_hours"].grid(row=1, column=0, sticky="ew", padx=5)
        self.entries["horizon_hours"].insert(0, "24")

        ctk.CTkButton(
            row,
            text="Forecast Total Load",
            command=self.forecast_callback,
            height=36,
            fg_color="#0891b2",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=1, column=1, sticky="ew", padx=5)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self.forecast_plot = ForecastPlot(body)
        self.forecast_plot.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.forecast_table = DataFrameTable(body, "Forecast Table")
        self.forecast_table.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    def _build_risk_tab(self) -> None:
        tab = self.workflow_tabs.tab("Forecasted PF Risk")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(tab, corner_radius=18, border_width=1, fg_color="#ffffff")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top,
            text="Forecasted Power-Flow Risk",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(9, 1))

        ctk.CTkLabel(
            top,
            text="Hourly bus-load allocation and deterministic power-flow assessment",
            text_color="#64748b",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        for c in range(5):
            row.grid_columnconfigure(c, weight=1)

        fields = [
            ("V min pu", "vmin_limit", "0.95"),
            ("V max pu", "vmax_limit", "1.05"),
            ("PF max iter", "pf_max_iter", "100"),
            ("PF tolerance", "pf_tolerance", "1e-6"),
        ]
        for col, (label, key, default) in enumerate(fields):
            ctk.CTkLabel(row, text=label, text_color="#475569", anchor="w").grid(row=0, column=col, sticky="ew", padx=5)
            entry = ctk.CTkEntry(row, height=30)
            entry.grid(row=1, column=col, sticky="ew", padx=5)
            entry.insert(0, default)
            self.entries[key] = entry

        ctk.CTkButton(
            row,
            text="Run Forecasted PF",
            command=self.run_pf_callback,
            height=32,
            fg_color="#7c3aed",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=1, column=4, sticky="ew", padx=5)

        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self.risk_summary_table = DataFrameTable(body, "Risk Summary")
        self.risk_summary_table.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.risk_table = DataFrameTable(body, "Forecasted PF Risk Over Time")
        self.risk_table.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    def _build_export_log_tab(self) -> None:
        tab = self.workflow_tabs.tab("Export / Log")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(tab, corner_radius=18, border_width=1, fg_color="#ffffff")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top,
            text="Export Results and Workflow Log",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color="#0f172a",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))

        ctk.CTkLabel(
            top,
            text="Export historical load, model metrics, forecast, and forecasted power-flow risk to Excel.",
            text_color="#64748b",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))

        ctk.CTkButton(
            top,
            text="Export Results",
            command=self.export_callback,
            height=38,
            fg_color="#15803d",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=16, pady=14)

        self.log_box = ctk.CTkTextbox(tab, wrap="word", font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    # ------------------------------------------------------------------
    # API expected by LoadForecastingDashboardApp callbacks
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        cfg = {
            "horizon_hours": int(float(self.entries["horizon_hours"].get().strip())),
            "test_size": float(self.entries["test_size"].get().strip()),
            "max_iter": int(float(self.entries["max_iter"].get().strip())),
            "learning_rate": float(self.entries["learning_rate"].get().strip()),
            "vmin_limit": float(self.entries["vmin_limit"].get().strip()),
            "vmax_limit": float(self.entries["vmax_limit"].get().strip()),
            "pf_max_iter": int(float(self.entries["pf_max_iter"].get().strip())),
            "pf_tolerance": float(self.entries["pf_tolerance"].get().strip()),
        }
        if cfg["horizon_hours"] <= 0 or cfg["horizon_hours"] > 24 * 14:
            raise ValueError("Forecast horizon must be between 1 and 336 hours.")
        if not (0.05 <= cfg["test_size"] <= 0.5):
            raise ValueError("Test size must be between 0.05 and 0.5.")
        if cfg["vmin_limit"] <= 0 or cfg["vmax_limit"] <= cfg["vmin_limit"]:
            raise ValueError("Voltage limits must satisfy 0 < V min < V max.")
        if cfg["pf_max_iter"] <= 0 or cfg["pf_tolerance"] <= 0:
            raise ValueError("Power-flow max iterations and tolerance must be positive.")
        return cfg

    def set_badge(self, text: str, color: str = "#64748b") -> None:
        self.status_badge.configure(text=text, fg_color=color)

    def append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", text)
        self.log_box.configure(state="disabled")

    def _set_textbox(self, box: ctk.CTkTextbox, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    def clear(self) -> None:
        self.set_badge("NOT READY")
        self.data_card.set("No CSV", "Upload historical data", "info")
        self.model_card.set("Not trained", "Train model", "purple")
        self.forecast_card.set("Not run", "Forecast total feeder load", "neutral")
        self.risk_card.set("Unknown", "Run forecasted PF", "neutral")

        self.overview_plot.clear()
        self.forecast_plot.clear()
        self.history_table.set_dataframe(pd.DataFrame())
        self.metrics_table.set_dataframe(pd.DataFrame())
        self.forecast_table.set_dataframe(pd.DataFrame())
        self.risk_summary_table.set_dataframe(pd.DataFrame({"Status": ["No forecasted PF results yet."]}))
        self.overview_summary_table.set_dataframe(pd.DataFrame({"Status": ["No forecasted PF results yet."]}))
        self.risk_table.set_dataframe(pd.DataFrame())

        self._set_textbox(
            self.overview_notes,
            "Use the workflow tabs from left to right:\n\n"
            "1. Data\n"
            "2. Model Training\n"
            "3. Forecast\n"
            "4. Forecasted PF Risk\n"
            "5. Export / Log\n\n"
            "The deterministic solver remains the source of truth.",
        )
        self._set_textbox(
            self.data_status_box,
            "Required CSV:\n"
            "timestamp,total_p_mw\n\n"
            "Optional:\n"
            "total_q_mvar,temperature_c\n\n"
            "Use the same Q sign convention as the simulator.",
        )
        self._set_textbox(
            self.model_notes,
            "Model: Gradient Boosting Regressor\n\n"
            "Purpose: forecast total feeder load only.\n"
            "Bus-level forecasting is a later extension.",
        )
        self.set_log(
            "Workflow initialized.\n\n"
            "Expected CSV format:\n"
            "timestamp,total_p_mw,total_q_mvar,temperature_c\n"
            "2025-01-01 00:00,3.42,-2.10,22.5\n"
        )

    def set_history(self, history: pd.DataFrame, path: str) -> None:
        self.history_table.set_dataframe(history.tail(200))
        filename = path.split("/")[-1].split("\\")[-1]
        self.data_card.set(f"{len(history)} rows", filename, "ok")
        self.set_badge("CSV LOADED", "#0f766e")

        try:
            start = pd.to_datetime(history["timestamp"]).min()
            end = pd.to_datetime(history["timestamp"]).max()
            text = (
                f"Loaded file: {filename}\n"
                f"Rows: {len(history)}\n"
                f"Columns: {', '.join([str(c) for c in history.columns])}\n"
                f"Start: {start}\n"
                f"End: {end}\n\n"
                f"Next step: Model Training."
            )
        except Exception:
            text = f"Loaded file: {filename}\nRows: {len(history)}\nNext step: Model Training."

        self._set_textbox(self.data_status_box, text)
        self.append_log(f"Loaded historical load CSV: {path}")
        self.append_log(f"Rows: {len(history)}, columns: {list(history.columns)}")
        self.workflow_tabs.set("Data")

    def set_metrics(self, metrics: pd.DataFrame) -> None:
        self.metrics_table.set_dataframe(metrics)
        value, subtitle, state = "Trained", "Gradient boosting ready", "ok"

        try:
            row = metrics.loc[metrics["Target"].astype(str) == "total_p_mw"].iloc[0]
            r2 = float(row["R2"])
            mae = float(row["MAE"])
            value = f"R² {r2:.3f}"
            subtitle = f"MAE {mae:.4f} MW"
            state = "ok" if r2 >= 0.95 else ("warning" if r2 >= 0.85 else "critical")
            self._set_textbox(
                self.model_notes,
                f"Model trained successfully.\n\n"
                f"Total P forecast:\n"
                f"R² = {r2:.4f}\n"
                f"MAE = {mae:.5f} MW\n\n"
                f"Next step: Forecast.",
            )
        except Exception:
            self._set_textbox(self.model_notes, "Model trained. Review the metrics table for details.")

        self.model_card.set(value, subtitle, state)
        self.set_badge("MODEL TRAINED", "#15803d")
        self.append_log("Load forecasting model trained.")
        self.workflow_tabs.set("Model Training")

    def set_forecast(self, forecast_df: pd.DataFrame) -> None:
        self.forecast_table.set_dataframe(forecast_df)
        self.forecast_plot.plot_forecast(forecast_df)
        self.overview_plot.plot_forecast(forecast_df)
        self.forecast_card.set(f"{len(forecast_df)} h", "Total feeder forecast ready", "ok")
        self.set_badge("FORECAST READY", "#0891b2")
        self.append_log(f"Generated total feeder forecast for {len(forecast_df)} hour(s).")
        self.workflow_tabs.set("Forecast")

    def set_risk(self, forecast_df: pd.DataFrame, risk_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        self.risk_table.set_dataframe(risk_df)
        self.risk_summary_table.set_dataframe(summary_df)
        self.overview_summary_table.set_dataframe(summary_df)
        self.forecast_plot.plot_forecast(forecast_df, risk_df=risk_df)
        self.overview_plot.plot_forecast(forecast_df, risk_df=risk_df)

        risk_state, risk_value, risk_sub = "ok", "Low", "No serious issues"
        try:
            sev = risk_df["Severity"].astype(str).str.lower().tolist()
            n_crit = sum(1 for s in sev if s in ["critical", "error"])
            n_warn = sum(1 for s in sev if s == "warning")
            if n_crit:
                risk_state, risk_value, risk_sub = "critical", "Critical", f"{n_crit} critical hour(s)"
            elif n_warn:
                risk_state, risk_value, risk_sub = "warning", "Warning", f"{n_warn} warning hour(s)"
        except Exception:
            pass

        self.forecast_card.set(f"{len(risk_df)} h", "Forecasted PF complete", "ok")
        self.risk_card.set(risk_value, risk_sub, risk_state)
        self.set_badge("PF RISK READY", "#7c3aed")
        self.append_log(f"Completed forecasted power flow for {len(risk_df)} hour(s).")
        self.workflow_tabs.set("Forecasted PF Risk")

    def set_error(self, message: str) -> None:
        self.set_badge("ERROR", "#dc2626")
        self.risk_card.set("Error", "Check Export / Log tab", "critical")
        self.append_log("ERROR: " + message)
        self.workflow_tabs.set("Export / Log")


class PowerSystemSimulatorApp(ctk.CTk):
    """Consolidated desktop application. No imports from earlier UI versions.

    Reusable widgets live in this same file; numerical, project, AI, and forecasting
    engines remain separate backend modules by design.
    """

    def __init__(self) -> None:
        # Forecast state is initialized before the GUI, matching the original
        # application's callback and setup order.
        self.load_history_df: Optional[pd.DataFrame] = None
        self.load_forecast_bundle: Optional[LoadForecastBundle] = None
        self.load_forecast_df: Optional[pd.DataFrame] = None
        self.forecast_risk_df: Optional[pd.DataFrame] = None
        self.forecast_risk_summary_df: Optional[pd.DataFrame] = None

        super().__init__()
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(1400, 820)

        self.power_flow_result = None
        self.fault_result = None
        self.ai_report = None
        self.ml_dataset = None
        self.ml_scenario_summary = None
        self.ml_model_bundle: Optional[MLModelBundle] = None
        self.ml_prediction = None
        self._configure_style()
        self._build_layout()
        self.load_default_project()
        self._install_load_forecasting_tab()

    def _build_settings_tab(self) -> None:
        frame = ctk.CTkFrame(self.tabs.tab("Settings"))
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)

        fields = [
            ("Project name", "project_name", "Untitled Project"),
            ("Base power (MVA)", "base_power_mva", "100"),
            ("Frequency (Hz)", "frequency_hz", "60"),
            ("Legacy current scale (unused)", "default_base_current_a", "1"),
            ("Default ampacity (pu)", "default_ampacity_pu", "10"),
        ]
        self.setting_entries: Dict[str, ctk.CTkEntry] = {}
        for row, (label, key, default) in enumerate(fields):
            ctk.CTkLabel(frame, text=label, anchor="w").grid(row=row, column=0, sticky="ew", padx=12, pady=8)
            entry = ctk.CTkEntry(frame)
            entry.grid(row=row, column=1, sticky="ew", padx=12, pady=8)
            entry.insert(0, default)
            self.setting_entries[key] = entry

        ctk.CTkLabel(frame, text="Notes", anchor="nw").grid(row=len(fields), column=0, sticky="new", padx=12, pady=8)
        self.notes_box = ctk.CTkTextbox(frame, height=160)
        self.notes_box.grid(row=len(fields), column=1, sticky="nsew", padx=12, pady=8)
        frame.grid_rowconfigure(len(fields), weight=1)
    def _build_editor_tabs(self) -> None:
        self.bus_editor = EditableTable(
            self.tabs.tab("Buses"),
            "Bus Data",
            ["bus_id", "bus_type", "voltage_pu", "angle_deg", "p_load_pu", "q_load_pu", "p_gen_pu", "base_kv"],
            defaults={"bus_type": "PQ", "voltage_pu": "1", "angle_deg": "0", "p_load_pu": "0", "q_load_pu": "0", "p_gen_pu": "0", "base_kv": "230"},
        )
        self.bus_editor.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.branch_editor = EditableTable(
            self.tabs.tab("Branches"),
            "Branch / Line / Transformer-Equivalent Data",
            ["name", "from_bus", "to_bus", "r_pu", "x_pu", "b_pu", "ampacity_pu", "ampacity_verified", "element_type"],
            defaults={"r_pu": "0.01", "x_pu": "0.1", "b_pu": "0", "ampacity_pu": "10", "ampacity_verified": "False", "element_type": "Line"},
        )
        self.branch_editor.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.source_editor = EditableTable(
            self.tabs.tab("Sequence Sources"),
            "Sequence Source Data for Fault Studies",
            ["name", "bus", "r1_pu", "x1_pu", "r2_pu", "x2_pu", "r0_pu", "x0_pu"],
            defaults={"r1_pu": "0", "x1_pu": "0.1", "r2_pu": "0", "x2_pu": "0.1", "r0_pu": "0", "x0_pu": "0.1"},
        )
        self.source_editor.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    @staticmethod
    def _float(value: str, field: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be numeric. Got: {value!r}") from exc
    def collect_project(self) -> PowerSystemProject:
        settings = ProjectSettings(
            name=self.setting_entries["project_name"].get().strip() or "Untitled Project",
            base_power_mva=self._float(self.setting_entries["base_power_mva"].get(), "base_power_mva"),
            frequency_hz=self._float(self.setting_entries["frequency_hz"].get(), "frequency_hz"),
            default_base_current_a=self._float(self.setting_entries["default_base_current_a"].get(), "default_base_current_a"),
            default_ampacity_pu=self._float(self.setting_entries["default_ampacity_pu"].get(), "default_ampacity_pu"),
            notes=self.notes_box.get("1.0", "end").strip(),
        )
        buses = [
            BusRecord(
                bus_id=row["bus_id"],
                bus_type=row["bus_type"],
                voltage_pu=self._float(row["voltage_pu"], "voltage_pu"),
                angle_deg=self._float(row["angle_deg"], "angle_deg"),
                p_load_pu=self._float(row["p_load_pu"], "p_load_pu"),
                q_load_pu=self._float(row["q_load_pu"], "q_load_pu"),
                p_gen_pu=self._float(row["p_gen_pu"], "p_gen_pu"),
                base_kv=self._float(row["base_kv"], "base_kv"),
            )
            for row in self.bus_editor.get_rows()
        ]
        branches = [
            BranchRecord(
                name=row["name"],
                from_bus=row["from_bus"],
                to_bus=row["to_bus"],
                r_pu=self._float(row["r_pu"], "r_pu"),
                x_pu=self._float(row["x_pu"], "x_pu"),
                b_pu=self._float(row["b_pu"], "b_pu"),
                ampacity_pu=self._float(row["ampacity_pu"], "ampacity_pu"),
                ampacity_verified=str(row.get("ampacity_verified", "False")).strip().lower() in {"true", "1", "yes"},
                element_type=row["element_type"],
            )
            for row in self.branch_editor.get_rows()
        ]
        sequence_sources = [
            SequenceSourceRecord(
                name=row["name"],
                bus=row["bus"],
                r1_pu=self._float(row["r1_pu"], "r1_pu"),
                x1_pu=self._float(row["x1_pu"], "x1_pu"),
                r2_pu=self._float(row["r2_pu"], "r2_pu"),
                x2_pu=self._float(row["x2_pu"], "x2_pu"),
                r0_pu=self._float(row["r0_pu"], "r0_pu"),
                x0_pu=self._float(row["x0_pu"], "x0_pu"),
            )
            for row in self.source_editor.get_rows()
        ]
        return PowerSystemProject(settings, buses, branches, sequence_sources)
    def display_project(self, project: PowerSystemProject) -> None:
        self.setting_entries["project_name"].delete(0, "end")
        self.setting_entries["project_name"].insert(0, project.settings.name)
        self.setting_entries["base_power_mva"].delete(0, "end")
        self.setting_entries["base_power_mva"].insert(0, str(project.settings.base_power_mva))
        self.setting_entries["frequency_hz"].delete(0, "end")
        self.setting_entries["frequency_hz"].insert(0, str(project.settings.frequency_hz))
        self.setting_entries["default_base_current_a"].delete(0, "end")
        self.setting_entries["default_base_current_a"].insert(0, str(project.settings.default_base_current_a))
        self.setting_entries["default_ampacity_pu"].delete(0, "end")
        self.setting_entries["default_ampacity_pu"].insert(0, str(project.settings.default_ampacity_pu))
        self.notes_box.delete("1.0", "end")
        self.notes_box.insert("1.0", project.settings.notes)
        self.bus_editor.set_rows([b.__dict__ for b in project.buses])
        self.branch_editor.set_rows([b.__dict__ for b in project.branches])
        self.source_editor.set_rows([s.__dict__ for s in project.sequence_sources])
    def _base_load_default_project(self) -> None:
        self.display_project(default_7_bus_project())
        self.status_label.configure(text="Status: default 7-bus project loaded; generic adapter accepts user-defined bus counts")
    def _show_validation_report(self, errors: List[str], warnings: List[str]) -> None:
        rows: List[Dict[str, str]] = []
        for item in errors:
            rows.append({"Severity": "Error", "Message": item})
        for item in warnings:
            rows.append({"Severity": "Warning", "Message": item})
        if not rows:
            rows.append({"Severity": "OK", "Message": "Project validation passed. No blocking issues detected."})
        self.validation_table.set_dataframe(pd.DataFrame(rows))
        self.tabs.set("Validation")
    @staticmethod
    def _is_sequence_source_warning(message: str) -> bool:
        text = str(message).lower()
        return "sequence sources" in text or "sequence y-bus" in text
    def _warnings_for_power_flow(self, warnings: List[str]) -> List[str]:
        """Power flow does not require sequence-source data; fault study does."""
        return [w for w in warnings if not self._is_sequence_source_warning(w)]
    def _warnings_for_fault_study(self, warnings: List[str]) -> List[str]:
        filtered = [w for w in warnings if not self._is_sequence_source_warning(w)]
        if self.project_has_no_sequence_sources():
            filtered.append("Sequence data is required for fault analysis, but not for ordinary power flow.")
        return filtered
    def project_has_no_sequence_sources(self) -> bool:
        try:
            return len(self.source_editor.get_rows()) == 0
        except Exception:
            return True
    def _validate_fault_inputs(self, project: PowerSystemProject) -> tuple[list[str], int | None, complex | None]:
        errors: list[str] = []
        faulted_bus: int | None = None
        fault_impedance: complex | None = None
        bus_ids = {str(bus.bus_id).strip() for bus in project.buses}
        if not getattr(project, "sequence_sources", []):
            errors.append(
                "Fault study requires at least one sequence source/equivalent source. "
                "Power flow can run without sequence sources, but fault analysis may produce a singular sequence Y-bus without them."
            )
        raw_bus = self.fault_bus_entry.get().strip()
        try:
            faulted_bus = int(raw_bus)
            if str(faulted_bus) not in bus_ids:
                errors.append(f"Faulted bus {faulted_bus} does not exist in the bus table.")
        except ValueError:
            errors.append("Faulted bus must be an integer bus ID, for example 4.")

        raw_z = self.fault_z_entry.get().strip() or "0"
        try:
            fault_impedance = complex(raw_z)
        except ValueError:
            errors.append("Fault impedance must be numeric. Examples: 0, 0.01, 0.01+0.02j.")
        return errors, faulted_bus, fault_impedance
    def _base_validate_project(self) -> bool:
        try:
            project = self.collect_project()
            ok, errors, warnings = validate_project_generic(project)
            power_flow_warnings = self._warnings_for_power_flow(warnings)
            self._show_validation_report(errors, power_flow_warnings)
            if ok:
                self.status_label.configure(
                    text="Status: validation passed for power flow; sequence data is checked when running fault study"
                )
                return True
            self.status_label.configure(text="Status: validation failed")
            return False
        except Exception as exc:
            self._show_validation_report([str(exc)], [])
            self.status_label.configure(text="Status: validation error")
            return False
    def _base_run_power_flow(self) -> None:
        try:
            project = self.collect_project()
            ok, errors, warnings = validate_project_generic(project)
            power_flow_warnings = self._warnings_for_power_flow(warnings)
            self._show_validation_report(errors, power_flow_warnings)
            if not ok:
                messagebox.showerror("Project is invalid", "Fix the errors shown in the Validation tab before running power flow.")
                return
            max_iterations = int(self.max_iter_entry.get())
            tolerance = float(self.tolerance_entry.get())
            if max_iterations <= 0:
                raise ValueError("Max iterations must be a positive integer.")
            if tolerance <= 0:
                raise ValueError("Tolerance must be a positive number.")
            solver_method = self.solver_method_menu.get()
            result = run_power_flow_project(
                project,
                max_iterations=max_iterations,
                tolerance=tolerance,
                suppress_output=True,
                solver_method=solver_method,
            )
            self.power_flow_result = result
            self.bus_table.set_dataframe(result.bus_table)
            self.current_table.set_dataframe(result.line_current_table)
            self.flow_table.set_dataframe(result.line_flow_table)
            self.loss_table.set_dataframe(result.losses_table_pu)
            self.voltage_plot.update_plot(result.bus_table)
            self.summary_panel.update_summary(result, project, requested_solver=solver_method)
            self.generate_ai_insights(auto=True)
            self.update_idletasks()
            warning_text = ""
            if power_flow_warnings:
                warning_text = "\nWarnings: " + "; ".join(power_flow_warnings)
            solver_label = result.summary.get("solver_method", solver_method)
            self.status_label.configure(text=f"Status: [{solver_label}] {result.message}{warning_text}")
            self.tabs.set("Summary")
        except Exception as exc:
            messagebox.showerror("Power flow failed", str(exc))
            self._show_validation_report([str(exc)], [])
            self.status_label.configure(text="Status: power flow failed")
    def _base_generate_ai_insights(self, auto: bool = False) -> None:
        """Generate the AI Engineering Advisor report from the latest power-flow result."""
        try:
            if self.power_flow_result is None:
                if not auto:
                    messagebox.showwarning("No power-flow result", "Run power flow before generating AI insights.")
                return
            project = self.collect_project()
            thresholds = self.ai_panel.get_thresholds()
            report = build_ai_insights(self.power_flow_result, project=project, thresholds=thresholds)
            self.ai_report = report
            self.ai_panel.set_report(report)
            if not auto:
                self.status_label.configure(text="Status: AI insights generated")
                self.tabs.set("AI Insights")
        except Exception as exc:
            self.ai_report = None
            self.ai_panel.set_error(str(exc))
            if not auto:
                messagebox.showerror("AI Advisor failed", str(exc))
                self.status_label.configure(text="Status: AI Advisor failed")
    def generate_ml_dataset(self) -> None:
        """Generate randomized scenarios and solve them to create ML training data."""
        try:
            project = self.collect_project()
            ok, errors, warnings = validate_project_generic(project)
            power_flow_warnings = self._warnings_for_power_flow(warnings)
            self._show_validation_report(errors, power_flow_warnings)
            if not ok:
                messagebox.showerror("Project is invalid", "Fix project validation errors before generating ML data.")
                return
            cfg = self.ml_panel.get_config()
            solver_method = self.solver_method_menu.get()
            max_iterations = int(self.max_iter_entry.get())
            tolerance = float(self.tolerance_entry.get())
            self.ml_panel.set_status("training", f"Generating {cfg['n_scenarios']} scenarios using {solver_method}. Please wait...")
            self.status_label.configure(text="Status: generating ML training dataset...")
            self.update_idletasks()

            def progress(i: int, n: int) -> None:
                # Keep UI responsive without updating too frequently.
                if i == 1 or i == n or i % max(1, n // 10) == 0:
                    self.ml_panel.set_status("training", f"Generating ML dataset: {i}/{n} scenarios completed...")
                    self.status_label.configure(text=f"Status: ML dataset generation {i}/{n}")
                    self.update_idletasks()

            dataset, summary = generate_training_dataset(
                project,
                n_scenarios=cfg["n_scenarios"],
                solver_method=solver_method,
                max_iterations=max_iterations,
                tolerance=tolerance,
                global_load_min=cfg["global_load_min"],
                global_load_max=cfg["global_load_max"],
                local_variation_percent=cfg["local_variation_percent"],
                voltage_min_pu=cfg["voltage_min_pu"],
                voltage_max_pu=cfg["voltage_max_pu"],
                random_seed=cfg["random_seed"],
                progress_callback=progress,
            )
            self.ml_dataset = dataset
            self.ml_scenario_summary = summary
            self.ml_model_bundle = None
            self.ml_prediction = None
            self.ml_panel.set_dataset(dataset, summary)
            self.status_label.configure(text=f"Status: ML dataset generated with {len(dataset)} successful scenarios")
            self.tabs.set("Machine Learning")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("ML dataset generation failed", str(exc))
            self.status_label.configure(text="Status: ML dataset generation failed")
    def _base_train_ml_model(self) -> None:
        """Train gradient-boosted decision-tree models from the generated dataset."""
        try:
            if self.ml_dataset is None or self.ml_dataset.empty:
                messagebox.showwarning("No ML dataset", "Generate an ML dataset before training the model.")
                return
            cfg = self.ml_panel.get_config()
            self.ml_panel.set_status("training", "Training gradient-boosted decision-tree models. Please wait...")
            self.status_label.configure(text="Status: training ML model...")
            self.update_idletasks()
            bundle = train_gradient_boosting_models(
                self.ml_dataset,
                voltage_min_pu=cfg["voltage_min_pu"],
                voltage_max_pu=cfg["voltage_max_pu"],
                test_size=cfg["test_size"],
                random_seed=cfg["random_seed"],
                scenario_summary=self.ml_scenario_summary,
            )
            self.ml_model_bundle = bundle
            self.ml_prediction = None
            self.ml_panel.set_metrics(bundle.metrics_table, bundle.training_summary)
            self.status_label.configure(text="Status: ML model trained")
            self.tabs.set("Machine Learning")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("ML training failed", str(exc))
            self.status_label.configure(text="Status: ML training failed")
    def predict_ml_current_case(self) -> None:
        """Predict current project KPIs using the trained ML model."""
        try:
            if self.ml_model_bundle is None:
                messagebox.showwarning("No ML model", "Train or load an ML model before prediction.")
                return
            project = self.collect_project()
            prediction_table, prediction = predict_current_case(self.ml_model_bundle, project)
            self.ml_prediction = prediction
            self.ml_panel.set_prediction(prediction_table)
            if self.power_flow_result is not None:
                comparison = compare_prediction_with_solver(prediction, self.power_flow_result)
                self.ml_panel.set_comparison(comparison)
            self.status_label.configure(text="Status: ML prediction generated")
            self.tabs.set("Machine Learning")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("ML prediction failed", str(exc))
            self.status_label.configure(text="Status: ML prediction failed")
    def compare_ml_with_solver(self) -> None:
        """Compare the latest ML prediction with the latest deterministic solver result."""
        try:
            if self.ml_prediction is None:
                self.predict_ml_current_case()
                if self.ml_prediction is None:
                    return
            if self.power_flow_result is None:
                messagebox.showwarning("No solver result", "Run power flow before comparing ML prediction with the solver.")
                return
            comparison = compare_prediction_with_solver(self.ml_prediction, self.power_flow_result)
            self.ml_panel.set_comparison(comparison)
            self.status_label.configure(text="Status: ML prediction compared with deterministic solver")
            self.tabs.set("Machine Learning")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("ML comparison failed", str(exc))
    def save_ml_model(self) -> None:
        """Save trained ML model bundle to a joblib file."""
        try:
            if self.ml_model_bundle is None:
                messagebox.showwarning("No ML model", "Train a model before saving it.")
                return
            path = filedialog.asksaveasfilename(
                title="Save ML model",
                defaultextension=".joblib",
                filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            )
            if not path:
                return
            save_model_bundle(self.ml_model_bundle, path)
            self.ml_panel.append_log(f"Model saved to: {path}")
            self.status_label.configure(text=f"Status: ML model saved to {path}")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("Save ML model failed", str(exc))
    def _base_load_ml_model(self) -> None:
        """Load a saved ML model bundle."""
        try:
            path = filedialog.askopenfilename(
                title="Load ML model",
                filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            )
            if not path:
                return
            bundle = load_model_bundle(path)
            self.ml_model_bundle = bundle
            self.ml_prediction = None
            self.ml_panel.set_metrics(bundle.metrics_table, bundle.training_summary)
            self.ml_panel.append_log(f"Model loaded from: {path}")
            self.status_label.configure(text=f"Status: ML model loaded from {path}")
            self.tabs.set("Machine Learning")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("Load ML model failed", str(exc))
    def export_ml_dataset_csv(self) -> None:
        """Export the generated ML dataset to CSV."""
        try:
            if self.ml_dataset is None or self.ml_dataset.empty:
                messagebox.showwarning("No ML dataset", "Generate an ML dataset before exporting it.")
                return
            path = filedialog.asksaveasfilename(
                title="Export ML dataset",
                defaultextension=".csv",
                filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            self.ml_dataset.to_csv(path, index=False)
            self.ml_panel.append_log(f"Dataset exported to: {path}")
            self.status_label.configure(text=f"Status: ML dataset exported to {path}")
        except Exception as exc:
            self.ml_panel.set_error(str(exc))
            messagebox.showerror("Export ML dataset failed", str(exc))
    def _base_run_fault_study(self) -> None:
        try:
            project = self.collect_project()
            ok, errors, warnings = validate_project_generic(project)
            fault_errors, faulted_bus, fault_impedance = self._validate_fault_inputs(project)
            all_errors = errors + fault_errors
            fault_warnings = self._warnings_for_fault_study(warnings)
            self._show_validation_report(all_errors, fault_warnings)
            if not ok or fault_errors:
                messagebox.showerror("Project is invalid", "Fix the errors shown in the Validation tab before running the fault study.")
                return
            assert faulted_bus is not None
            assert fault_impedance is not None
            result = run_fault_study_project(
                project,
                fault_type=self.fault_type_menu.get(),
                faulted_bus=faulted_bus,
                prefault_voltage=1.0,
                fault_impedance=fault_impedance,
                suppress_output=True,
            )
            self.fault_result = result
            combined = result.current_table
            if not result.voltage_table.empty:
                voltage = result.voltage_table.copy()
                voltage.insert(0, "Table", "Voltage")
                current = result.current_table.copy()
                current.insert(0, "Table", "Current")
                combined = pd.concat([current, voltage], ignore_index=True, sort=False)
            self.fault_table.set_dataframe(combined)
            self.status_label.configure(text=f"Status: {result.message}")
            self.tabs.set("Fault Results")
        except Exception as exc:
            messagebox.showerror("Fault study failed", str(exc))
            self._show_validation_report([str(exc)], [])
            self.status_label.configure(text="Status: fault study failed")
    def _base_clear_results(self) -> None:
        """Clear all result displays without changing the project input data."""
        self.power_flow_result = None
        self.fault_result = None
        self.ai_report = None
        self.bus_table.set_dataframe(pd.DataFrame())
        self.current_table.set_dataframe(pd.DataFrame())
        self.flow_table.set_dataframe(pd.DataFrame())
        self.loss_table.set_dataframe(pd.DataFrame())
        self.fault_table.set_dataframe(pd.DataFrame())
        self.summary_panel.clear()
        self.ai_report = None
        self.ai_panel.clear()
        self.voltage_plot.clear()
        self.validation_table.set_dataframe(pd.DataFrame())
        self.status_label.configure(text="Status: results cleared")
    @staticmethod
    def _safe_excel_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Return a copy of df that is safe for Excel export."""
        if df is None or df.empty:
            return pd.DataFrame({"Status": ["No data"]})
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].map(lambda x: f"{x.real:.10g} + j{x.imag:.10g}" if isinstance(x, complex) else x)
        return out
    def export_results_excel(self) -> None:
        """Export the latest power-flow/fault results and project input data to Excel."""
        if self.power_flow_result is None and self.fault_result is None:
            messagebox.showwarning(
                "No results to export",
                "Run a power-flow study or a fault study before exporting results.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Export results to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            project = self.collect_project()
            sheets: Dict[str, pd.DataFrame] = {
                "Project Settings": pd.DataFrame([project.settings.__dict__]),
                "Input Buses": pd.DataFrame([b.__dict__ for b in project.buses]),
                "Input Branches": pd.DataFrame([b.__dict__ for b in project.branches]),
                "Sequence Sources": pd.DataFrame([s.__dict__ for s in project.sequence_sources]),
            }

            if self.power_flow_result is not None:
                pf = self.power_flow_result
                sheets.update(
                    {
                        "PF Summary": pd.DataFrame([{k: str(v) if isinstance(v, complex) else v for k, v in pf.summary.items()}]),
                        "Bus Voltages": pf.bus_table,
                        "Line Currents": pf.line_current_table,
                        "Line Flows": pf.line_flow_table,
                        "Losses PU": pf.losses_table_pu,
                        "Losses MVA": pf.losses_table_mva,
                    }
                )
                if pf.log:
                    sheets["PF Solver Log"] = pd.DataFrame({"Log": pf.log.splitlines()})

            if self.fault_result is not None:
                fault = self.fault_result
                sheets.update(
                    {
                        "Fault Currents": fault.current_table,
                        "Fault Voltages": fault.voltage_table,
                    }
                )
                if not fault.sequence_current_table.empty:
                    sheets["Seq Currents"] = fault.sequence_current_table
                if not fault.sequence_voltage_table.empty:
                    sheets["Seq Voltages"] = fault.sequence_voltage_table
                if fault.log:
                    sheets["Fault Log"] = pd.DataFrame({"Log": fault.log.splitlines()})

            if self.ai_report is not None:
                sheets.update(
                    {
                        "AI Summary": pd.DataFrame([{k: str(v) for k, v in self.ai_report.summary.items()}]),
                        "AI Thresholds": pd.DataFrame([self.ai_report.threshold_dict()]),
                        "AI Findings": self.ai_report.findings_table,
                        "AI Actions": self.ai_report.actions_table,
                        "AI Report": pd.DataFrame({"Report": self.ai_report.report_text.splitlines()}),
                    }
                )

            # Keep sheet names unique and within Excel's 31-character limit.
            used_sheet_names: set[str] = set()
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for raw_name, df in sheets.items():
                    name = raw_name[:31]
                    base = name
                    counter = 1
                    while name in used_sheet_names:
                        suffix = f"_{counter}"
                        name = (base[: 31 - len(suffix)] + suffix)
                        counter += 1
                    used_sheet_names.add(name)
                    self._safe_excel_dataframe(df).to_excel(writer, sheet_name=name, index=False)

            self.status_label.configure(text=f"Status: results exported to {path}")
            messagebox.showinfo("Export complete", f"Results exported successfully:\n{path}")
        except ImportError as exc:
            messagebox.showerror(
                "Export failed",
                "Excel export requires openpyxl. Install it with:\n\npip install openpyxl",
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            self.status_label.configure(text="Status: export failed")
    def save_project(self) -> None:
        try:
            project = self.collect_project()
            path = filedialog.asksaveasfilename(
                title="Save project",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            project.save_json(path)
            self.status_label.configure(text=f"Status: project saved to {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
    def load_project(self) -> None:
        try:
            path = filedialog.askopenfilename(
                title="Load project",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            project = PowerSystemProject.load_json(path)
            self.display_project(project)
            ok, errors, warnings = validate_project_generic(project)
            self._show_validation_report(errors, warnings)
            self.status_label.configure(text=f"Status: project loaded from {path}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Treeview",
            rowheight=30,
            font=("Segoe UI", 10),
            background="#ffffff",
            fieldbackground="#ffffff",
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background="#e2e8f0",
            foreground="#0f172a",
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#bfdbfe")], foreground=[("selected", "#0f172a")])
    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self, width=260, corner_radius=0, fg_color="#0f172a")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)
        self._build_sidebar()

        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="#f8fafc")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_ribbon()
        self._build_tabs()
    def _build_sidebar(self) -> None:
        ctk.CTkLabel(
            self.sidebar,
            text="Power System\nSimulation and\nAnalysis Platform",
            text_color="white",
            font=ctk.CTkFont(size=20, weight="bold"),
            justify="left",
            anchor="w",
            wraplength=220,
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(24, 6))
        ctk.CTkLabel(
            self.sidebar,
            text="KACST | FUTURE GRIDS LAB\nAnalysis workspace for power flow, faults, AI insights, and ML screening.",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=12),
            justify="left",
            anchor="w",
            wraplength=210,
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 18))

        # Study settings
        settings = ctk.CTkFrame(self.sidebar, corner_radius=16, fg_color="#1e293b")
        settings.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        settings.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(settings, text="Study Settings", text_color="white", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 8)
        )

        ctk.CTkLabel(settings, text="Solver", text_color="#cbd5e1", anchor="w").grid(row=1, column=0, sticky="ew", padx=14)
        self.solver_method_menu = ctk.CTkOptionMenu(settings, values=available_solver_methods(), height=34)
        self.solver_method_menu.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.solver_method_menu.set("Auto")

        ctk.CTkLabel(settings, text="Max iterations", text_color="#cbd5e1", anchor="w").grid(row=3, column=0, sticky="ew", padx=14)
        self.max_iter_entry = ctk.CTkEntry(settings, height=34)
        self.max_iter_entry.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.max_iter_entry.insert(0, "30")

        ctk.CTkLabel(settings, text="Tolerance", text_color="#cbd5e1", anchor="w").grid(row=5, column=0, sticky="ew", padx=14)
        self.tolerance_entry = ctk.CTkEntry(settings, height=34)
        self.tolerance_entry.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.tolerance_entry.insert(0, "0.0001")

        # Fault settings
        fault = ctk.CTkFrame(self.sidebar, corner_radius=16, fg_color="#1e293b")
        fault.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        fault.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(fault, text="Fault Settings", text_color="white", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 8)
        )
        self.fault_type_menu = ctk.CTkOptionMenu(fault, values=["balanced", "LG", "LL", "LLG"], height=34)
        self.fault_type_menu.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.fault_type_menu.set("balanced")

        ctk.CTkLabel(fault, text="Faulted bus", text_color="#cbd5e1", anchor="w").grid(row=2, column=0, sticky="ew", padx=14)
        self.fault_bus_entry = ctk.CTkEntry(fault, height=34)
        self.fault_bus_entry.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.fault_bus_entry.insert(0, "4")

        ctk.CTkLabel(fault, text="Fault impedance Zf", text_color="#cbd5e1", anchor="w").grid(row=4, column=0, sticky="ew", padx=14)
        self.fault_z_entry = ctk.CTkEntry(fault, height=34)
        self.fault_z_entry.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.fault_z_entry.insert(0, "0")

        # Main actions
        actions = ctk.CTkFrame(self.sidebar, corner_radius=16, fg_color="#1e293b")
        actions.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        actions.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(actions, text="Primary Actions", text_color="white", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 8)
        )
        ctk.CTkButton(actions, text="Validate Project", command=self.validate_project, height=36, fg_color="#0f766e").grid(
            row=1, column=0, sticky="ew", padx=14, pady=4
        )
        ctk.CTkButton(actions, text="Run Power Flow", command=self.run_power_flow, height=38, fg_color="#2563eb").grid(
            row=2, column=0, sticky="ew", padx=14, pady=4
        )
        ctk.CTkButton(actions, text="Run Fault Study", command=self.run_fault_study, height=36, fg_color="#7c3aed").grid(
            row=3, column=0, sticky="ew", padx=14, pady=(4, 12)
        )

        # Appearance/status
        self.appearance_menu = ctk.CTkOptionMenu(self.sidebar, values=["Light", "Dark", "System"], command=ctk.set_appearance_mode, height=32)
        self.appearance_menu.grid(row=5, column=0, sticky="ew", padx=14, pady=(2, 10))
        self.appearance_menu.set("Light")

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Status: ready",
            text_color="#e2e8f0",
            justify="left",
            anchor="w",
            wraplength=220,
        )
        self.status_label.grid(row=6, column=0, sticky="ew", padx=18, pady=(4, 18))
    def _build_header(self) -> None:
        """Brand the main dashboard, without changing the numerical backend.

        Assets are resolved relative to this source file, not the shell's
        working directory. Missing/corrupt assets show a text label instead
        of crashing the GUI.
        """
        header = ctk.CTkFrame(self.main, corner_radius=0, fg_color="#ffffff", border_width=0)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 8))
        header.grid_columnconfigure(0, minsize=133)
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(2, minsize=166)
        self._brand_images = {}  # Keep CTkImage objects alive.

        def brand_label(filename: str, key: str, text: str, size: tuple[int, int]):
            image_path = Path(__file__).resolve().parent / filename
            if image_path.exists():
                try:
                    with Image.open(image_path) as im:
                        image = im.convert("RGBA")
                    ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=size)
                    self._brand_images[key] = ctk_image
                    return ctk.CTkLabel(header, text="", image=ctk_image, fg_color="transparent")
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f"Logo could not be displayed ({filename}): {exc}")
            else:
                print(f"Optional logo not found: {image_path}")
            return ctk.CTkLabel(header, text=text,
                                text_color="#0f234b",
                                font=ctk.CTkFont(size=13, weight="bold"),
                                fg_color="transparent")

        # Compact branding gives the engineering result panes more height.
        brand_label("kacst_logo.png", "kacst", "KACST", (77, 86)).grid(
            row=0, column=0, rowspan=3, sticky="w", padx=(14, 4), pady=5
        )
        brand_label("vision2030_logo.png", "vision", "VISION 2030", (127, 85)).grid(
            row=0, column=2, rowspan=3, sticky="e", padx=(4, 14), pady=5
        )
        ctk.CTkLabel(
            header,
            text="Engineering Dashboard",
            font=ctk.CTkFont(size=25, weight="bold"),
            text_color="#0f172a",
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", padx=10, pady=(9, 0))

        ctk.CTkLabel(
            header,
            text="Power-flow studies, fault studies, AI diagnosis, and ML surrogate screening",
            font=ctk.CTkFont(size=12),
            text_color="#64748b",
            anchor="center",
        ).grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 4))

        self.header_badge = ctk.CTkLabel(
            header,
            text="READY",
            fg_color="#2563eb",
            text_color="white",
            corner_radius=12,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=11,
            pady=4,
        )
        self.header_badge.grid(row=2, column=1, pady=(0, 5))
    def _ribbon_button(self, master: Any, text: str, command: Any, color: str = "#475569") -> None:
        ctk.CTkButton(
            master,
            text=text,
            command=command,
            height=34,
            corner_radius=10,
            fg_color=color,
            hover_color="#1d4ed8",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", padx=4, pady=5)
    def _build_ribbon(self) -> None:
        ribbon = ctk.CTkFrame(self.main, corner_radius=16, fg_color="#e2e8f0")
        ribbon.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        ribbon.grid_columnconfigure(0, weight=1)

        # Left command cluster
        left = ctk.CTkFrame(ribbon, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=10)
        self._ribbon_button(left, "Dashboard", lambda: self.tabs.set("Dashboard"), "#2563eb")
        self._ribbon_button(left, "Load JSON", self.load_project)
        self._ribbon_button(left, "Save JSON", self.save_project)
        self._ribbon_button(left, "Default 7-Bus", self.load_default_project)

        # Right command cluster
        right = ctk.CTkFrame(ribbon, fg_color="transparent")
        right.grid(row=0, column=1, sticky="e", padx=10)
        self._ribbon_button(right, "AI Insights", lambda: self.tabs.set("AI Insights"), "#0891b2")
        self._ribbon_button(right, "Machine Learning", lambda: self.tabs.set("Machine Learning"), "#9333ea")
        self._ribbon_button(right, "Export Excel", self.export_results_excel, "#15803d")
        self._ribbon_button(right, "Clear", self.clear_results, "#991b1b")
    def _build_tabs(self) -> None:
        container = ctk.CTkFrame(self.main, corner_radius=18, border_width=1, fg_color="#ffffff")
        container.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 8))
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(container, corner_radius=16)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        tab_names = [
            "Dashboard",
            "Settings",
            "Buses",
            "Branches",
            "Sequence Sources",
            "Validation",
            "Summary",
            "Voltage Profile",
            "AI Insights",
            "Machine Learning",
            "Bus Voltages",
            "Line Currents",
            "Line Flows",
            "Losses",
            "Fault Results",
        ]

        for name in tab_names:
            self.tabs.add(name)
            self.tabs.tab(name).grid_columnconfigure(0, weight=1)
            self.tabs.tab(name).grid_rowconfigure(0, weight=1)

        self.dashboard_home = DashboardHome(self.tabs.tab("Dashboard"), self)
        self.dashboard_home.grid(row=0, column=0, sticky="nsew")

        self._build_settings_tab()
        self._build_editor_tabs()

        self.validation_table = DataFrameTable(self.tabs.tab("Validation"), "Validation Report")
        self.validation_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.summary_panel = VisualSummaryPanel(self.tabs.tab("Summary"))
        self.summary_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.voltage_plot = DashboardVoltagePlot(self.tabs.tab("Voltage Profile"))
        self.voltage_plot.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.ai_panel = AIInsightsPanel(self.tabs.tab("AI Insights"), generate_callback=self.generate_ai_insights)
        self.ai_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.ml_panel = MachineLearningPanel(
            self.tabs.tab("Machine Learning"),
            generate_dataset_callback=self.generate_ml_dataset,
            train_model_callback=self.train_ml_model,
            predict_callback=self.predict_ml_current_case,
            compare_callback=self.compare_ml_with_solver,
            save_model_callback=self.save_ml_model,
            load_model_callback=self.load_ml_model,
            export_dataset_callback=self.export_ml_dataset_csv,
        )
        self.ml_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.bus_table = DataFrameTable(self.tabs.tab("Bus Voltages"), "Bus Voltage Results")
        self.bus_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.current_table = DataFrameTable(self.tabs.tab("Line Currents"), "Line Current Results")
        self.current_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.flow_table = DataFrameTable(self.tabs.tab("Line Flows"), "Line Flow Results")
        self.flow_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.loss_table = DataFrameTable(self.tabs.tab("Losses"), "Loss Results")
        self.loss_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.fault_table = DataFrameTable(self.tabs.tab("Fault Results"), "Fault Study Results")
        self.fault_table.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    @staticmethod
    def _col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None
    def _set_badge(self, text: str, color: str) -> None:
        self.header_badge.configure(text=text, fg_color=color)
    def _project_info(self) -> None:
        try:
            project = self.collect_project()
            self.dashboard_home.project_card.set(
                project.settings.name or "Untitled",
                f"{len(project.buses)} buses, {len(project.branches)} branches",
                "neutral",
            )
        except Exception:
            self.dashboard_home.project_card.set("Project issue", "Check input data", "warning")
    def _update_dashboard_power_flow(self) -> None:
        self._project_info()

        if self.power_flow_result is None:
            return

        result = self.power_flow_result
        summary = dict(getattr(result, "summary", {}) or {})
        solver = summary.get("solver_method", self.solver_method_menu.get())
        converged = bool(getattr(result, "converged", False))
        message = str(getattr(result, "message", "")).strip()

        self.dashboard_home.solver_card.set(str(solver), "Effective solver", "info")
        self.dashboard_home.status_card.set(
            "Converged" if converged else "Failed",
            message or "Solver status",
            "ok" if converged else "critical",
        )

        bus_table = getattr(result, "bus_table", pd.DataFrame())
        vmin = None
        vmin_bus = "N/A"
        vmax = None
        vmax_bus = "N/A"
        if bus_table is not None and not bus_table.empty:
            v_col = self._col(bus_table, ["Voltage_pu", "Voltage_mag_pu", "V_pu", "Voltage"])
            b_col = self._col(bus_table, ["Bus", "bus_id", "Bus_ID"])
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

        if vmin is not None:
            v_state = "ok" if vmin >= 0.95 else ("warning" if vmin >= 0.90 else "critical")
            self.dashboard_home.vmin_card.set(f"{vmin:.5f} pu", f"Weakest bus: {vmin_bus}", v_state)
            self.dashboard_home.voltage_chart.set_voltage_profile(bus_table)
            self.voltage_plot.set_voltage_profile(bus_table)

        p_loss = summary.get("total_loss_MVA_real", None)
        q_loss = summary.get("total_loss_MVA_imag", None)
        losses_mva = getattr(result, "losses_table_mva", pd.DataFrame())
        if p_loss is None and losses_mva is not None and not losses_mva.empty:
            if "P_loss_MW" in losses_mva.columns:
                p_loss = pd.to_numeric(losses_mva["P_loss_MW"], errors="coerce").sum()
            if "Q_loss_MVAR" in losses_mva.columns:
                q_loss = pd.to_numeric(losses_mva["Q_loss_MVAR"], errors="coerce").sum()

        try:
            p_loss_float = float(p_loss)
        except Exception:
            p_loss_float = None

        if p_loss_float is not None:
            loss_state = "ok" if p_loss_float < 0.5 else ("warning" if p_loss_float < 2.0 else "critical")
            self.dashboard_home.loss_card.set(f"{p_loss_float:.5f} MW", "Total active loss", loss_state)

        health_rows = [
            {
                "Severity": "OK" if converged else "Error",
                "Indicator": "Convergence",
                "Value": "Converged" if converged else "Failed",
                "Meaning": message or "No message",
            },
            {
                "Severity": "OK" if (vmin is not None and vmin >= 0.95) else "Warning",
                "Indicator": "Low-voltage check",
                "Value": f"{vmin:.5f} pu at Bus {vmin_bus}" if vmin is not None else "N/A",
                "Meaning": "Pass" if vmin is not None and vmin >= 0.95 else "Below 0.95 pu threshold",
            },
            {
                "Severity": "OK" if (vmax is not None and vmax <= 1.05) else "Warning",
                "Indicator": "High-voltage check",
                "Value": f"{vmax:.5f} pu at Bus {vmax_bus}" if vmax is not None else "N/A",
                "Meaning": "Pass" if vmax is not None and vmax <= 1.05 else "Above 1.05 pu threshold",
            },
            {
                "Severity": "Info",
                "Indicator": "Loss check",
                "Value": f"{p_loss_float:.5f} MW" if p_loss_float is not None else "N/A",
                "Meaning": "Review detailed branch losses",
            },
        ]
        self.dashboard_home.health_panel.set_dataframe(pd.DataFrame(health_rows))

        action_rows = []
        if vmin is not None and vmin < 0.95:
            action_rows.append(
                {
                    "Severity": "Warning",
                    "Priority": "High",
                    "Action": f"Investigate voltage support near Bus {vmin_bus}",
                    "Reason": "Minimum voltage below 0.95 pu",
                }
            )
            action_rows.append(
                {
                    "Severity": "Info",
                    "Priority": "Medium",
                    "Action": "Test capacitor or DG support scenario",
                    "Reason": "Reactive/active support may improve downstream voltage",
                }
            )
        else:
            action_rows.append(
                {
                    "Severity": "OK",
                    "Priority": "Normal",
                    "Action": "No immediate voltage action",
                    "Reason": "Voltage within selected engineering limits",
                }
            )

        if p_loss_float is not None:
            action_rows.append(
                {
                    "Severity": "Info",
                    "Priority": "Medium",
                    "Action": "Review branch-level losses",
                    "Reason": f"Total real loss is {p_loss_float:.5f} MW",
                }
            )

        self.dashboard_home.action_panel.set_dataframe(pd.DataFrame(action_rows))

        notes = [
            "LATEST POWER-FLOW RESULT",
            "",
            f"Solver: {solver}",
            f"Status: {'Converged' if converged else 'Failed'}",
            f"Minimum voltage: {vmin:.5f} pu at Bus {vmin_bus}" if vmin is not None else "Minimum voltage: N/A",
            f"Maximum voltage: {vmax:.5f} pu at Bus {vmax_bus}" if vmax is not None else "Maximum voltage: N/A",
            f"Total real loss: {p_loss_float:.5f} MW" if p_loss_float is not None else "Total real loss: N/A",
            "",
            "Recommended workflow:",
            "1. Review the System Health panel.",
            "2. Open AI Insights for diagnosis and corrective actions.",
            "3. Use Machine Learning for fast screening only.",
            "4. Verify final decisions using FBS or Newton-Raphson.",
        ]
        self.dashboard_home.set_notes("\n".join(notes))
        self._set_badge("STUDY COMPLETE" if converged else "CHECK RESULT", "#15803d" if converged else "#dc2626")
    def load_default_project(self) -> None:
        self._base_load_default_project()
        self._project_info()
        self.dashboard_home.set_notes("Default 7-bus project loaded. Run validation or power flow to populate the dashboard.")
        self.tabs.set("Dashboard")
        self._set_badge("READY", "#2563eb")
    def validate_project(self) -> bool:
        ok = self._base_validate_project()
        self._project_info()
        self._set_badge("VALIDATED" if ok else "INPUT ISSUE", "#0f766e" if ok else "#d97706")
        try:
            rows = []
            for iid in self.validation_table.tree.get_children():
                vals = self.validation_table.tree.item(iid, "values")
                if len(vals) >= 2:
                    rows.append({"Severity": vals[0], "Indicator": "Validation", "Value": vals[1], "Meaning": ""})
                elif len(vals) == 1:
                    rows.append({"Status": vals[0]})
            if rows:
                self.dashboard_home.health_panel.set_dataframe(pd.DataFrame(rows))
        except Exception:
            pass
        self.tabs.set("Dashboard")
        return ok
    def run_power_flow(self) -> None:
        self._base_run_power_flow()
        self._update_dashboard_power_flow()
        self.tabs.set("Dashboard")
    def generate_ai_insights(self, auto: bool = False) -> None:
        self._base_generate_ai_insights(auto=auto)
        if self.ai_report is not None:
            findings = getattr(self.ai_report, "findings_table", pd.DataFrame())
            actions = getattr(self.ai_report, "actions_table", pd.DataFrame())
            if isinstance(findings, pd.DataFrame) and not findings.empty:
                self.dashboard_home.health_panel.set_dataframe(findings.head(8))
            if isinstance(actions, pd.DataFrame) and not actions.empty:
                self.dashboard_home.action_panel.set_dataframe(actions.head(8))
            try:
                severities = findings.get("Severity", pd.Series(dtype=str)).astype(str).str.lower().tolist()
                critical = sum(1 for s in severities if "critical" in s or "error" in s or "fail" in s)
                warnings = sum(1 for s in severities if "warning" in s or "warn" in s)
                if critical:
                    self._set_badge("AI CRITICAL", "#dc2626")
                elif warnings:
                    self._set_badge("AI WARNING", "#d97706")
            except Exception:
                pass
    def train_ml_model(self) -> None:
        self._base_train_ml_model()
        if self.ml_model_bundle is not None:
            self.dashboard_home.ml_card.set("Trained", "Gradient-boosted trees", "ok")
    def load_ml_model(self) -> None:
        self._base_load_ml_model()
        if self.ml_model_bundle is not None:
            self.dashboard_home.ml_card.set("Loaded", "Model ready", "ok")
    def run_fault_study(self) -> None:
        self._base_run_fault_study()
        if self.fault_result is not None:
            self._set_badge("FAULT COMPLETE", "#7c3aed")
    def clear_results(self) -> None:
        self._base_clear_results()
        self.dashboard_home.clear()
        self._project_info()
        self._set_badge("READY", "#2563eb")
        self.tabs.set("Dashboard")
    def load_forecast_csv(self) -> None:
        try:
            path = filedialog.askopenfilename(
                title="Select historical total feeder load CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            history = load_history_csv(path)
            self.load_history_df = history
            self.load_forecast_bundle = None
            self.load_forecast_df = None
            self.forecast_risk_df = None
            self.forecast_risk_summary_df = None
            self.load_forecast_panel.set_history(history, path)
            self.status_label.configure(text="Status: historical load CSV loaded")
            self.tabs.set("Load Forecasting")
        except Exception as exc:
            self.load_forecast_panel.set_error(str(exc))
            messagebox.showerror("Load CSV failed", str(exc))
    def train_load_forecast_model(self) -> None:
        try:
            if self.load_history_df is None or self.load_history_df.empty:
                messagebox.showwarning("No historical load CSV", "Upload a historical load CSV first.")
                return
            cfg = self.load_forecast_panel.get_config()
            self.load_forecast_panel.set_badge("TRAINING", "#2563eb")
            self.status_label.configure(text="Status: training load forecast model...")
            self.update_idletasks()

            bundle = train_total_load_forecaster(
                self.load_history_df,
                test_size=cfg["test_size"],
                max_iter=cfg["max_iter"],
                learning_rate=cfg["learning_rate"],
            )
            self.load_forecast_bundle = bundle
            self.load_forecast_df = None
            self.forecast_risk_df = None
            self.forecast_risk_summary_df = None
            self.load_forecast_panel.set_metrics(bundle.metrics)
            self.status_label.configure(text="Status: load forecast model trained")
            self.tabs.set("Load Forecasting")
        except Exception as exc:
            self.load_forecast_panel.set_error(str(exc))
            messagebox.showerror("Training failed", str(exc))
            self.status_label.configure(text="Status: load forecast training failed")
    def forecast_total_feeder_load(self) -> None:
        try:
            if self.load_forecast_bundle is None:
                messagebox.showwarning("No trained forecast model", "Train the load forecast model first.")
                return
            cfg = self.load_forecast_panel.get_config()
            forecast_df = forecast_total_load(self.load_forecast_bundle, horizon_hours=cfg["horizon_hours"])
            self.load_forecast_df = forecast_df
            self.forecast_risk_df = None
            self.forecast_risk_summary_df = None
            self.load_forecast_panel.set_forecast(forecast_df)
            self.status_label.configure(text=f"Status: total feeder load forecast generated for {len(forecast_df)} hours")
            self.tabs.set("Load Forecasting")
        except Exception as exc:
            self.load_forecast_panel.set_error(str(exc))
            messagebox.showerror("Forecast failed", str(exc))
            self.status_label.configure(text="Status: load forecast failed")
    def _confirm_q_sign_if_needed(self, project: PowerSystemProject) -> bool:
        """Explicit user confirmation for capacitive-vs-inductive history sign mismatch."""
        history = self.load_forecast_bundle.history
        if "total_q_mvar" not in history.columns:
            return False  # P-only CSV: reference project Q/P is used in PF stage.
        good = history[["total_p_mw", "total_q_mvar"]].dropna()
        good = good.loc[good["total_p_mw"].abs() > 1e-9]
        if good.empty:
            return False
        ratio = float((good["total_q_mvar"] / good["total_p_mw"]).median())
        baseline_q = sum(float(b.q_load_pu) for b in project.buses if str(b.bus_type).lower() != "slack")
        if abs(ratio) > 1e-9 and abs(baseline_q) > 1e-9 and baseline_q * ratio < 0:
            return messagebox.askyesno(
                "Reactive-power sign mismatch",
                "The CSV Q/P sign opposes the reference network's Q sign. "
                "In the bundled 33-bus project, negative Q is inductive. "
                "Continue only if the opposite sign represents an intentional capacitive study."
            )
        return False

    def run_forecasted_power_flow(self) -> None:
        try:
            if self.load_forecast_bundle is None:
                messagebox.showwarning("No trained forecast model", "Train the load forecast model first.")
                return
            project = self.collect_project()
            ok, errors, warnings = validate_project_generic(project)
            power_flow_warnings = self._warnings_for_power_flow(warnings)
            self._show_validation_report(errors, power_flow_warnings)
            if not ok:
                messagebox.showerror("Project is invalid", "Fix validation errors before running forecasted power flow.")
                return

            cfg = self.load_forecast_panel.get_config()
            solver_method = self.solver_method_menu.get()
            self.load_forecast_panel.set_badge("RUNNING PF", "#7c3aed")
            self.status_label.configure(text="Status: running forecasted power-flow simulations...")
            self.update_idletasks()

            def progress(i: int, n: int) -> None:
                if i == 1 or i == n or i % max(1, n // 10) == 0:
                    self.status_label.configure(text=f"Status: forecasted power flow {i}/{n}")
                    self.load_forecast_panel.append_log(f"Forecasted power flow progress: {i}/{n}")
                    self.update_idletasks()

            forecast_df, risk_df = run_forecasted_power_flows(
                self.load_forecast_bundle,
                project,
                horizon_hours=cfg["horizon_hours"],
                solver_method=solver_method,
                max_iterations=cfg["pf_max_iter"],
                tolerance=cfg["pf_tolerance"],
                vmin_limit=cfg["vmin_limit"],
                vmax_limit=cfg["vmax_limit"],
                progress_callback=progress,
                allow_opposite_q_sign=self._confirm_q_sign_if_needed(project),
            )
            summary_df = summarize_forecast_risk(risk_df)
            self.load_forecast_df = forecast_df
            self.forecast_risk_df = risk_df
            self.forecast_risk_summary_df = summary_df
            self._last_risk_project_hash = hashlib.sha256(
                json.dumps(project.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest()
            self.load_forecast_panel.set_risk(forecast_df, risk_df, summary_df)

            # Surface forecasted risk on the main dashboard.
            try:
                self.dashboard_home.health_panel.set_dataframe(risk_df.head(10))
                self.dashboard_home.action_panel.set_dataframe(summary_df)
                violation_hours = int(risk_df["voltage_violation"].astype(bool).sum()) if "voltage_violation" in risk_df.columns else 0
                if violation_hours:
                    self._set_badge("FORECAST WARNING", "#d97706")
                else:
                    self._set_badge("FORECAST OK", "#15803d")
            except Exception:
                pass

            self.status_label.configure(text=f"Status: forecasted power flow completed for {len(risk_df)} hours")
            self.tabs.set("Load Forecasting")
        except Exception as exc:
            self.load_forecast_panel.set_error(str(exc))
            messagebox.showerror("Forecasted power flow failed", str(exc))
            self.status_label.configure(text="Status: forecasted power flow failed")
    def export_load_forecast_results(self) -> None:
        try:
            if self.load_forecast_df is None and self.forecast_risk_df is None:
                messagebox.showwarning("No forecast results", "Generate a forecast or forecasted power-flow risk result first.")
                return
            path = filedialog.asksaveasfilename(
                title="Export load forecast results",
                defaultextension=".xlsx",
                filetypes=[("Excel workbook", "*.xlsx"), ("CSV files", "*.csv")],
            )
            if not path:
                return
            current_project = self.collect_project()
            project_hash = hashlib.sha256(json.dumps(current_project.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
            if (self.forecast_risk_df is not None and
                    getattr(self, "_last_risk_project_hash", None) != project_hash):
                raise ValueError("Project changed after forecasted power flow. Rerun the study before exporting risk results.")
            metadata = {
                "Project": current_project.settings.name,
                "Project SHA256": project_hash,
                "Reference base MVA": float(current_project.settings.base_power_mva),
                "Solver selection": self.solver_method_menu.get(),
                "Thermal ratings": "Verified" if all(b.ampacity_verified for b in current_project.branches) else "Not evaluated",
                "Q sign convention": "Signed project injection: negative Q is inductive in bundled case33bw",
                "Software": "Python-Based Power System Simulation and Analysis Platform",
                "Export timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            if path.lower().endswith(".csv"):
                if self.forecast_risk_df is not None and not self.forecast_risk_df.empty:
                    self.forecast_risk_df.to_csv(path, index=False)
                else:
                    self.load_forecast_df.to_csv(path, index=False)
                Path(path).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    pd.DataFrame([{"Field": k, "Value": v} for k,v in metadata.items()]).to_excel(
                        writer, sheet_name="Study Metadata", index=False)
                    if self.load_history_df is not None:
                        self.load_history_df.to_excel(writer, sheet_name="Historical Load", index=False)
                    if self.load_forecast_bundle is not None:
                        self.load_forecast_bundle.metrics.to_excel(writer, sheet_name="Model Metrics", index=False)
                    if self.load_forecast_df is not None:
                        self.load_forecast_df.to_excel(writer, sheet_name="Total Forecast", index=False)
                    if self.forecast_risk_df is not None:
                        self.forecast_risk_df.to_excel(writer, sheet_name="Forecasted PF Risk", index=False)
                    if self.forecast_risk_summary_df is not None:
                        self.forecast_risk_summary_df.to_excel(writer, sheet_name="Risk Summary", index=False)
            self.load_forecast_panel.append_log(f"Exported forecast results to: {path}")
            self.status_label.configure(text=f"Status: exported forecast results to {path}")
        except Exception as exc:
            self.load_forecast_panel.set_error(str(exc))
            messagebox.showerror("Export failed", str(exc))
    def _install_load_forecasting_tab(self) -> None:
        self.tabs.add("Load Forecasting")
        self.tabs.tab("Load Forecasting").grid_columnconfigure(0, weight=1)
        self.tabs.tab("Load Forecasting").grid_rowconfigure(0, weight=1)

        self.load_forecast_panel = LoadForecastingPanel(
            self.tabs.tab("Load Forecasting"),
            app=self,
            load_csv_callback=self.load_forecast_csv,
            train_callback=self.train_load_forecast_model,
            forecast_callback=self.forecast_total_feeder_load,
            run_pf_callback=self.run_forecasted_power_flow,
            export_callback=self.export_load_forecast_results,
        )
        self.load_forecast_panel.grid(row=0, column=0, sticky="nsew")

        try:
            ctk.CTkButton(
                self.sidebar,
                text="Load Forecasting",
                command=lambda: self.tabs.set("Load Forecasting"),
                height=34,
                fg_color="#0891b2",
            ).grid(row=7, column=0, sticky="ew", padx=18, pady=(4, 12))
        except Exception:
            pass


if __name__ == "__main__":
    app = PowerSystemSimulatorApp()
    app.mainloop()
