"""
project_model.py

Data model for a user-entered power-system project.

This file is intentionally independent from the simulator backend.  It stores
network data in plain Python dataclasses/dictionaries so a desktop UI can let
non-programming engineers create, validate, save, and load projects.

Version 1 scope
---------------
- Power-flow studies with buses and generic branch elements.
- Branches can represent transmission lines or transformers if the user enters
  equivalent per-unit R, X, and B values.
- Optional sequence source data is stored for the next fault-study UI stage.

Backend compatibility note
--------------------------
The uploaded Newton-Raphson backend currently assumes sequential numeric bus
IDs.  For this reason, validation requires bus IDs: 1, 2, 3, ... N.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ProjectSettings:
    name: str = "Untitled Project"
    base_power_mva: float = 100.0
    frequency_hz: float = 60.0
    default_base_current_a: float = 1.0  # Legacy field: physical I_base is derived from MVA and bus kV.
    default_ampacity_pu: float = 10.0
    notes: str = ""


@dataclass
class BusRecord:
    bus_id: str
    bus_type: str = "PQ"  # Slack, PQ, PV
    voltage_pu: float = 1.0
    angle_deg: float = 0.0
    p_load_pu: float = 0.0
    q_load_pu: float = 0.0
    p_gen_pu: float = 0.0
    base_kv: float = 230.0


@dataclass
class BranchRecord:
    name: str
    from_bus: str
    to_bus: str
    r_pu: float
    x_pu: float
    b_pu: float = 0.0
    ampacity_pu: float = 10.0
    element_type: str = "Line"  # Line or Transformer-equivalent
    ampacity_verified: bool = False  # Explicitly established rating; False until the branch rating is independently verified.


@dataclass
class SequenceSourceRecord:
    name: str
    bus: str
    r1_pu: float = 0.0
    x1_pu: float = 0.1
    r2_pu: float = 0.0
    x2_pu: float = 0.1
    r0_pu: float = 0.0
    x0_pu: float = 0.1


@dataclass
class PowerSystemProject:
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    buses: List[BusRecord] = field(default_factory=list)
    branches: List[BranchRecord] = field(default_factory=list)
    sequence_sources: List[SequenceSourceRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PowerSystemProject":
        settings = ProjectSettings(**data.get("settings", {}))
        buses = [BusRecord(**row) for row in data.get("buses", [])]
        branches = [BranchRecord(**row) for row in data.get("branches", [])]
        sequence_sources = [SequenceSourceRecord(**row) for row in data.get("sequence_sources", [])]
        return cls(settings=settings, buses=buses, branches=branches, sequence_sources=sequence_sources)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "PowerSystemProject":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Return: (is_valid, errors, warnings).

        These checks are intentionally strict because the current uploaded
        backend is not yet a fully generic commercial-grade power-flow engine.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not self.buses:
            errors.append("At least one bus is required.")
            return False, errors, warnings

        bus_ids = [str(b.bus_id).strip() for b in self.buses]
        if len(bus_ids) != len(set(bus_ids)):
            errors.append("Duplicate bus IDs are not allowed.")

        # Current backend assumes bus IDs are numeric strings and sequential.
        try:
            bus_nums = sorted(int(x) for x in bus_ids)
            expected = list(range(1, len(bus_nums) + 1))
            if bus_nums != expected:
                errors.append(f"Bus IDs must be sequential numeric IDs: {expected}.")
        except ValueError:
            errors.append("Bus IDs must be numeric: 1, 2, 3, ...")
            bus_nums = []

        type_counts = {"slack": 0, "pq": 0, "pv": 0}
        for bus in self.buses:
            t = str(bus.bus_type).strip().lower()
            if t not in type_counts:
                errors.append(f"Bus {bus.bus_id}: bus_type must be Slack, PQ, or PV.")
            else:
                type_counts[t] += 1

        if type_counts["slack"] != 1:
            errors.append("Exactly one Slack bus is required.")

        # Compatibility constraints imposed by the current PowerFlow implementation.
        slack_buses = [b for b in self.buses if str(b.bus_type).strip().lower() == "slack"]
        if slack_buses and str(slack_buses[0].bus_id).strip() != "1":
            errors.append("For the current backend version, the Slack bus must be Bus 1.")

        pv_buses = [b for b in self.buses if str(b.bus_type).strip().lower() == "pv"]
        if len(pv_buses) != 1:
            errors.append("For the current backend version, exactly one PV bus is required.")
        elif bus_nums and int(pv_buses[0].bus_id) != max(bus_nums):
            errors.append("For the current backend version, the PV bus must be the highest-numbered bus.")

        if not self.branches:
            errors.append("At least one branch/line is required.")

        valid_bus_set = set(bus_ids)
        for branch in self.branches:
            if str(branch.from_bus) not in valid_bus_set:
                errors.append(f"Branch {branch.name}: from_bus {branch.from_bus} does not exist.")
            if str(branch.to_bus) not in valid_bus_set:
                errors.append(f"Branch {branch.name}: to_bus {branch.to_bus} does not exist.")
            if str(branch.from_bus) == str(branch.to_bus):
                errors.append(f"Branch {branch.name}: from_bus and to_bus cannot be the same.")
            if abs(float(branch.x_pu)) < 1e-12 and abs(float(branch.r_pu)) < 1e-12:
                errors.append(f"Branch {branch.name}: R and X cannot both be zero.")
            if float(branch.ampacity_pu) <= 0:
                errors.append(f"Branch {branch.name}: ampacity_pu must be positive.")

        for src in self.sequence_sources:
            if str(src.bus) not in valid_bus_set:
                errors.append(f"Sequence source {src.name}: bus {src.bus} does not exist.")
            if abs(float(src.x1_pu)) < 1e-12 and abs(float(src.r1_pu)) < 1e-12:
                warnings.append(f"Sequence source {src.name}: positive-sequence impedance is nearly zero.")

        if not self.sequence_sources:
            warnings.append("No sequence sources are defined. Fault study may not be meaningful yet.")

        return len(errors) == 0, errors, warnings


# -----------------------------------------------------------------------------
# Default 7-bus project, converted to user-editable data
# -----------------------------------------------------------------------------

def _partridge_line_per_mile_pu() -> Tuple[float, float, float]:
    """Return approximate per-mile R, X, B pu from the original 7-bus case."""
    rd = (0.642 / 2) * (1 / 12)  # ft
    bundle_spacing = 1.5  # ft
    ds = 0.0217  # ft
    rc = 0.385  # ohm/mile per conductor
    num_conductors = 2
    xa, ya, xb, yb, xc, yc = 0, 0, 19.5, 0, 39, 0

    dab = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
    dbc = ((xc - xb) ** 2 + (yc - yb) ** 2) ** 0.5
    dca = ((xa - xc) ** 2 + (ya - yc) ** 2) ** 0.5
    deq = (dab * dbc * dca) ** (1 / 3)

    dsl = math.sqrt(ds * bundle_spacing)
    dsc = math.sqrt(rd * bundle_spacing)

    rp = rc / num_conductors
    lp = 2e-7 * math.log(deq / dsl)
    xp = 2 * math.pi * 60 * lp * 1609
    cp = (2 * math.pi * 8.854e-12) / math.log(deq / dsc)
    bp = 2 * math.pi * 60 * cp * 1609

    base_z = (230e3 ** 2) / 100e6
    base_b = 1 / base_z
    return rp / base_z, xp / base_z, bp / base_b


def _transformer_rx_pu(z_t: float, x_r: float, syst_base_mva: float, rating_mva: float) -> Tuple[float, float]:
    theta = math.atan(x_r)
    r_old = z_t * math.cos(theta)
    x_old = z_t * math.sin(theta)
    scale = syst_base_mva / rating_mva
    return r_old * scale, x_old * scale


def default_7_bus_project() -> PowerSystemProject:
    """Return the original 7-bus case as editable project data."""
    r_per_mile, x_per_mile, b_per_mile = _partridge_line_per_mile_pu()
    line_lengths = {
        "Line1": ("2", "3", 25.0),
        "Line2": ("2", "4", 10.0),
        "Line3": ("3", "5", 20.0),
        "Line4": ("4", "6", 20.0),
        "Line5": ("5", "6", 10.0),
        "Line6": ("4", "5", 35.0),
    }

    buses = [
        BusRecord("1", "Slack", 1.0, 0.0, 0.0, 0.0, 0.0, 20.0),
        BusRecord("2", "PQ", 1.0, 0.0, 0.0, 0.0, 0.0, 230.0),
        BusRecord("3", "PQ", 1.0, 0.0, 1.1, -0.50, 0.0, 230.0),
        BusRecord("4", "PQ", 1.0, 0.0, 1.0, -0.70, 0.0, 230.0),
        BusRecord("5", "PQ", 1.0, 0.0, 1.0, -0.65, 0.0, 230.0),
        BusRecord("6", "PQ", 1.0, 0.0, 0.0, 0.0, 0.0, 230.0),
        BusRecord("7", "PV", 1.0, 0.0, 0.0, 0.0, 2.0, 18.0),
    ]

    branches: List[BranchRecord] = []
    t1_r, t1_x = _transformer_rx_pu(0.085, 10, 100, 125)
    branches.append(BranchRecord("Transformer1", "1", "2", t1_r, t1_x, 0.0, 10.0, "Transformer"))

    for name, (from_bus, to_bus, length) in line_lengths.items():
        branches.append(
            BranchRecord(
                name=name,
                from_bus=from_bus,
                to_bus=to_bus,
                r_pu=r_per_mile * length,
                x_pu=x_per_mile * length,
                b_pu=b_per_mile * length,
                ampacity_pu=10.0,
                element_type="Line",
            )
        )

    t2_r, t2_x = _transformer_rx_pu(0.105, 12, 100, 200)
    branches.append(BranchRecord("Transformer2", "6", "7", t2_r, t2_x, 0.0, 10.0, "Transformer"))

    # Sequence source shunts are included for future fault-study support.
    x1 = 0.12 * 100 / 90
    x2 = 0.14 * 100 / 90
    x0_g1 = 0.05 * 100 / 100
    grounding_base_g2 = (18e3 ** 2) / 100e6
    r0_g2 = 3 * 1.0 / grounding_base_g2
    x0_g2 = 0.05 * 100 / 100
    sequence_sources = [
        SequenceSourceRecord("Gen1", "1", 0.0, x1, 0.0, x2, 0.0, x0_g1),
        SequenceSourceRecord("Gen2", "7", 0.0, x1, 0.0, x2, r0_g2, x0_g2),
    ]

    return PowerSystemProject(
        settings=ProjectSettings(
            name="Original 7-Bus Example",
            base_power_mva=100.0,
            frequency_hz=60.0,
            default_base_current_a=1.0,
            default_ampacity_pu=10.0,
            notes="Converted from the original hardcoded Main.py case.",
        ),
        buses=buses,
        branches=branches,
        sequence_sources=sequence_sources,
    )


if __name__ == "__main__":
    project = default_7_bus_project()
    ok, errors, warnings = project.validate()
    print({"valid": ok, "errors": errors, "warnings": warnings})
