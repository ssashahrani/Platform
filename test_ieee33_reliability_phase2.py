"""
test_ieee33_reliability_phase2.py

Console test for the Phase-2 analytical reliability adapter.
Run after placing ieee33_project_template.json and the reliability adapter files
in the same folder.
"""

from project_model import PowerSystemProject
from adapter_reliability_phase2 import run_switching_protection_reliability_project


project = PowerSystemProject.load_json("ieee33_project_template.json")

result = run_switching_protection_reliability_project(
    project,
    default_branch_failure_rate_per_year=0.10,
    default_branch_repair_time_hours=5.0,
    default_total_customers=3300,
    protection_strategy="Root breaker only",
    assume_branch_sectionalizing_switches=True,
    default_isolation_time_hours=0.25,
    default_healthy_section_restoration_time_hours=0.75,
    sustained_interruption_threshold_hours=5.0 / 60.0,
    assume_tie_transfer_all_downstream=False,
    validate_project=False,
)

print("=== System Indices ===")
print(result.system_indices_table.to_string(index=False))

print("\n=== Top 10 Branch/Event Contributions ===")
cols = [
    "Faulted_Branch",
    "Protection_Device",
    "Tripped_Customers",
    "Downstream_Customers",
    "Customer_Outage_Hours_per_year",
    "ENS_MWh_per_year",
]
print(result.event_impact_table[cols].head(10).to_string(index=False))
