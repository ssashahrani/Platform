"""Source-level smoke and regression checks; not a PowerWorld or Windows GUI certification."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from project_model import PowerSystemProject, default_7_bus_project
from adapter_project_fbs import run_power_flow_project
from load_forecasting import (
    apply_total_load_forecast_to_project, load_history_csv,
    train_total_load_forecaster,run_forecasted_power_flows, summarize_forecast_risk,
)
from ml_engine import generate_training_dataset, train_gradient_boosting_models, predict_current_case
HERE=Path(__file__).resolve().parent

class PhysicsAndForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project=PowerSystemProject.load_json(HERE/'ieee33_bus_fbs_test.json')
        cls.history=load_history_csv(str(HERE/'sample_total_feeder_load_history.csv'))

    def test_base_fbs_and_physical_current(self):
        r=run_power_flow_project(self.project,solver_method='Forward/Backward Sweep',max_iterations=100,tolerance=1e-6)
        self.assertTrue(r.converged)
        self.assertAlmostEqual(float(r.bus_table.Voltage_pu.min()),0.91309,places=4)
        self.assertAlmostEqual(float(r.summary['total_loss_MVA_real']),0.202677,places=4)
        c=r.line_current_table.iloc[0]
        ib=10e6/(math.sqrt(3)*12.66e3)
        self.assertAlmostEqual(c.Current_base_A,ib,places=5)
        self.assertAlmostEqual(c.Line_current_A,c.I_pu_mag*ib,places=4)
        self.assertFalse(bool(c.Ampacity_verified))

    def test_unverified_thermal_is_not_evaluated(self):
        b=train_total_load_forecaster(self.history,max_iter=15)
        f,r=run_forecasted_power_flows(b,self.project,horizon_hours=2)
        self.assertEqual(r['thermal_status'].tolist(),['Not evaluated']*2)
        self.assertTrue(r['max_loading_percent'].isna().all())
        self.assertTrue(r['overload'].isna().all())
        summary=summarize_forecast_risk(r)
        self.assertIn('Not evaluated',str(summary.loc[summary.Metric.eq('Overload hours'),'Value'].iloc[0]))

    def test_missing_historical_q_uses_reference_ratio(self):
        h=self.history.drop(columns=['total_q_mvar'])
        b=train_total_load_forecaster(h,max_iter=15)
        f,r=run_forecasted_power_flows(b,self.project,horizon_hours=1)
        ratio=sum(z.q_load_pu for z in self.project.buses)/sum(z.p_load_pu for z in self.project.buses)
        self.assertAlmostEqual(float(f.iloc[0].forecast_total_q_mvar),float(f.iloc[0].forecast_total_p_mw)*ratio,places=6)
        self.assertEqual(f.iloc[0].Q_forecast_method,'project_baseline_q_over_p')

    def test_q_sign_mismatch_is_rejected_by_default(self):
        h=self.history.copy()
        h['total_q_mvar']=0.6*h['total_p_mw']
        b=train_total_load_forecaster(h,max_iter=15)
        with self.assertRaisesRegex(ValueError,'sign differs'):
            run_forecasted_power_flows(b,self.project,horizon_hours=1)

    def test_zero_reference_q_nonzero_target_fails(self):
        p=copy.deepcopy(self.project)
        for bus in p.buses:bus.q_load_pu=0.0
        with self.assertRaisesRegex(ValueError,'Nonzero forecast Q'):
            apply_total_load_forecast_to_project(p,4.0,-2.0)

    def test_history_requires_hourly_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.csv'
            h=self.history.iloc[:210].copy()
            h.loc[5,'timestamp']=h.loc[4,'timestamp']
            h.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_history_csv(str(path))
            h=self.history.iloc[:210].copy()
            h.loc[5,'timestamp']=h.loc[5,'timestamp']+pd.Timedelta(minutes=30)
            h.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'consecutive hourly'):
                load_history_csv(str(path))
            self.history.iloc[:197].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'198'):
                load_history_csv(str(path))

    def test_ml_overload_is_not_trained_without_ratings(self):
        dataset,summary=generate_training_dataset(self.project,n_scenarios=35,solver_method='Forward/Backward Sweep')
        self.assertTrue(dataset.target_overload.isna().all())
        bundle=train_gradient_boosting_models(dataset,scenario_summary=summary)
        self.assertNotIn('overload_classifier',bundle.models)
        table,_=predict_current_case(bundle,self.project)
        self.assertEqual(table.loc[table.Item.eq('Predicted overload risk'),'Value'].iloc[0],'Not evaluated')

    def test_legacy_project_constructor_keeps_branch_type(self):
        seven=default_7_bus_project()
        self.assertEqual(seven.branches[0].element_type,'Transformer')
        self.assertFalse(seven.branches[0].ampacity_verified)

if __name__=='__main__':unittest.main(verbosity=2)
