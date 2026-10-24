# Python-Based Power System Simulation and Analysis Platform
**KACST – Future Grids Lab | Training and engineering research prototype**

## Running on Windows

From the extracted folder containing `app_project.py`, run:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python test_regressions.py
python app_project.py
```

The desktop program expects the existing backend and KACST/Vision 2030 PNGs in the same folder. The `test_regressions.py` suite is a smoke/regression suite, **not** a PowerWorld acceptance test.

## Engineering conventions and safeguards

- The bundled `ieee33_bus_fbs_test.json` uses **positive P load and negative Q for inductive demand**. The Q field is a signed injection, not a positive consumption convention.
- The historical load-forecast CSV must contain 198 or more **consecutive hourly rows** with `timestamp` and positive `total_p_mw`. Optional `total_q_mvar` must use the project's signed convention. Opposite Q signs are blocked by default, with explicit user approval for a deliberate capacitive study.
- Without historical Q, forecasted power flow preserves the **reference network's aggregate signed Q/P**. This is an engineering assumption, not measured-Q forecasting.
- Physical currents in amperes are calculated using `I_base,A = S_base,MVA * 1000 / (sqrt(3) * V_base,kV)` at each bus. The `default_base_current_a` field is retained only for compatibility and is ignored for this conversion.
- Branch `ampacity_pu` is **not** a verified thermal rating unless the project contains `ampacity_verified: true` for that branch. Unverified branch ratings (including the 10 pu value in the supplied example) produce **Thermal status: Not evaluated**. The AI advisor, forecasting risk and ML surrogate now reflect this limitation.
- The thermal-status logic does not automatically verify that a user-entered rating is genuinely representative of a specific conductor/transformer. An engineer must verify those ratings independently.
- A single nominal IEEE 33-bus FBS case should give `Vmin ≈ 0.91309 pu at Bus 18`, `real losses ≈ 0.202677 MW` and `reactive losses ≈ 0.135141 MVAr`.
- Solver tolerances differ between the sidebar (typically 30, 1e-4) and the forecasted-PF panel (typically 100, 1e-6).

## Files and compatibility

`app_project.py` is the consolidated UI entry point. The numerical/ML modules remain separate. Old project JSON files load with `ampacity_verified=False` when the new field is absent. To confirm a branch thermal rating, set `ampacity_verified` to true **after** independently establishing the true per-unit rating.

The latest exporter writes project metadata to an Excel `Study Metadata` sheet or a JSON companion to CSV. It blocks export of previously calculated PF-risk results if the project has changed. The program has not been certified for operational decisions.

## Required external validation

PowerWorld comparisons must be performed on consistent network data, bases, models and solver settings. Forecast accuracy must be assessed on actual out-of-sample measured load via multiple rolling-origin 24-hour backtests. Windows GUI/DPI testing and full G1–G8 gate records remain outstanding.

Project repository: https://github.com/ssashahrani/Platform
