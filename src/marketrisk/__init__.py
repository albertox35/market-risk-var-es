"""Motor de riesgo de mercado: VaR y Expected Shortfall con backtesting.

Lo que distingue este repositorio de un calculador de cuantiles es la **suite de
backtesting**. Un numero de VaR sin backtest es un numero sin respaldo.

- :mod:`marketrisk.garch` GARCH(1,1) y DCC-GARCH por cuasi-maxima verosimilitud.
- :mod:`marketrisk.fhs` simulacion historica filtrada.
- :mod:`marketrisk.evt` picos sobre umbral y Pareto generalizada para la cola.
- :mod:`marketrisk.measures` VaR y ES, con las perdidas como cantidades positivas.
- :mod:`marketrisk.backtesting` Kupiec, Christoffersen y Acerbi-Szekely.
- :mod:`marketrisk.attribution` contribuciones al riesgo por asignacion de Euler.

Marco regulatorio: Basilea III / FRTB sustituyo el VaR al 99 % a 10 dias por el
**Expected Shortfall al 97.5 %** con horizontes de liquidez escalados. Los
valores por omision del modulo reflejan ese marco, no el anterior.
"""

from __future__ import annotations

from .attribution import RiskContributions, diversification_benefit, euler_contributions
from .backtesting import (
    BacktestResult,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    run_all_backtests,
    traffic_light,
)
from .evt import GPDFit, evt_expected_shortfall, evt_var, fit_gpd_pot, mean_excess_plot
from .fhs import (
    FHSResult,
    RollingForecast,
    filtered_historical_simulation,
    rolling_fhs_backtest,
)
from .garch import DCCResult, GARCHResult, fit_dcc, fit_garch, simulate_garch
from .measures import RiskMeasures, expected_shortfall, risk_measures, scale_to_horizon, value_at_risk

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "DCCResult",
    "FHSResult",
    "GARCHResult",
    "GPDFit",
    "RiskContributions",
    "RiskMeasures",
    "RollingForecast",
    "__version__",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "diversification_benefit",
    "euler_contributions",
    "evt_expected_shortfall",
    "evt_var",
    "expected_shortfall",
    "filtered_historical_simulation",
    "fit_dcc",
    "fit_garch",
    "fit_gpd_pot",
    "kupiec_pof",
    "mean_excess_plot",
    "risk_measures",
    "rolling_fhs_backtest",
    "run_all_backtests",
    "scale_to_horizon",
    "simulate_garch",
    "traffic_light",
    "value_at_risk",
]
