"""Simulacion historica filtrada (Barone-Adesi y Giannopoulos).

La simulacion historica simple remuestrea retornos pasados directamente. Tiene un
defecto grave: **mezcla regimenes de volatilidad**. Si la ventana incluye 2008 y
2017, los retornos de ambos periodos entran con el mismo peso, asi que el VaR de
un dia tranquilo queda inflado y el de un dia turbulento, subestimado. Y el
modelo reacciona con retraso: solo cuando un dia extremo entra o sale de la
ventana.

La simulacion historica **filtrada** resuelve esto en tres pasos:

1. **Filtrar**: ajustar un GARCH y dividir cada retorno por su volatilidad
   condicional. Los residuos resultantes si son aproximadamente independientes e
   identicamente distribuidos.
2. **Remuestrear** esos residuos, que conservan la forma de la cola empirica
   --asimetria y curtosis incluidas-- sin imponer ninguna distribucion.
3. **Reescalar** por la volatilidad pronosticada para **hoy**.

El resultado capta colas gruesas sin suponer normalidad y reacciona de inmediato
a los cambios de regimen, que son las dos cosas que la simulacion historica
simple no hace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .garch import GARCHResult, fit_garch
from .measures import RiskMeasures, risk_measures

__all__ = [
    "FHSResult",
    "RollingForecast",
    "filtered_historical_simulation",
    "rolling_fhs_backtest",
]


@dataclass(frozen=True)
class FHSResult:
    """Resultado de una simulacion historica filtrada."""

    simulated_losses: np.ndarray
    """Perdidas simuladas al horizonte, positivas."""

    garch: GARCHResult
    horizon: int
    forecast_volatility: float
    """Volatilidad pronosticada para el primer dia del horizonte."""

    def measures(self, confidence: float = 0.975) -> RiskMeasures:
        """VaR y ES de la distribucion simulada."""
        return risk_measures(self.simulated_losses, confidence)

    @property
    def n_simulations(self) -> int:
        """Numero de trayectorias simuladas."""
        return int(self.simulated_losses.size)


def filtered_historical_simulation(
    returns: np.ndarray,
    horizon: int = 1,
    n_simulations: int = 50_000,
    seed: int = 20260101,
    garch: GARCHResult | None = None,
) -> FHSResult:
    """Distribucion de perdidas por simulacion historica filtrada.

    Para horizontes mayores que un dia la trayectoria se simula **dia a dia**
    propagando la recursion GARCH, no escalando por la raiz del tiempo. Esa es la
    diferencia que importa: la regla de la raiz supone independencia, y si la
    volatilidad de hoy esta lejos de su nivel de largo plazo el escalado es
    sesgado en una direccion conocida.

    Parameters
    ----------
    returns:
        Serie de retornos historicos.
    horizon:
        Horizonte en dias.
    n_simulations:
        Numero de trayectorias.
    seed:
        Semilla, para reproducibilidad.
    garch:
        Ajuste GARCH previo. Si es ``None`` se estima.

    Returns
    -------
    FHSResult
    """
    r = np.asarray(returns, dtype=float).ravel()
    if horizon < 1:
        raise ValueError("el horizonte debe ser al menos 1")
    if n_simulations < 100:
        raise ValueError("se requieren al menos 100 simulaciones")

    ajuste = garch if garch is not None else fit_garch(r)
    z = ajuste.standardized_residuals
    # Centrar y estandarizar los residuos: si su varianza muestral no es 1, el
    # reescalado introduce un sesgo multiplicativo en todo el VaR.
    z = (z - z.mean()) / z.std(ddof=1)

    rng = np.random.default_rng(seed)
    var_inicial = float(ajuste.forecast_variance(1)[0])

    acumulado = np.zeros(n_simulations)
    var_actual = np.full(n_simulations, var_inicial)
    for _ in range(horizon):
        innovaciones = rng.choice(z, size=n_simulations, replace=True)
        retorno = np.sqrt(var_actual) * innovaciones
        acumulado += retorno
        var_actual = ajuste.omega + ajuste.alpha * retorno**2 + ajuste.beta * var_actual

    # Convencion del modulo: perdidas positivas.
    return FHSResult(
        simulated_losses=-acumulado,
        garch=ajuste,
        horizon=horizon,
        forecast_volatility=float(np.sqrt(var_inicial)),
    )


@dataclass(frozen=True)
class RollingForecast:
    """Pronosticos fuera de muestra, alineados con las perdidas realizadas.

    Se devuelven **dos** vectores de VaR, y no por redundancia: los backtests de
    Acerbi-Szekely definen las excepciones con el VaR del **mismo** nivel que el
    ES, mientras Kupiec y Christoffersen usan el nivel regulatorio del VaR. Si se
    mezclan los niveles, Z1 y Z2 se contradicen y el diagnostico no significa
    nada, asi que la estructura entrega cada uno por separado para que no se
    puedan confundir.
    """

    realized_losses: np.ndarray
    var_forecast: np.ndarray
    """VaR al nivel ``confidence``; para Kupiec y Christoffersen."""

    es_var_forecast: np.ndarray
    """VaR al nivel ``es_confidence``; para los backtests de ES."""

    es_forecast: np.ndarray
    """ES al nivel ``es_confidence``."""

    confidence: float
    es_confidence: float

    @property
    def n_exceptions(self) -> int:
        """Excepciones del VaR al nivel ``confidence``."""
        return int(np.sum(self.realized_losses > self.var_forecast))

    @property
    def exception_rate(self) -> float:
        """Frecuencia observada de excepciones."""
        return self.n_exceptions / self.realized_losses.size

    def __len__(self) -> int:
        return int(self.realized_losses.size)


def rolling_fhs_backtest(
    returns: np.ndarray,
    window: int = 500,
    confidence: float = 0.99,
    es_confidence: float = 0.975,
    n_simulations: int = 5_000,
    seed: int = 20260101,
    refit_every: int = 25,
) -> RollingForecast:
    """Genera pronosticos de VaR y ES fuera de muestra con ventana movil.

    Es la unica forma honesta de backtestear: en cada fecha el modelo se estima
    **solo con informacion anterior** y se pronostica el dia siguiente. Estimar
    con toda la muestra y luego «backtestear» sobre ella produce resultados
    inservibles, y es el error mas comun del area.

    Parameters
    ----------
    returns:
        Serie completa de retornos.
    window:
        Tamanio de la ventana de estimacion.
    refit_every:
        Reestimar el GARCH cada tantos dias. Reestimar a diario es correcto pero
        costoso; el filtro de varianza se actualiza todos los dias de todas
        formas, que es lo que de verdad mueve el pronostico.

    Returns
    -------
    RollingForecast
        Con las perdidas realizadas y tres vectores de pronostico alineados: el
        elemento ``i`` corresponde al dia ``window + i``.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size <= window + 20:
        raise ValueError(
            f"la serie tiene {r.size} observaciones; se requieren mas de "
            f"{window + 20} para una ventana de {window}"
        )

    n_pronosticos = r.size - window
    var_pred = np.empty(n_pronosticos)
    es_var_pred = np.empty(n_pronosticos)
    es_pred = np.empty(n_pronosticos)
    realizadas = -r[window:]

    ajuste: GARCHResult | None = None
    for i in range(n_pronosticos):
        ventana = r[i : i + window]
        if ajuste is None or i % refit_every == 0:
            ajuste = fit_garch(ventana)
        else:
            # Reutilizar los parametros y solo refiltrar la ventana nueva.
            ajuste = _refilter(ventana, ajuste)
        fhs = filtered_historical_simulation(
            ventana, horizon=1, n_simulations=n_simulations, seed=seed + i, garch=ajuste
        )
        var_pred[i] = fhs.measures(confidence).var
        medidas_es = fhs.measures(es_confidence)
        es_var_pred[i] = medidas_es.var
        es_pred[i] = medidas_es.expected_shortfall

    return RollingForecast(
        realized_losses=realizadas,
        var_forecast=var_pred,
        es_var_forecast=es_var_pred,
        es_forecast=es_pred,
        confidence=confidence,
        es_confidence=es_confidence,
    )


def _refilter(returns: np.ndarray, previous: GARCHResult) -> GARCHResult:
    """Refiltra una ventana con parametros ya estimados.

    Mucho mas rapido que reoptimizar y suficiente entre reestimaciones: los
    parametros GARCH son estables en el tiempo, mientras que la varianza
    condicional cambia todos los dias.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r - r.mean()
    n = r.size
    var = np.empty(n)
    var[0] = float(np.var(r))
    for t in range(1, n):
        var[t] = previous.omega + previous.alpha * r[t - 1] ** 2 + previous.beta * var[t - 1]
    return GARCHResult(
        omega=previous.omega,
        alpha=previous.alpha,
        beta=previous.beta,
        conditional_variance=var,
        standardized_residuals=r / np.sqrt(var),
        log_likelihood=previous.log_likelihood,
        converged=previous.converged,
    )
