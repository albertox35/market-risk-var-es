"""Teoria de valores extremos: picos sobre umbral y distribucion de Pareto generalizada.

El problema de estimar un VaR al 99.9 % con mil observaciones es que hay **una**
observacion mas alla del cuantil. El cuantil empirico ahi no es un estimador, es
un dato suelto.

La teoria de valores extremos da la salida. El teorema de Pickands,
Balkema y de Haan establece que, para una clase muy amplia de distribuciones, los
excesos sobre un umbral alto convergen a una **Pareto generalizada**:

.. math::

    P(X - u \\leq y \\mid X > u) \\to
    1 - \\left(1 + \\frac{\\xi y}{\\beta}\\right)^{-1/\\xi}

Eso permite **extrapolar** mas alla del dato mas extremo observado, con una forma
funcional justificada en lugar de inventada.

El parametro de forma ``xi`` es el que importa: gobierna el grosor de la cola.

- ``xi > 0``: cola de ley de potencia (Frechet). Lo habitual en retornos
  financieros, con valores tipicos entre 0.1 y 0.4.
- ``xi = 0``: cola exponencial (Gumbel).
- ``xi < 0``: cola acotada (Weibull).

Y hay una consecuencia que suele pasarse por alto: si ``xi >= 0.5`` la varianza
no existe, y si ``xi >= 1`` **ni siquiera existe la media**, con lo que el
Expected Shortfall es infinito. Un ``xi`` estimado por encima de 0.5 no es un
numero mas: es una senial de que las medidas basadas en momentos no aplican.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto

__all__ = ["GPDFit", "fit_gpd_pot", "evt_var", "evt_expected_shortfall", "mean_excess_plot"]


@dataclass(frozen=True)
class GPDFit:
    """Ajuste de Pareto generalizada a los excesos sobre un umbral."""

    threshold: float
    shape: float
    """Parametro de forma ``xi``."""

    scale: float
    """Parametro de escala ``beta``."""

    n_exceedances: int
    n_total: int
    converged: bool

    @property
    def exceedance_rate(self) -> float:
        """Proporcion de observaciones por encima del umbral."""
        return self.n_exceedances / self.n_total

    @property
    def has_finite_variance(self) -> bool:
        """``True`` si ``xi < 0.5``, condicion para que exista la varianza."""
        return self.shape < 0.5

    @property
    def has_finite_mean(self) -> bool:
        """``True`` si ``xi < 1``, condicion para que exista la media.

        Si es ``False``, el Expected Shortfall es **infinito** y no tiene sentido
        reportarlo como numero.
        """
        return self.shape < 1.0

    @property
    def tail_index(self) -> float:
        """Indice de cola ``1/xi``: el momento de orden ``1/xi`` ya no existe."""
        return float(1.0 / self.shape) if self.shape > 0.0 else float("inf")

    def __str__(self) -> str:
        return (
            f"GPD(xi={self.shape:.4f}, beta={self.scale:.6f}) sobre umbral "
            f"{self.threshold:.6f} con {self.n_exceedances}/{self.n_total} excesos "
            f"({self.exceedance_rate:.2%})"
        )


def fit_gpd_pot(
    losses: np.ndarray,
    threshold: float | None = None,
    threshold_quantile: float = 0.95,
) -> GPDFit:
    """Ajusta una Pareto generalizada a los excesos sobre un umbral.

    **La eleccion del umbral es el compromiso central del metodo** y no tiene
    solucion automatica: un umbral alto hace valido el resultado asintotico pero
    deja pocos datos (varianza alta); uno bajo da mas datos pero el teorema
    limite ya no aplica (sesgo). La practica es el percentil 90-95 y comprobar la
    estabilidad del ajuste frente a ese umbral, para lo que sirve
    :func:`mean_excess_plot`.

    Parameters
    ----------
    losses:
        Muestra de perdidas, positivas.
    threshold:
        Umbral explicito. Si es ``None`` se toma el percentil indicado.
    threshold_quantile:
        Percentil que define el umbral.

    Returns
    -------
    GPDFit
    """
    arr = np.asarray(losses, dtype=float).ravel()
    if arr.size < 100:
        raise ValueError(f"se requieren al menos 100 observaciones, hay {arr.size}")
    u = float(np.quantile(arr, threshold_quantile)) if threshold is None else float(threshold)
    excesos = arr[arr > u] - u
    if excesos.size < 20:
        raise ValueError(
            f"solo {excesos.size} excesos sobre el umbral {u:.6f}; se requieren al menos 20. "
            "Bajar threshold_quantile o ampliar la muestra."
        )

    def neg_ll(params: np.ndarray) -> float:
        xi, beta = params
        if beta <= 0.0:
            return 1e10
        if xi < 0.0 and np.max(excesos) >= -beta / xi:
            # Con xi negativo el soporte esta acotado; si algun exceso lo supera
            # la verosimilitud es cero.
            return 1e10
        return float(-np.sum(genpareto.logpdf(excesos, c=xi, scale=beta)))

    media = float(np.mean(excesos))
    salida = minimize(
        neg_ll,
        np.array([0.1, media]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
    )
    xi, beta = (float(v) for v in salida.x)
    return GPDFit(
        threshold=u,
        shape=xi,
        scale=beta,
        n_exceedances=int(excesos.size),
        n_total=int(arr.size),
        converged=bool(salida.success),
    )


def evt_var(fit: GPDFit, confidence: float = 0.999) -> float:
    """VaR extrapolado por teoria de valores extremos.

    .. math::

        VaR_\\alpha = u + \\frac{\\beta}{\\xi}
            \\left[\\left(\\frac{n(1-\\alpha)}{N_u}\\right)^{-\\xi} - 1\\right]

    Esta es la razon de usar EVT: permite estimar cuantiles **mas alla** del dato
    mas extremo de la muestra con una forma funcional justificada.

    Raises
    ------
    ValueError
        Si el nivel pedido queda por debajo del umbral, caso en el que la
        extrapolacion no aporta nada y hay que usar el cuantil empirico.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("el nivel de confianza debe estar en (0, 1)")
    if confidence <= 1.0 - fit.exceedance_rate:
        raise ValueError(
            f"el nivel {confidence:.4%} esta por debajo del umbral, que corresponde al "
            f"{1.0 - fit.exceedance_rate:.4%}. La extrapolacion EVT no aplica ahi: usar "
            "el cuantil empirico."
        )
    razon = fit.n_total * (1.0 - confidence) / fit.n_exceedances
    if abs(fit.shape) < 1e-8:
        # Limite exponencial.
        return float(fit.threshold - fit.scale * np.log(razon))
    return float(fit.threshold + (fit.scale / fit.shape) * (razon ** (-fit.shape) - 1.0))


def evt_expected_shortfall(fit: GPDFit, confidence: float = 0.999) -> float:
    """Expected Shortfall extrapolado por EVT.

    .. math::

        ES_\\alpha = \\frac{VaR_\\alpha}{1-\\xi}
            + \\frac{\\beta - \\xi u}{1-\\xi}

    Raises
    ------
    ValueError
        Si ``xi >= 1`` el ES es infinito: la media de la cola no existe. Se
        lanza una excepcion en lugar de devolver un numero sin sentido.
    """
    if not fit.has_finite_mean:
        raise ValueError(
            f"el parametro de forma estimado es xi = {fit.shape:.4f} >= 1, asi que la "
            "media de la cola no existe y el Expected Shortfall es infinito. No es un "
            "problema numerico: la distribucion ajustada no tiene primer momento."
        )
    var = evt_var(fit, confidence)
    return float(var / (1.0 - fit.shape) + (fit.scale - fit.shape * fit.threshold) / (1.0 - fit.shape))


def mean_excess_plot(
    losses: np.ndarray, n_thresholds: int = 40, max_quantile: float = 0.995
) -> tuple[np.ndarray, np.ndarray]:
    """Funcion de exceso medio, para elegir el umbral.

    Para una Pareto generalizada el exceso medio es **lineal** en el umbral:

    .. math:: E[X - u \\mid X > u] = \\frac{\\beta + \\xi u}{1 - \\xi}

    Asi que el umbral adecuado es aquel a partir del cual la grafica se vuelve
    recta. Es el diagnostico estandar, y evita elegir el umbral a ojo.

    Returns
    -------
    umbrales, exceso_medio
    """
    arr = np.asarray(losses, dtype=float).ravel()
    umbrales = np.quantile(arr, np.linspace(0.5, max_quantile, n_thresholds))
    medias = np.array(
        [np.mean(arr[arr > u] - u) if np.any(arr > u) else np.nan for u in umbrales]
    )
    return umbrales, medias
