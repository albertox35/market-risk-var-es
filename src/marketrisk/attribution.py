"""Atribucion de riesgo por asignacion de Euler.

Un numero de riesgo agregado no sirve para gestionar: la pregunta operativa es
**que posicion lo esta causando**. La respuesta correcta no es el riesgo de cada
posicion por separado --eso ignora la diversificacion-- sino su **contribucion
marginal** al riesgo del portafolio.

La asignacion de Euler es la unica que suma exactamente al total, porque el ES es
homogeneo de grado 1 y por el teorema de Euler:

.. math::

    ES(w) = \\sum_i w_i \\frac{\\partial ES}{\\partial w_i}

Y para el ES la derivada tiene una forma notablemente simple:

.. math::

    \\frac{\\partial ES_\\alpha}{\\partial w_i}
        = E\\left[L_i \\mid L_{portafolio} \\geq VaR_\\alpha\\right]

es decir, **la perdida media de la posicion i en los escenarios donde el
portafolio sufre su cola**. Eso es directamente interpretable: dice quien pierde
cuando pierde el portafolio, que no es lo mismo que quien es volatil.

Nota tecnica importante: esta identidad vale para el ES porque es coherente y
diferenciable. Para el VaR la asignacion de Euler existe pero **no suma al
total** en general, porque el VaR no es subaditivo. Es otro argumento practico
para usar ES.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RiskContributions", "euler_contributions", "diversification_benefit"]


@dataclass(frozen=True)
class RiskContributions(object):
    """Descomposicion del riesgo del portafolio por posicion."""

    total: float
    contributions: np.ndarray
    """Contribucion absoluta de cada posicion; suma al total."""

    standalone: np.ndarray
    """Riesgo de cada posicion por separado, para comparar."""

    confidence: float
    n_scenarios_in_tail: int

    @property
    def percentage(self) -> np.ndarray:
        """Contribucion relativa, en fraccion del total."""
        return self.contributions / self.total if self.total else np.full_like(self.contributions, np.nan)

    @property
    def sum_of_standalone(self) -> float:
        """Suma de los riesgos individuales, sin diversificacion."""
        return float(np.sum(self.standalone))

    @property
    def diversification_benefit(self) -> float:
        """Reduccion de riesgo por diversificacion, en fraccion.

        ``1 - total / suma_individual``. Positivo cuando la cartera se beneficia
        de la diversificacion.
        """
        s = self.sum_of_standalone
        return float(1.0 - self.total / s) if s else float("nan")

    def sums_to_total(self, tol: float = 1e-9) -> bool:
        """``True`` si las contribuciones suman el total.

        Es la propiedad que distingue la asignacion de Euler de los repartos ad
        hoc, y se comprueba en lugar de suponerse.
        """
        return bool(abs(float(np.sum(self.contributions)) - self.total) < tol)


def euler_contributions(
    position_losses: np.ndarray,
    weights: np.ndarray | None = None,
    confidence: float = 0.975,
) -> RiskContributions:
    """Contribuciones al Expected Shortfall por asignacion de Euler.

    Parameters
    ----------
    position_losses:
        Matriz ``(n_escenarios, n_posiciones)`` de perdidas por posicion y
        escenario, **positivas**.
    weights:
        Pesos de las posiciones. Por omision todos valen 1, es decir que
        ``position_losses`` ya viene en unidades monetarias.
    confidence:
        Nivel del ES.

    Returns
    -------
    RiskContributions

    Notes
    -----
    El estimador es la media muestral de cada posicion **condicionada a los
    escenarios de cola del portafolio**. Con pocos escenarios en la cola el
    estimador es ruidoso, por eso se devuelve su conteo.
    """
    l = np.asarray(position_losses, dtype=float)
    if l.ndim != 2:
        raise ValueError("position_losses debe ser una matriz (n_escenarios, n_posiciones)")
    n_esc, n_pos = l.shape
    if n_esc < 100:
        raise ValueError(f"se requieren al menos 100 escenarios, hay {n_esc}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("el nivel de confianza debe estar en (0, 1)")

    w = np.ones(n_pos) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != n_pos:
        raise ValueError(f"weights debe tener {n_pos} elementos, tiene {w.size}")

    ponderadas = l * w
    portafolio = ponderadas.sum(axis=1)
    var = float(np.quantile(portafolio, confidence))
    cola = portafolio >= var
    n_cola = int(cola.sum())
    if n_cola == 0:
        raise ValueError(f"cola vacia al {confidence:.1%} con {n_esc} escenarios")

    # Contribucion de Euler: perdida media de cada posicion en los escenarios de
    # cola del PORTAFOLIO, no en su propia cola.
    contribuciones = ponderadas[cola].mean(axis=0)
    total = float(portafolio[cola].mean())

    # Riesgo individual: el ES de cada posicion en su propia cola.
    individuales = np.empty(n_pos)
    for i in range(n_pos):
        serie = ponderadas[:, i]
        umbral = float(np.quantile(serie, confidence))
        propia = serie[serie >= umbral]
        individuales[i] = float(np.mean(propia)) if propia.size else np.nan

    return RiskContributions(
        total=total,
        contributions=contribuciones,
        standalone=individuales,
        confidence=confidence,
        n_scenarios_in_tail=n_cola,
    )


def diversification_benefit(
    position_losses: np.ndarray,
    weights: np.ndarray | None = None,
    confidence: float = 0.975,
) -> float:
    """Beneficio de diversificacion del portafolio, en fraccion.

    Returns
    -------
    float
        ``1 - ES(portafolio) / suma de ES individuales``.
    """
    return euler_contributions(position_losses, weights, confidence).diversification_benefit
