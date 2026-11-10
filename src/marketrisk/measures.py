"""Medidas de riesgo: VaR y Expected Shortfall.

**El VaR no es subaditivo.** Puede ocurrir que el VaR de una cartera supere la
suma de los VaR de sus partes, lo que significa que diversificar «aumenta» el
riesgo medido. No es una curiosidad teorica: es la razon por la que Basilea III
y FRTB sustituyeron el VaR al 99 % a 10 dias por el **Expected Shortfall al
97.5 %** como medida de capital.

El ES si es coherente en el sentido de Artzner y coautores (1999): monotono,
homogeneo, invariante por traslacion y **subaditivo**. Y ademas mira dentro de la
cola, mientras el VaR solo dice donde empieza.

El nivel 97.5 % no es casual: para una normal, el ES al 97.5 % y el VaR al 99 %
son casi iguales, asi que el cambio de medida no altera el nivel de capital en
condiciones normales pero si captura mejor los eventos extremos.

Convencion de signos de este modulo: las **perdidas son positivas**. Un VaR de
0.03 significa una perdida del 3 %. Es la convencion de gestion de riesgos y
evita el error de signo mas frecuente del area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RiskMeasures",
    "value_at_risk",
    "expected_shortfall",
    "risk_measures",
    "scale_to_horizon",
]


@dataclass(frozen=True)
class RiskMeasures:
    """VaR y ES a un nivel de confianza, con perdidas positivas."""

    confidence: float
    var: float
    expected_shortfall: float
    n_observations: int
    n_tail: int
    """Observaciones usadas para el ES. Pocas observaciones en la cola hacen
    inestable la estimacion, por eso se expone."""

    @property
    def tail_ratio(self) -> float:
        """``ES / VaR``. Para una normal vale ~1.15 al 99 %; mas alto indica colas gruesas."""
        return float(self.expected_shortfall / self.var) if self.var else float("nan")

    def __str__(self) -> str:
        return (
            f"VaR({self.confidence:.1%}) = {self.var:.6f}, "
            f"ES({self.confidence:.1%}) = {self.expected_shortfall:.6f}, "
            f"ES/VaR = {self.tail_ratio:.3f} ({self.n_tail}/{self.n_observations} en cola)"
        )


def _validate(losses: np.ndarray, confidence: float) -> np.ndarray:
    arr = np.asarray(losses, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("la muestra de perdidas esta vacia")
    if not np.all(np.isfinite(arr)):
        raise ValueError("la muestra contiene valores no finitos")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"el nivel de confianza debe estar en (0, 1), es {confidence}")
    return arr


def value_at_risk(losses: np.ndarray, confidence: float = 0.99) -> float:
    """Valor en Riesgo: el cuantil de la distribucion de perdidas.

    Parameters
    ----------
    losses:
        Muestra de perdidas, **positivas**.
    confidence:
        Nivel de confianza.

    Returns
    -------
    float

    Notes
    -----
    Se usa el cuantil empirico con interpolacion lineal. Para muestras pequenias
    la eleccion de definicion de cuantil cambia el resultado de forma
    apreciable; con las ~1000 observaciones de una ventana regulatoria de cuatro
    anios la diferencia es menor que el error de muestreo.
    """
    arr = _validate(losses, confidence)
    return float(np.quantile(arr, confidence))


def _expected_shortfall_tail(
    arr: np.ndarray, var: float, confidence: float
) -> tuple[float, int]:
    """Estimador integral del ES. Devuelve el ES y el tamanio de la cola estricta.

    **El estimador ingenuo es incorrecto.** La media condicional
    ``E[L | L >= VaR]`` solo coincide con el ES cuando la distribucion es
    continua en el cuantil. Con atomos --y una cartera de credito los tiene por
    construccion-- falla, y puede fallar de forma espectacular.

    El caso que lo evidencia: un bono con 4 % de probabilidad de impago y
    perdida 100, al 95 % de confianza. Como ``P(L > 0) = 4 % < 5 %``, el VaR es
    **cero**, con lo que la condicion ``L >= 0`` incluye *toda* la muestra y el
    «ES» ingenuo devuelve la media incondicional, 4. El ES correcto es 80.

    Es un error de factor 20 en la medida que FRTB usa para capital, y ademas
    rompe la subaditividad del ES: con el estimador ingenuo, dos bonos
    independientes dan ``ES(A+B) = 102 > ES(A) + ES(B) = 8``, que es absurdo.

    La definicion correcta promedia la peor fraccion ``1-alpha`` de la
    distribucion, repartiendo el atomo que cae sobre el cuantil:

    .. math::

        ES_\\alpha = \\frac{1}{1-\\alpha}\\left(
            E\\left[L\\,\\mathbf{1}_{\\{L > VaR_\\alpha\\}}\\right]
            + VaR_\\alpha\\left(F(VaR_\\alpha) - \\alpha\\right)\\right)
    """
    n = arr.size
    masa_objetivo = n * (1.0 - confidence)
    if masa_objetivo <= 0.0:
        raise ValueError(
            f"el nivel {confidence:.6%} no deja masa en la cola con {n} observaciones; "
            "el ES no es estimable. Usar un nivel menor o un ajuste parametrico."
        )
    cola_estricta = arr[arr > var]
    # Peso residual que corresponde al atomo situado exactamente en el VaR.
    peso_atomo = max(masa_objetivo - cola_estricta.size, 0.0)
    suma = float(np.sum(cola_estricta)) + var * peso_atomo
    return float(suma / masa_objetivo), int(cola_estricta.size)


def expected_shortfall(losses: np.ndarray, confidence: float = 0.975) -> float:
    """Expected Shortfall: promedio de la peor fraccion ``1-alpha`` de las perdidas.

    .. math:: ES_\\alpha = E\\left[L \\mid L \\geq VaR_\\alpha\\right]

    Returns
    -------
    float

    Para distribuciones continuas coincide con la media condicional
    ``E[L | L >= VaR]``. Para distribuciones **con atomos** no, y ahi la version
    condicional rompe la subaditividad del ES, que es justamente la propiedad por
    la que se eligio esta medida. Ver :func:`_expected_shortfall_tail`.

    Returns
    -------
    float
    """
    arr = _validate(losses, confidence)
    var = float(np.quantile(arr, confidence))
    es, _ = _expected_shortfall_tail(arr, var, confidence)
    return es


def risk_measures(losses: np.ndarray, confidence: float = 0.975) -> RiskMeasures:
    """VaR y ES juntos, con el conteo de observaciones de la cola."""
    arr = _validate(losses, confidence)
    var = float(np.quantile(arr, confidence))
    es, n_cola = _expected_shortfall_tail(arr, var, confidence)
    return RiskMeasures(
        confidence=confidence,
        var=var,
        expected_shortfall=es,
        n_observations=int(arr.size),
        n_tail=n_cola,
    )


def scale_to_horizon(
    measure: float,
    horizon_days: int = 10,
    current_variance: float | None = None,
    long_run_variance: float | None = None,
    persistence: float | None = None,
) -> float:
    """Escala una medida de riesgo de un dia a un horizonte de varios dias.

    La **regla de la raiz del tiempo** supone retornos independientes e
    identicamente distribuidos. Con volatilidad persistente es incorrecta, y el
    sentido del sesgo depende del estado:

    - Si la volatilidad de hoy esta **por debajo** de su nivel de largo plazo,
      la raiz del tiempo **subestima** el riesgo, porque la varianza va a subir.
    - Si esta **por encima**, lo **sobreestima**.

    Para corregirlo hace falta la varianza acumulada del GARCH:

    .. math::

        \\sum_{j=1}^{h} E[\\sigma_{T+j}^2] = h\\bar\\sigma^2
            + \\left(\\sigma_{T+1}^2 - \\bar\\sigma^2\\right)
              \\frac{1-p^{h}}{1-p}

    y eso requiere las **dos** varianzas, no solo la persistencia: con la
    persistencia sola el problema esta indeterminado, porque no se sabe de que
    nivel se parte.

    Parameters
    ----------
    measure:
        Medida de riesgo a un dia.
    horizon_days:
        Horizonte objetivo en dias.
    current_variance, long_run_variance, persistence:
        Si se dan los tres, se aplica la correccion GARCH. Si se omite alguno se
        usa la raiz del tiempo.

    Returns
    -------
    float

    Notes
    -----
    FRTB no escala: exige simular directamente a los horizontes de liquidez
    correspondientes. Esta funcion existe para comparar con la practica previa y
    para cuantificar el error de la raiz del tiempo, no como sustituto.
    """
    if horizon_days <= 0:
        raise ValueError("el horizonte debe ser positivo")
    h = int(horizon_days)

    parametros_garch = (current_variance, long_run_variance, persistence)
    if any(p is None for p in parametros_garch):
        if any(p is not None for p in parametros_garch):
            raise ValueError(
                "la correccion GARCH requiere current_variance, long_run_variance y "
                "persistence a la vez; con la persistencia sola el escalado esta "
                "indeterminado, porque no se sabe de que nivel de volatilidad se parte"
            )
        return float(measure * np.sqrt(h))

    assert current_variance is not None and long_run_variance is not None
    assert persistence is not None
    if not 0.0 <= persistence < 1.0:
        raise ValueError("la persistencia debe estar en [0, 1)")
    if current_variance <= 0.0 or long_run_variance <= 0.0:
        raise ValueError("las varianzas deben ser positivas")

    acumulada = h * long_run_variance + (current_variance - long_run_variance) * (
        1.0 - persistence**h
    ) / (1.0 - persistence)
    if acumulada <= 0.0:
        raise ValueError("la varianza acumulada resulto no positiva; revisar los parametros")
    # La medida de un dia es proporcional a sqrt(current_variance).
    return float(measure * np.sqrt(acumulada / current_variance))
