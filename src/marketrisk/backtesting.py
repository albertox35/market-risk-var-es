"""Backtesting de VaR y Expected Shortfall.

Esta es la pieza sin la cual un motor de VaR es un generador de numeros. Un
modelo de riesgo no se juzga por su sofisticacion sino por si sus predicciones se
cumplen, y eso se contrasta con pruebas estadisticas concretas.

**Kupiec (1995), cobertura incondicional.** Comprueba que la *frecuencia* de
excepciones coincida con la esperada. Es necesaria pero insuficiente: un modelo
que acierta la frecuencia pero concentra todas las excepciones en una semana de
crisis pasa Kupiec y es inservible.

**Christoffersen (1998), cobertura condicional.** Anade la prueba de
**independencia**: las excepciones no deben agruparse. Esta es la que detecta el
fallo que Kupiec no ve, y es la razon por la que las dos se reportan juntas.

**Acerbi y Szekely (2014), para el ES.** El ES no es *elicitable* de forma
individual, asi que no admite un backtest del tipo «contar excepciones». Estas
pruebas comparan la magnitud de las perdidas en la cola contra lo que el modelo
predijo, no solo su numero. Con FRTB exigiendo ES al 97.5 %, no tener backtest de
ES deja el modelo sin validar en la medida que de verdad determina el capital.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "BacktestResult",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_conditional_coverage",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "traffic_light",
    "run_all_backtests",
]


@dataclass(frozen=True)
class BacktestResult:
    """Resultado de una prueba de backtesting."""

    name: str
    statistic: float
    p_value: float
    reject: bool
    """``True`` si se rechaza el modelo al nivel de significancia dado."""

    significance: float = 0.05
    detail: str = ""

    def __str__(self) -> str:
        veredicto = "RECHAZA el modelo" if self.reject else "no rechaza"
        extra = f" | {self.detail}" if self.detail else ""
        return (
            f"{self.name}: estadistico {self.statistic:.4f}, "
            f"p = {self.p_value:.4f} -> {veredicto}{extra}"
        )


def _validate_es_inputs(
    losses: np.ndarray, var_forecast: np.ndarray, es_forecast: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valida las entradas de un backtest de ES.

    El requisito que mas se incumple al cablear estas pruebas: **el VaR y el ES
    tienen que estar al MISMO nivel de confianza**. Las pruebas de
    Acerbi-Szekely definen las excepciones con el VaR de nivel alfa y comparan su
    magnitud contra el ES del mismo alfa. Pasar un VaR al 99 % con un ES al
    97.5 % --la combinacion natural si se toman los valores por omision de cada
    uno-- produce un numero que no contrasta nada: Z1 y Z2 se contradicen y el
    diagnostico es inservible.

    Como la coherencia de niveles no se puede comprobar desde los vectores, se
    verifica la condicion que si es observable: ``ES >= VaR`` punto por punto, que
    todo par coherente cumple por definicion.
    """
    l = np.asarray(losses, dtype=float).ravel()
    v = np.asarray(var_forecast, dtype=float).ravel()
    e = np.asarray(es_forecast, dtype=float).ravel()
    if not (l.shape == v.shape == e.shape):
        raise ValueError(
            f"perdidas {l.shape}, VaR {v.shape} y ES {e.shape} deben tener la misma forma"
        )
    if np.any(e <= 0.0):
        raise ValueError("el ES pronosticado debe ser positivo")
    violaciones = int(np.sum(e < v))
    if violaciones:
        raise ValueError(
            f"el ES pronosticado es menor que el VaR en {violaciones} de {l.size} fechas. "
            "El ES de un nivel nunca puede ser menor que el VaR del MISMO nivel, asi que "
            "los dos vectores corresponden a niveles de confianza distintos. Pasar el VaR "
            "y el ES evaluados al mismo alfa."
        )
    return l, v, e


def _exceptions(losses: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """Indicadora de excepciones: perdida que supera el VaR pronosticado."""
    l = np.asarray(losses, dtype=float).ravel()
    v = np.asarray(var_forecast, dtype=float).ravel()
    if l.shape != v.shape:
        raise ValueError(f"perdidas {l.shape} y pronosticos {v.shape} deben tener la misma forma")
    if l.size < 20:
        raise ValueError(f"se requieren al menos 20 observaciones, hay {l.size}")
    return (l > v).astype(int)


def kupiec_pof(
    losses: np.ndarray,
    var_forecast: np.ndarray,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> BacktestResult:
    """Prueba de proporcion de fallos de Kupiec (cobertura incondicional).

    Razon de verosimilitud bajo la hipotesis de que la probabilidad de excepcion
    es exactamente ``1 - confidence``:

    .. math::

        LR = -2\\log\\frac{(1-p)^{T-x}p^{x}}
                          {(1-\\hat p)^{T-x}\\hat p^{x}}
        \\;\\sim\\; \\chi^2_1

    Limitacion que hay que tener presente: **solo mira la frecuencia**. Un modelo
    con todas sus excepciones agrupadas en una semana la pasa sin problema.

    Returns
    -------
    BacktestResult
    """
    exc = _exceptions(losses, var_forecast)
    t_obs = int(exc.size)
    x = int(exc.sum())
    p = 1.0 - confidence
    p_hat = x / t_obs

    if x == 0:
        # Con cero excepciones la verosimilitud alternativa degenera; el limite
        # del estadistico es finito y vale -2*T*log(1-p).
        lr = -2.0 * t_obs * np.log(1.0 - p)
    elif x == t_obs:
        lr = -2.0 * t_obs * np.log(p)
    else:
        ll_nulo = (t_obs - x) * np.log(1.0 - p) + x * np.log(p)
        ll_alt = (t_obs - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        lr = -2.0 * (ll_nulo - ll_alt)

    p_valor = float(stats.chi2.sf(lr, df=1))
    return BacktestResult(
        name="Kupiec POF (cobertura incondicional)",
        statistic=float(lr),
        p_value=p_valor,
        reject=p_valor < significance,
        significance=significance,
        detail=f"{x} excepciones de {t_obs} ({p_hat:.3%}), esperadas {p:.3%}",
    )


def christoffersen_independence(
    losses: np.ndarray,
    var_forecast: np.ndarray,
    significance: float = 0.05,
) -> BacktestResult:
    """Prueba de independencia de Christoffersen.

    Contrasta si la probabilidad de excepcion depende de que haya habido
    excepcion el dia anterior, mediante una cadena de Markov de primer orden
    sobre la serie de excepciones:

    .. math::

        LR_{ind} = -2\\log\\frac{(1-\\hat\\pi)^{n_{00}+n_{10}}
                                  \\hat\\pi^{n_{01}+n_{11}}}
            {(1-\\hat\\pi_0)^{n_{00}}\\hat\\pi_0^{n_{01}}
             (1-\\hat\\pi_1)^{n_{10}}\\hat\\pi_1^{n_{11}}}
        \\;\\sim\\; \\chi^2_1

    Es la prueba que detecta el **agrupamiento** de excepciones, el fallo que
    Kupiec no puede ver y que en una crisis es exactamente lo que ocurre.

    Returns
    -------
    BacktestResult
    """
    exc = _exceptions(losses, var_forecast)
    anterior, actual = exc[:-1], exc[1:]
    n00 = int(np.sum((anterior == 0) & (actual == 0)))
    n01 = int(np.sum((anterior == 0) & (actual == 1)))
    n10 = int(np.sum((anterior == 1) & (actual == 0)))
    n11 = int(np.sum((anterior == 1) & (actual == 1)))

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        # Sin suficientes transiciones el estadistico no esta definido; no
        # rechazar es lo correcto, pero hay que decir que la prueba no informa.
        return BacktestResult(
            name="Christoffersen independencia",
            statistic=0.0,
            p_value=1.0,
            reject=False,
            significance=significance,
            detail=(
                f"transiciones insuficientes (n00={n00}, n01={n01}, n10={n10}, n11={n11}); "
                "la prueba no es informativa con esta muestra"
            ),
        )

    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)

    def ll(p0: float, p1: float) -> float:
        total = 0.0
        for n, p in ((n00, 1.0 - p0), (n01, p0), (n10, 1.0 - p1), (n11, p1)):
            if n > 0:
                if p <= 0.0:
                    return -np.inf
                total += n * np.log(p)
        return total

    lr = -2.0 * (ll(pi, pi) - ll(pi0, pi1))
    p_valor = float(stats.chi2.sf(max(lr, 0.0), df=1))
    return BacktestResult(
        name="Christoffersen independencia",
        statistic=float(lr),
        p_value=p_valor,
        reject=p_valor < significance,
        significance=significance,
        detail=f"P(exc|exc previa) = {pi1:.3%} frente a P(exc|sin exc) = {pi0:.3%}",
    )


def christoffersen_conditional_coverage(
    losses: np.ndarray,
    var_forecast: np.ndarray,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> BacktestResult:
    """Cobertura condicional: Kupiec mas independencia, conjuntamente.

    .. math:: LR_{cc} = LR_{uc} + LR_{ind} \\sim \\chi^2_2

    Es la prueba que hay que reportar como veredicto global, porque un modelo
    tiene que acertar **la frecuencia y el patron temporal** de sus excepciones.
    """
    uc = kupiec_pof(losses, var_forecast, confidence, significance)
    ind = christoffersen_independence(losses, var_forecast, significance)
    lr = uc.statistic + ind.statistic
    p_valor = float(stats.chi2.sf(lr, df=2))
    return BacktestResult(
        name="Christoffersen cobertura condicional",
        statistic=float(lr),
        p_value=p_valor,
        reject=p_valor < significance,
        significance=significance,
        detail=f"LR_uc = {uc.statistic:.4f} + LR_ind = {ind.statistic:.4f}",
    )


def acerbi_szekely_z1(
    losses: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    significance: float = 0.05,
    n_simulations: int = 20_000,
    seed: int = 20260101,
) -> BacktestResult:
    """Prueba Z1 de Acerbi y Szekely para el Expected Shortfall.

    .. math::

        Z_1 = \\frac{1}{N_T}\\sum_{t: L_t > VaR_t}
              \\frac{L_t}{ES_t} - 1

    donde ``N_T`` es el numero de excepciones. Bajo la hipotesis nula el
    estadistico tiene media cero: las perdidas en la cola promedian exactamente
    el ES pronosticado.

    Un ``Z1`` positivo significa que las perdidas de cola fueron **mayores** que
    lo predicho, es decir que el modelo subestima el riesgo. La distribucion
    nula se obtiene por simulacion, porque no tiene forma cerrada.

    Notes
    -----
    Z1 esta **condicionada a las excepciones**: solo valida la magnitud de la
    cola, no su frecuencia. Por eso se usa junto con Kupiec, no en su lugar.
    """
    l, v, e = _validate_es_inputs(losses, var_forecast, es_forecast)
    mascara = l > v
    n_exc = int(mascara.sum())
    if n_exc == 0:
        return BacktestResult(
            name="Acerbi-Szekely Z1 (ES)",
            statistic=0.0,
            p_value=1.0,
            reject=False,
            significance=significance,
            detail="sin excepciones: Z1 no es calculable",
        )

    z1 = float(np.sum(l[mascara] / e[mascara]) / n_exc - 1.0)

    # Distribucion nula por simulacion: se remuestrean las razones observadas
    # bajo la hipotesis de que su media es 1.
    rng = np.random.default_rng(seed)
    razones = l[mascara] / e[mascara]
    centradas = razones - np.mean(razones) + 1.0
    simulados = np.array(
        [
            np.mean(rng.choice(centradas, size=n_exc, replace=True)) - 1.0
            for _ in range(n_simulations)
        ]
    )
    # Cola de interes: Z1 grande significa subestimacion del riesgo.
    p_valor = float(np.mean(simulados >= z1))
    return BacktestResult(
        name="Acerbi-Szekely Z1 (ES)",
        statistic=z1,
        p_value=p_valor,
        reject=p_valor < significance,
        significance=significance,
        detail=f"{n_exc} excepciones; Z1 > 0 indica perdidas de cola mayores que el ES",
    )


def acerbi_szekely_z2(
    losses: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    confidence: float = 0.975,
    significance: float = 0.05,
    n_simulations: int = 20_000,
    seed: int = 20260101,
) -> BacktestResult:
    """Prueba Z2 de Acerbi y Szekely, no condicionada a las excepciones.

    .. math::

        Z_2 = \\frac{1}{T(1-\\alpha)}\\sum_{t}
              \\frac{L_t\\,\\mathbf{1}_{\\{L_t > VaR_t\\}}}{ES_t} - 1

    A diferencia de Z1, el denominador es el numero **esperado** de excepciones y
    no el observado, asi que Z2 valida **frecuencia y magnitud a la vez**. Es la
    prueba mas completa de las dos y la que conviene reportar como veredicto de
    ES.
    """
    l, v, e = _validate_es_inputs(losses, var_forecast, es_forecast)
    if not 0.0 < confidence < 1.0:
        raise ValueError("el nivel de confianza debe estar en (0, 1)")
    t_obs = l.size
    p = 1.0 - confidence
    contribuciones = np.where(l > v, l / e, 0.0)
    z2 = float(np.sum(contribuciones) / (t_obs * p) - 1.0)

    rng = np.random.default_rng(seed)
    centradas = contribuciones - np.mean(contribuciones) + p
    simulados = np.array(
        [
            np.sum(rng.choice(centradas, size=t_obs, replace=True)) / (t_obs * p) - 1.0
            for _ in range(n_simulations)
        ]
    )
    p_valor = float(np.mean(simulados >= z2))
    return BacktestResult(
        name="Acerbi-Szekely Z2 (ES)",
        statistic=z2,
        p_value=p_valor,
        reject=p_valor < significance,
        significance=significance,
        detail=f"valida frecuencia y magnitud; {int(np.sum(l > v))} excepciones de {t_obs}",
    )


def traffic_light(n_exceptions: int, n_observations: int, confidence: float = 0.99) -> str:
    """Semaforo de Basilea sobre el numero de excepciones.

    Zonas definidas por la probabilidad acumulada binomial de observar ese numero
    de excepciones o menos: verde por debajo del 95 %, amarilla entre 95 % y
    99.99 %, roja por encima.

    Returns
    -------
    str
        ``"verde"``, ``"amarilla"`` o ``"roja"``.
    """
    if n_observations <= 0:
        raise ValueError("se requiere al menos una observacion")
    if not 0 <= n_exceptions <= n_observations:
        raise ValueError("el numero de excepciones esta fuera de rango")
    acumulada = float(stats.binom.cdf(n_exceptions, n_observations, 1.0 - confidence))
    if acumulada < 0.95:
        return "verde"
    if acumulada < 0.9999:
        return "amarilla"
    return "roja"


def run_all_backtests(
    losses: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray | None = None,
    es_var_forecast: np.ndarray | None = None,
    confidence: float = 0.99,
    es_confidence: float = 0.975,
    significance: float = 0.05,
) -> list[BacktestResult]:
    """Ejecuta la bateria completa de backtests.

    Parameters
    ----------
    var_forecast:
        VaR pronosticado al nivel ``confidence``. Alimenta Kupiec y
        Christoffersen.
    es_forecast:
        ES pronosticado al nivel ``es_confidence``.
    es_var_forecast:
        VaR pronosticado al nivel **``es_confidence``**, no al de ``confidence``.
        Es obligatorio para los backtests de ES, porque Acerbi-Szekely define
        las excepciones con el VaR del mismo alfa que el ES. Si se omite, las
        pruebas de ES se saltan en lugar de ejecutarse con niveles incoherentes.

    Returns
    -------
    list[BacktestResult]
        Kupiec, independencia y cobertura condicional; mas Z1 y Z2 cuando se
        suministran ``es_forecast`` y ``es_var_forecast``.
    """
    resultados = [
        kupiec_pof(losses, var_forecast, confidence, significance),
        christoffersen_independence(losses, var_forecast, significance),
        christoffersen_conditional_coverage(losses, var_forecast, confidence, significance),
    ]
    if es_forecast is None:
        return resultados
    if es_var_forecast is None:
        raise ValueError(
            "para backtestear el ES hace falta es_var_forecast: el VaR al mismo nivel "
            f"que el ES ({es_confidence:.1%}), no al de confidence ({confidence:.1%}). "
            "Acerbi-Szekely define las excepciones con el VaR del mismo alfa."
        )
    resultados.append(acerbi_szekely_z1(losses, es_var_forecast, es_forecast, significance))
    resultados.append(
        acerbi_szekely_z2(losses, es_var_forecast, es_forecast, es_confidence, significance)
    )
    return resultados
