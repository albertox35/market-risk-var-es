"""GARCH(1,1) y DCC-GARCH por cuasi-maxima verosimilitud.

La volatilidad de los retornos financieros no es constante: se agrupa. GARCH
captura esa persistencia, y es lo que permite que la simulacion historica
filtrada reescale los residuos al nivel de riesgo **de hoy** en lugar de al
promedio de la muestra.

**Por que cuasi-maxima verosimilitud.** Se maximiza la verosimilitud gaussiana
aunque se sepa que los retornos no son normales. El estimador sigue siendo
consistente para los parametros de la varianza condicional mientras la ecuacion
de varianza este bien especificada (Bollerslev y Wooldridge, 1992); solo los
errores estandar necesitan la correccion sandwich. Suponer normalidad **para la
distribucion del VaR** seria otra cosa, y es justamente lo que la simulacion
historica filtrada evita.

**DCC en dos etapas.** Estimar un GARCH multivariado completo es inviable:
el numero de parametros crece con el cuadrado de la dimension. DCC (Engle, 2002)
lo descompone en volatilidades univariadas primero y correlaciones condicionales
despues, con solo dos parametros para toda la matriz de correlacion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["GARCHResult", "DCCResult", "fit_garch", "simulate_garch", "fit_dcc"]

_TINY = 1e-12


@dataclass(frozen=True)
class GARCHResult:
    """Ajuste de un GARCH(1,1).

    .. math:: \\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2
                            + \\beta \\sigma_{t-1}^2
    """

    omega: float
    alpha: float
    beta: float
    conditional_variance: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    converged: bool

    @property
    def persistence(self) -> float:
        """``alpha + beta``. Debe ser menor que 1 para que el proceso sea estacionario."""
        return float(self.alpha + self.beta)

    @property
    def long_run_variance(self) -> float:
        """Varianza incondicional implicita ``omega / (1 - alpha - beta)``."""
        p = self.persistence
        if p >= 1.0:
            return float("inf")
        return float(self.omega / (1.0 - p))

    @property
    def conditional_volatility(self) -> np.ndarray:
        """Volatilidad condicional."""
        return np.sqrt(self.conditional_variance)

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        """Pronostico de varianza a varios pasos.

        .. math:: E[\\sigma_{T+h}^2] = \\bar\\sigma^2
            + (\\alpha+\\beta)^{h-1}(\\sigma_{T+1}^2 - \\bar\\sigma^2)

        La varianza converge geometricamente a su nivel de largo plazo a tasa
        ``alpha + beta``. Con persistencia de 0.99 esa convergencia es lentisima,
        que es justo lo que se observa en datos reales.
        """
        if horizon < 1:
            raise ValueError("el horizonte debe ser al menos 1")
        ultima = float(self.conditional_variance[-1])
        eps2 = float(self.standardized_residuals[-1] ** 2 * ultima)
        siguiente = self.omega + self.alpha * eps2 + self.beta * ultima
        if self.persistence >= 1.0:
            return np.full(horizon, siguiente)
        largo = self.long_run_variance
        h = np.arange(horizon)
        return largo + (self.persistence**h) * (siguiente - largo)


def _garch_filter(
    returns: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """Recursion de la varianza condicional."""
    n = returns.size
    var = np.empty(n)
    # Inicializar en la varianza muestral: es el arranque estandar y evita que
    # la eleccion del primer valor contamine el inicio de la muestra.
    var[0] = float(np.var(returns))
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
    return var


def _negative_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.9999:
        return 1e10
    var = _garch_filter(returns, omega, alpha, beta)
    if np.any(var <= 0.0) or not np.all(np.isfinite(var)):
        return 1e10
    return float(0.5 * np.sum(np.log(var) + returns**2 / var))


def fit_garch(returns: np.ndarray, mean_adjust: bool = True) -> GARCHResult:
    """Estima un GARCH(1,1) por cuasi-maxima verosimilitud gaussiana.

    Parameters
    ----------
    returns:
        Serie de retornos.
    mean_adjust:
        Restar la media muestral antes de ajustar. Para retornos diarios la
        media es despreciable frente a la volatilidad, pero no restarla sesga
        ``omega`` al alza.

    Returns
    -------
    GARCHResult

    Notes
    -----
    **La serie se escala a varianza unitaria antes de optimizar**, y no es un
    detalle cosmetico. En retornos diarios ``omega`` es del orden de ``1e-6``
    mientras ``alpha`` y ``beta`` son del orden de ``0.1`` y ``0.9``: cuatro
    ordenes de magnitud de diferencia. Un optimizador que estima el gradiente
    por diferencias finitas con un paso unico no puede manejar esa disparidad y
    termina casi sin moverse del punto inicial, devolviendo los valores de
    arranque como si fueran el optimo.

    Es un fallo silencioso y especialmente traicionero: si el punto inicial esta
    cerca de los valores verdaderos, el ajuste parece excelente. Tras escalar,
    los tres parametros son del mismo orden y la optimizacion se comporta; al
    final ``omega`` se transforma de vuelta multiplicando por la varianza de
    escala, y ``alpha`` y ``beta`` son invariantes de escala.
    """
    r_original = np.asarray(returns, dtype=float).ravel()
    if r_original.size < 100:
        raise ValueError(f"se requieren al menos 100 observaciones, hay {r_original.size}")
    r = r_original - r_original.mean() if mean_adjust else r_original.copy()

    escala2 = float(np.var(r))
    if escala2 <= 0.0:
        raise ValueError("la serie tiene varianza nula")
    escala = float(np.sqrt(escala2))
    r_s = r / escala  # varianza unitaria: los tres parametros quedan comparables

    mejor = None
    # Multiarranque: la verosimilitud GARCH es plana a lo largo de la direccion
    # de persistencia, asi que un solo arranque puede quedarse atrapado.
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.20, 0.70), (0.03, 0.96)):
        x0 = np.array([1.0 - a0 - b0, a0, b0])
        salida = minimize(
            _negative_log_likelihood,
            x0,
            args=(r_s,),
            method="SLSQP",
            bounds=[(1e-10, 10.0), (0.0, 0.5), (0.0, 0.999)],
            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}],
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        if mejor is None or salida.fun < mejor.fun:
            mejor = salida

    assert mejor is not None
    omega_s, alpha, beta = (float(v) for v in mejor.x)
    omega = omega_s * escala2  # deshacer el escalado
    var = _garch_filter(r, omega, alpha, beta)
    return GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_variance=var,
        standardized_residuals=r / np.sqrt(var),
        log_likelihood=float(-mejor.fun),
        converged=bool(mejor.success),
    )


def simulate_garch(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    seed: int = 20260101,
    innovation_df: float | None = None,
    burn_in: int = 500,
) -> np.ndarray:
    """Simula una serie GARCH(1,1).

    Parameters
    ----------
    innovation_df:
        Si se da, las innovaciones son t de Student estandarizadas con esos
        grados de libertad en lugar de normales. Sirve para comprobar que la
        estimacion por cuasi-maxima verosimilitud sigue siendo consistente con
        innovaciones de colas gruesas, que es su propiedad central.
    burn_in:
        Observaciones iniciales que se descartan para eliminar el efecto del
        arranque.
    """
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta debe ser menor que 1 para que el proceso sea estacionario")
    rng = np.random.default_rng(seed)
    total = n + burn_in
    if innovation_df is None:
        z = rng.standard_normal(total)
    else:
        if innovation_df <= 2.0:
            raise ValueError("se requieren mas de 2 grados de libertad para que exista la varianza")
        z = rng.standard_t(innovation_df, total)
        z = z / np.sqrt(innovation_df / (innovation_df - 2.0))

    var = omega / (1.0 - alpha - beta)
    r = np.empty(total)
    for t in range(total):
        r[t] = np.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    return r[burn_in:]


@dataclass(frozen=True)
class DCCResult:
    """Ajuste DCC-GARCH."""

    a: float
    b: float
    garch_fits: list[GARCHResult]
    correlations: np.ndarray
    """Matrices de correlacion condicional, ``(T, n, n)``."""

    standardized_residuals: np.ndarray
    """Residuos estandarizados por el GARCH univariado, ``(T, n)``."""

    converged: bool

    @property
    def persistence(self) -> float:
        """``a + b``; debe ser menor que 1."""
        return float(self.a + self.b)

    @property
    def latest_correlation(self) -> np.ndarray:
        """Matriz de correlacion condicional mas reciente."""
        return self.correlations[-1]


def _dcc_recursion(
    z: np.ndarray, a: float, b: float, q_bar: np.ndarray
) -> np.ndarray:
    """Recursion de la matriz Q de DCC y normalizacion a correlacion."""
    t_obs, n = z.shape
    correlaciones = np.empty((t_obs, n, n))
    q = q_bar.copy()
    for t in range(t_obs):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            q = (1.0 - a - b) * q_bar + a * outer + b * q
        d = np.sqrt(np.clip(np.diag(q), _TINY, None))
        correlaciones[t] = q / np.outer(d, d)
        np.fill_diagonal(correlaciones[t], 1.0)
    return correlaciones


def fit_dcc(returns: np.ndarray) -> DCCResult:
    """Estima un DCC-GARCH en dos etapas (Engle, 2002).

    1. Un GARCH(1,1) univariado por activo; se guardan los residuos
       estandarizados.
    2. Sobre esos residuos, dos parametros ``a`` y ``b`` que gobiernan la
       dinamica de **toda** la matriz de correlacion:

       .. math:: Q_t = (1-a-b)\\bar Q + a\\, z_{t-1}z_{t-1}' + b\\, Q_{t-1}

    La ventaja es de escala: un BEKK completo necesita ``O(n^4)`` parametros,
    DCC necesita ``2`` para la parte de correlacion sin importar ``n``.

    Parameters
    ----------
    returns:
        Matriz ``(T, n)`` de retornos.

    Returns
    -------
    DCCResult
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns debe ser una matriz (T, n_activos)")
    t_obs, n = r.shape
    if n < 2:
        raise ValueError("DCC requiere al menos dos activos")
    if t_obs < 100:
        raise ValueError(f"se requieren al menos 100 observaciones, hay {t_obs}")

    ajustes = [fit_garch(r[:, i]) for i in range(n)]
    z = np.column_stack([f.standardized_residuals for f in ajustes])
    q_bar = np.cov(z, rowvar=False)

    def neg_ll(params: np.ndarray) -> float:
        a, b = params
        if a < 0.0 or b < 0.0 or a + b >= 0.9999:
            return 1e10
        corr = _dcc_recursion(z, a, b, q_bar)
        total = 0.0
        for t in range(t_obs):
            c = corr[t]
            signo, logdet = np.linalg.slogdet(c)
            if signo <= 0:
                return 1e10
            try:
                cuadratica = float(z[t] @ np.linalg.solve(c, z[t]))
            except np.linalg.LinAlgError:
                return 1e10
            total += 0.5 * (logdet + cuadratica)
        return total

    salida = minimize(
        neg_ll,
        np.array([0.02, 0.95]),
        method="SLSQP",
        bounds=[(0.0, 0.5), (0.0, 0.999)],
        constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]}],
        options={"maxiter": 200, "ftol": 1e-9},
    )
    a, b = (float(v) for v in salida.x)
    return DCCResult(
        a=a,
        b=b,
        garch_fits=ajustes,
        correlations=_dcc_recursion(z, a, b, q_bar),
        standardized_residuals=z,
        converged=bool(salida.success),
    )
