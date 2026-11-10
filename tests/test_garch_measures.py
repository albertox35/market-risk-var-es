"""Validacion de GARCH y de las medidas de riesgo."""

from __future__ import annotations

import numpy as np
import pytest

from marketrisk import (
    euler_contributions,
    expected_shortfall,
    fit_dcc,
    fit_garch,
    risk_measures,
    scale_to_horizon,
    simulate_garch,
    value_at_risk,
)

VERDAD = dict(omega=5.0e-6, alpha=0.15, beta=0.80)


# -------------------------------------------------------------------- GARCH


def test_simulate_garch_rechaza_procesos_no_estacionarios():
    with pytest.raises(ValueError, match="menor que 1"):
        simulate_garch(1000, omega=1e-6, alpha=0.5, beta=0.6)


def test_fit_garch_exige_muestra_suficiente():
    with pytest.raises(ValueError, match="al menos 100"):
        fit_garch(np.zeros(50))


def test_garch_recupera_los_parametros_simulados():
    """Los valores verdaderos no coinciden con ningun punto de arranque.

    Es una condicion necesaria de la prueba: si el arranque estuviera en la
    verdad, un optimizador que no se mueve pasaria el test.
    """
    r = simulate_garch(12_000, seed=17, **VERDAD)
    f = fit_garch(r)
    assert f.converged
    assert f.alpha == pytest.approx(VERDAD["alpha"], abs=0.03)
    assert f.beta == pytest.approx(VERDAD["beta"], abs=0.03)
    assert f.persistence == pytest.approx(
        VERDAD["alpha"] + VERDAD["beta"], abs=0.02
    )


def test_el_error_de_estimacion_decrece_con_el_tamanio_de_muestra():
    """Consistencia: es lo que distingue un estimador de una coincidencia.

    Se promedia el error sobre varias semillas. Con una sola, una realizacion
    afortunada de la muestra pequenia puede superar a la grande sin que eso diga
    nada del estimador: la consistencia es una propiedad de la distribucion
    muestral, no de una trayectoria.
    """
    verdadera = VERDAD["alpha"] + VERDAD["beta"]
    errores = []
    for n in (1_000, 4_000, 16_000):
        por_semilla = [
            abs(fit_garch(simulate_garch(n, seed=s, **VERDAD)).persistence - verdadera)
            for s in (101, 202, 303, 404, 505)
        ]
        errores.append(float(np.mean(por_semilla)))
    assert errores[1] < errores[0], errores
    assert errores[2] < errores[1], errores


def test_el_ajuste_es_invariante_al_escalado_de_la_serie():
    """Prueba de la correccion de escalado interna.

    ``alpha`` y ``beta`` son invariantes de escala y ``omega`` escala con la
    varianza. Si el optimizador no estuviera trabajando sobre la serie
    normalizada, esta invariancia se rompe: con ``omega`` del orden de ``1e-6``
    junto a ``beta`` del orden de ``0.9``, las diferencias finitas de un
    optimizador con paso unico apenas mueven el iterado y devuelve el punto de
    arranque disfrazado de optimo.
    """
    r = simulate_garch(6_000, seed=31, **VERDAD)
    base = fit_garch(r)
    for factor in (0.01, 100.0):
        escalado = fit_garch(r * factor)
        assert escalado.alpha == pytest.approx(base.alpha, abs=1e-3)
        assert escalado.beta == pytest.approx(base.beta, abs=1e-3)
        assert escalado.omega == pytest.approx(base.omega * factor**2, rel=1e-2)


def test_la_cuasi_maxima_verosimilitud_sigue_siendo_consistente_con_colas_gruesas():
    """Propiedad central del estimador QML (Bollerslev y Wooldridge, 1992).

    Se maximiza una verosimilitud gaussiana sobre datos que **no** son normales,
    y el estimador sigue siendo consistente para los parametros de la varianza.
    """
    r = simulate_garch(12_000, seed=41, innovation_df=5.0, **VERDAD)
    f = fit_garch(r)
    assert f.alpha == pytest.approx(VERDAD["alpha"], abs=0.05)
    assert f.beta == pytest.approx(VERDAD["beta"], abs=0.05)


def test_la_varianza_de_largo_plazo_es_coherente_con_la_muestral():
    r = simulate_garch(20_000, seed=53, **VERDAD)
    f = fit_garch(r)
    assert f.long_run_variance == pytest.approx(float(np.var(r)), rel=0.25)


def test_los_residuos_estandarizados_tienen_varianza_unitaria():
    f = fit_garch(simulate_garch(8_000, seed=59, **VERDAD))
    assert float(np.std(f.standardized_residuals, ddof=1)) == pytest.approx(1.0, abs=0.05)


def test_el_pronostico_de_varianza_converge_al_nivel_de_largo_plazo():
    f = fit_garch(simulate_garch(8_000, seed=61, **VERDAD))
    pronostico = f.forecast_variance(400)
    assert pronostico[-1] == pytest.approx(f.long_run_variance, rel=0.02)
    # Y la convergencia es monotona, hacia arriba o hacia abajo segun el estado.
    diferencias = np.abs(pronostico - f.long_run_variance)
    assert np.all(np.diff(diferencias) <= 1e-12)


def test_dcc_estima_correlaciones_condicionales():
    rng = np.random.default_rng(71)
    base = simulate_garch(1_500, seed=73, **VERDAD)
    # Dos series con correlacion positiva fuerte y volatilidad propia.
    comun = base
    a = comun + 0.4 * rng.standard_normal(1_500) * base.std()
    b = comun + 0.4 * rng.standard_normal(1_500) * base.std()
    res = fit_dcc(np.column_stack([a, b]))
    assert res.persistence < 1.0
    assert res.correlations.shape == (1_500, 2, 2)
    # Diagonal unitaria y simetria en todas las fechas.
    np.testing.assert_allclose(np.diagonal(res.correlations, axis1=1, axis2=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        res.correlations, np.transpose(res.correlations, (0, 2, 1)), atol=1e-12
    )
    # La correlacion media debe ser claramente positiva por construccion.
    assert float(np.mean(res.correlations[:, 0, 1])) > 0.5


def test_dcc_exige_al_menos_dos_activos():
    with pytest.raises(ValueError, match="al menos dos activos"):
        fit_dcc(simulate_garch(500, seed=79, **VERDAD).reshape(-1, 1))


# ---------------------------------------------------------- medidas de riesgo


def test_el_expected_shortfall_nunca_es_menor_que_el_var():
    """Desigualdad que se cumple por definicion, en cualquier muestra."""
    rng = np.random.default_rng(83)
    for muestra in (
        rng.standard_normal(10_000),
        rng.standard_t(3.0, 10_000),
        rng.exponential(1.0, 10_000),
    ):
        perdidas = np.abs(muestra)
        for c in (0.90, 0.95, 0.99, 0.995):
            m = risk_measures(perdidas, c)
            assert m.expected_shortfall >= m.var


def test_el_expected_shortfall_es_subaditivo():
    """Propiedad de coherencia que el VaR no tiene."""
    rng = np.random.default_rng(89)
    a = np.abs(rng.standard_t(3.0, 50_000))
    b = np.abs(rng.standard_t(3.0, 50_000))
    es_a = expected_shortfall(a, 0.975)
    es_b = expected_shortfall(b, 0.975)
    es_suma = expected_shortfall(a + b, 0.975)
    assert es_suma <= es_a + es_b + 1e-9


def test_el_var_no_es_subaditivo_y_aqui_esta_el_contraejemplo():
    """La razon por la que FRTB cambio a Expected Shortfall.

    Dos bonos independientes, cada uno con 4 % de probabilidad de impago y
    perdida 100. Al 95 % de confianza:

    - Por separado, ``P(perdida) = 4 % < 5 %``, asi que cada VaR es **cero**.
    - Juntos, ``P(al menos un impago) = 1 - 0.96^2 = 7.84 % > 5 %``, asi que el
      VaR del portafolio es **positivo**.

    Es decir ``VaR(A+B) > VaR(A) + VaR(B)``: la medida dice que diversificar
    aumenta el riesgo. No es un artefacto numerico, es la medida que esta mal.
    """
    rng = np.random.default_rng(97)
    n = 400_000
    impago_a = (rng.random(n) < 0.04).astype(float) * 100.0
    impago_b = (rng.random(n) < 0.04).astype(float) * 100.0

    var_a = value_at_risk(impago_a, 0.95)
    var_b = value_at_risk(impago_b, 0.95)
    var_cartera = value_at_risk(impago_a + impago_b, 0.95)

    assert var_a == pytest.approx(0.0)
    assert var_b == pytest.approx(0.0)
    assert var_cartera > 0.0
    assert var_cartera > var_a + var_b

    # Y el ES no tiene este problema.
    es_a = expected_shortfall(impago_a, 0.95)
    es_b = expected_shortfall(impago_b, 0.95)
    es_cartera = expected_shortfall(impago_a + impago_b, 0.95)
    assert es_cartera <= es_a + es_b + 1e-9


def test_la_razon_es_var_de_una_normal_es_la_teorica():
    """Para una normal estandar, ``ES/VaR`` al 99 % vale ~1.15."""
    rng = np.random.default_rng(101)
    perdidas = rng.standard_normal(500_000)
    m = risk_measures(perdidas, 0.99)
    assert m.tail_ratio == pytest.approx(1.146, abs=0.02)


def test_las_colas_gruesas_elevan_la_razon_es_var():
    rng = np.random.default_rng(103)
    normal = risk_measures(rng.standard_normal(200_000), 0.99)
    t3 = risk_measures(rng.standard_t(3.0, 200_000), 0.99)
    assert t3.tail_ratio > normal.tail_ratio


def test_las_medidas_rechazan_entradas_invalidas():
    with pytest.raises(ValueError, match="vacia"):
        value_at_risk(np.array([]), 0.99)
    with pytest.raises(ValueError, match="no finitos"):
        value_at_risk(np.array([1.0, np.nan, 2.0]), 0.99)
    with pytest.raises(ValueError, match="nivel de confianza"):
        value_at_risk(np.ones(100), 1.5)


# ------------------------------------------------------ escalado de horizonte


def test_la_raiz_del_tiempo_es_el_comportamiento_por_omision():
    assert scale_to_horizon(1.0, horizon_days=10) == pytest.approx(np.sqrt(10.0))


def test_la_correccion_garch_exige_los_tres_parametros():
    """Con la persistencia sola el escalado esta indeterminado."""
    with pytest.raises(ValueError, match="indeterminado"):
        scale_to_horizon(1.0, horizon_days=10, persistence=0.95)


def test_la_correccion_garch_infla_el_riesgo_si_la_volatilidad_esta_baja():
    """Signo del sesgo de la raiz del tiempo, que depende del estado.

    Si la volatilidad de hoy esta por debajo de su nivel de largo plazo, va a
    subir, asi que la raiz del tiempo **subestima** el riesgo del horizonte.
    """
    raiz = scale_to_horizon(1.0, horizon_days=10)
    baja = scale_to_horizon(
        1.0, horizon_days=10, current_variance=0.5e-4, long_run_variance=1e-4, persistence=0.95
    )
    alta = scale_to_horizon(
        1.0, horizon_days=10, current_variance=2e-4, long_run_variance=1e-4, persistence=0.95
    )
    assert baja > raiz
    assert alta < raiz


def test_sin_persistencia_la_correccion_coincide_con_la_raiz_del_tiempo():
    """Con ``p = 0`` la varianza vuelve al nivel de largo plazo en un paso."""
    valor = scale_to_horizon(
        1.0, horizon_days=10, current_variance=1e-4, long_run_variance=1e-4, persistence=0.0
    )
    assert valor == pytest.approx(np.sqrt(10.0))


# -------------------------------------------------------------- atribucion


def test_las_contribuciones_de_euler_suman_el_total():
    """Propiedad que distingue Euler de cualquier reparto ad hoc."""
    rng = np.random.default_rng(107)
    perdidas = rng.standard_normal((20_000, 4)) @ np.diag([1.0, 2.0, 0.5, 1.5])
    contrib = euler_contributions(perdidas, confidence=0.975)
    assert contrib.sums_to_total()
    assert float(np.sum(contrib.percentage)) == pytest.approx(1.0, abs=1e-9)


def test_la_diversificacion_reduce_el_riesgo_agregado():
    rng = np.random.default_rng(109)
    independientes = rng.standard_normal((20_000, 5))
    contrib = euler_contributions(independientes, confidence=0.975)
    assert contrib.total < contrib.sum_of_standalone
    assert contrib.diversification_benefit > 0.0


def test_una_posicion_correlacionada_con_la_cola_contribuye_mas():
    """Lo que mide Euler no es volatilidad, es coincidencia con la cola.

    Dos posiciones con la misma volatilidad individual: una correlacionada con
    el resto del portafolio y otra independiente. La primera contribuye mas,
    aunque por separado sean indistinguibles.
    """
    rng = np.random.default_rng(113)
    n = 40_000
    factor = rng.standard_normal(n)
    nucleo = np.column_stack([factor + 0.3 * rng.standard_normal(n) for _ in range(3)])
    correlacionada = factor + 0.3 * rng.standard_normal(n)
    independiente = rng.standard_normal(n) * float(np.std(correlacionada))
    perdidas = np.column_stack([nucleo, correlacionada, independiente])

    contrib = euler_contributions(perdidas, confidence=0.975)
    # Volatilidades individuales practicamente iguales...
    assert contrib.standalone[3] == pytest.approx(contrib.standalone[4], rel=0.15)
    # ...pero contribuciones muy distintas.
    assert contrib.contributions[3] > 2.0 * contrib.contributions[4]


def test_la_atribucion_valida_sus_entradas():
    with pytest.raises(ValueError, match="matriz"):
        euler_contributions(np.zeros(500))
    with pytest.raises(ValueError, match="al menos 100 escenarios"):
        euler_contributions(np.zeros((50, 3)))
    with pytest.raises(ValueError, match="weights debe tener"):
        euler_contributions(np.ones((200, 3)), weights=np.ones(5))
