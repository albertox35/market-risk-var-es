"""Validacion del backtesting, de la EVT y de la simulacion historica filtrada.

Un backtest tiene que cumplir dos cosas para servir: **no rechazar** un modelo
correcto y **rechazar** uno incorrecto. Las dos se prueban aqui; comprobar solo
la primera deja pasar una prueba sin potencia.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto, norm

from marketrisk import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    evt_expected_shortfall,
    evt_var,
    filtered_historical_simulation,
    fit_gpd_pot,
    kupiec_pof,
    mean_excess_plot,
    rolling_fhs_backtest,
    run_all_backtests,
    simulate_garch,
    traffic_light,
)

CONF = 0.99
N = 2_500


def serie_normal_con_var_correcto(n: int = N, seed: int = 11):
    """Perdidas normales con el VaR exacto del modelo: el caso bien calibrado."""
    rng = np.random.default_rng(seed)
    sigma = 0.01
    perdidas = rng.standard_normal(n) * sigma
    var = np.full(n, float(norm.ppf(CONF) * sigma))
    return perdidas, var


# ------------------------------------------------------------------- Kupiec


def test_kupiec_no_rechaza_un_modelo_bien_calibrado():
    perdidas, var = serie_normal_con_var_correcto()
    assert not kupiec_pof(perdidas, var, CONF).reject


def test_kupiec_rechaza_un_var_demasiado_optimista():
    """Potencia de la prueba: un VaR al 30 % de su valor correcto."""
    perdidas, var = serie_normal_con_var_correcto()
    res = kupiec_pof(perdidas, var * 0.3, CONF)
    assert res.reject
    assert res.p_value < 1e-6


def test_kupiec_rechaza_un_var_demasiado_conservador():
    """Un modelo excesivamente prudente tambien esta mal especificado."""
    perdidas, var = serie_normal_con_var_correcto()
    assert kupiec_pof(perdidas, var * 5.0, CONF).reject


def test_kupiec_maneja_el_caso_de_cero_excepciones():
    perdidas, var = serie_normal_con_var_correcto()
    res = kupiec_pof(perdidas, var * 100.0, CONF)
    assert "0 excepciones" in res.detail
    assert np.isfinite(res.statistic)


def test_el_backtest_exige_muestra_suficiente():
    with pytest.raises(ValueError, match="al menos 20"):
        kupiec_pof(np.zeros(10), np.ones(10), CONF)


def test_el_backtest_exige_formas_compatibles():
    with pytest.raises(ValueError, match="misma forma"):
        kupiec_pof(np.zeros(100), np.ones(50), CONF)


# ----------------------------------------------------------- Christoffersen


def test_christoffersen_no_rechaza_excepciones_independientes():
    perdidas, var = serie_normal_con_var_correcto()
    assert not christoffersen_independence(perdidas, var).reject


def test_christoffersen_detecta_el_agrupamiento_de_excepciones():
    """El fallo que Kupiec no puede ver.

    Se construye una serie con la frecuencia de excepciones **correcta** pero
    todas concentradas en bloques consecutivos, que es lo que pasa en una
    crisis. Kupiec la aprueba; la prueba de independencia la rechaza.
    """
    n = 2_000
    var = np.ones(n)
    perdidas = np.zeros(n)
    # 20 excepciones (1 %, exactamente lo esperado) en cuatro rachas de cinco.
    for inicio in (300, 700, 1_200, 1_700):
        perdidas[inicio : inicio + 5] = 2.0

    kupiec = kupiec_pof(perdidas, var, CONF)
    independencia = christoffersen_independence(perdidas, var)
    condicional = christoffersen_conditional_coverage(perdidas, var, CONF)

    assert not kupiec.reject, "la frecuencia es la correcta, Kupiec no debe rechazar"
    assert independencia.reject, "el agrupamiento debe detectarse"
    assert condicional.reject, "la prueba conjunta debe rechazar por la via de independencia"


def test_la_cobertura_condicional_es_la_suma_de_las_dos_componentes():
    perdidas, var = serie_normal_con_var_correcto()
    uc = kupiec_pof(perdidas, var, CONF)
    ind = christoffersen_independence(perdidas, var)
    cc = christoffersen_conditional_coverage(perdidas, var, CONF)
    assert cc.statistic == pytest.approx(uc.statistic + ind.statistic)


def test_christoffersen_avisa_cuando_no_tiene_transiciones_suficientes():
    """Sin excepciones la cadena de Markov no tiene transiciones que estimar.

    No rechazar es lo correcto, pero el resultado tiene que decir que la prueba
    no es informativa, en lugar de presentar un p-valor de 1 como si el modelo
    hubiera sido validado.
    """
    n = 500
    perdidas = np.zeros(n)
    var = np.ones(n)  # ninguna perdida supera el VaR
    res = christoffersen_independence(perdidas, var)
    assert not res.reject
    assert "insuficientes" in res.detail
    assert "no es informativa" in res.detail


# --------------------------------------------------------- Acerbi-Szekely


def test_los_backtests_de_es_rechazan_niveles_incoherentes():
    """La proteccion contra el error de cableado mas comun.

    Si se pasa un VaR al 99 % con un ES al 97.5 %, el ES puede resultar menor
    que el VaR, lo que es imposible para un par coherente. En lugar de producir
    un diagnostico sin sentido, se lanza una excepcion.
    """
    perdidas, var = serie_normal_con_var_correcto()
    es_mas_bajo = var * 0.9  # imposible: el ES de un nivel nunca es menor que su VaR
    with pytest.raises(ValueError, match="niveles de confianza distintos"):
        acerbi_szekely_z1(perdidas, var, es_mas_bajo)


def test_run_all_backtests_exige_el_var_al_nivel_del_es():
    perdidas, var = serie_normal_con_var_correcto()
    with pytest.raises(ValueError, match="es_var_forecast"):
        run_all_backtests(perdidas, var, es_forecast=var * 1.2, confidence=CONF)


def test_z1_y_z2_no_rechazan_un_modelo_de_es_bien_calibrado():
    """ES exacto de una normal, calculado analiticamente."""
    rng = np.random.default_rng(13)
    n, sigma, alfa = 4_000, 0.01, 0.975
    perdidas = rng.standard_normal(n) * sigma
    var = float(norm.ppf(alfa)) * sigma
    # ES de una normal: sigma * phi(z_alfa) / (1 - alfa)
    es = sigma * float(norm.pdf(norm.ppf(alfa))) / (1.0 - alfa)
    assert not acerbi_szekely_z1(perdidas, np.full(n, var), np.full(n, es)).reject
    assert not acerbi_szekely_z2(perdidas, np.full(n, var), np.full(n, es), alfa).reject


def test_z2_rechaza_un_es_subestimado():
    """Potencia de la prueba de ES."""
    rng = np.random.default_rng(17)
    n, sigma, alfa = 4_000, 0.01, 0.975
    perdidas = rng.standard_normal(n) * sigma
    var = float(norm.ppf(alfa)) * sigma
    es_correcto = sigma * float(norm.pdf(norm.ppf(alfa))) / (1.0 - alfa)
    # Un ES que apenas supera el VaR: subestima la cola.
    es_malo = max(var * 1.001, es_correcto * 0.6)
    res = acerbi_szekely_z2(perdidas, np.full(n, var), np.full(n, es_malo), alfa)
    assert res.reject
    assert res.statistic > 0.0


def test_z1_informa_cuando_no_hay_excepciones():
    perdidas, var = serie_normal_con_var_correcto()
    res = acerbi_szekely_z1(perdidas, var * 100.0, var * 200.0)
    assert not res.reject
    assert "sin excepciones" in res.detail


# -------------------------------------------------------- semaforo Basilea


def test_las_zonas_del_semaforo_estan_bien_delimitadas():
    assert traffic_light(2, 250, 0.99) == "verde"
    assert traffic_light(6, 250, 0.99) == "amarilla"
    assert traffic_light(15, 250, 0.99) == "roja"


def test_el_semaforo_valida_sus_entradas():
    with pytest.raises(ValueError, match="fuera de rango"):
        traffic_light(300, 250, 0.99)


# ----------------------------------------------------------------------- EVT


def test_la_gpd_recupera_el_parametro_de_forma_simulado():
    """Contraste contra datos generados por la propia distribucion."""
    for xi_verdadero in (0.1, 0.25, 0.4):
        muestra = genpareto.rvs(c=xi_verdadero, scale=0.02, size=20_000, random_state=19)
        ajuste = fit_gpd_pot(muestra, threshold_quantile=0.90)
        assert ajuste.shape == pytest.approx(xi_verdadero, abs=0.06)


def test_el_var_de_evt_crece_con_el_nivel_de_confianza():
    muestra = genpareto.rvs(c=0.2, scale=0.02, size=20_000, random_state=23)
    ajuste = fit_gpd_pot(muestra, threshold_quantile=0.95)
    valores = [evt_var(ajuste, c) for c in (0.99, 0.995, 0.999, 0.9999)]
    assert all(b > a for a, b in zip(valores, valores[1:])), valores


def test_el_es_de_evt_supera_al_var():
    muestra = genpareto.rvs(c=0.2, scale=0.02, size=20_000, random_state=29)
    ajuste = fit_gpd_pot(muestra, threshold_quantile=0.95)
    for c in (0.99, 0.999):
        assert evt_expected_shortfall(ajuste, c) > evt_var(ajuste, c)


def test_evt_rechaza_niveles_por_debajo_del_umbral():
    """Fuera del rango de extrapolacion hay que usar el cuantil empirico."""
    muestra = genpareto.rvs(c=0.2, scale=0.02, size=5_000, random_state=31)
    ajuste = fit_gpd_pot(muestra, threshold_quantile=0.95)
    with pytest.raises(ValueError, match="por debajo del umbral"):
        evt_var(ajuste, 0.90)


def test_un_parametro_de_forma_mayor_que_uno_hace_infinito_el_es():
    """Consecuencia que suele pasarse por alto.

    Si ``xi >= 1`` la media de la cola no existe, asi que el Expected Shortfall
    es infinito. Devolver un numero ahi seria devolver una ficcion.
    """
    muestra = genpareto.rvs(c=1.4, scale=0.02, size=20_000, random_state=37)
    ajuste = fit_gpd_pot(muestra, threshold_quantile=0.90)
    assert not ajuste.has_finite_mean
    with pytest.raises(ValueError, match="no existe"):
        evt_expected_shortfall(ajuste, 0.999)


def test_las_banderas_de_momentos_finitos_son_coherentes_con_xi():
    for xi, varianza, media in ((0.2, True, True), (0.7, False, True), (1.3, False, False)):
        muestra = genpareto.rvs(c=xi, scale=0.02, size=20_000, random_state=41)
        ajuste = fit_gpd_pot(muestra, threshold_quantile=0.90)
        assert ajuste.has_finite_variance == varianza
        assert ajuste.has_finite_mean == media


def test_el_exceso_medio_es_creciente_para_una_cola_de_ley_de_potencia():
    """Diagnostico de umbral: con ``xi > 0`` el exceso medio crece linealmente."""
    muestra = genpareto.rvs(c=0.3, scale=0.02, size=20_000, random_state=43)
    umbrales, medias = mean_excess_plot(muestra, n_thresholds=20, max_quantile=0.98)
    # La tendencia global debe ser creciente.
    pendiente = np.polyfit(umbrales, medias, 1)[0]
    assert pendiente > 0.0


def test_evt_exige_excesos_suficientes():
    muestra = genpareto.rvs(c=0.2, scale=0.02, size=200, random_state=47)
    with pytest.raises(ValueError, match="al menos 20"):
        fit_gpd_pot(muestra, threshold_quantile=0.995)


# ----------------------------------------------------------------------- FHS


def test_la_fhs_produce_una_distribucion_coherente():
    r = simulate_garch(1_500, omega=3e-6, alpha=0.10, beta=0.88, seed=53)
    fhs = filtered_historical_simulation(r, horizon=1, n_simulations=20_000, seed=59)
    for c in (0.95, 0.975, 0.99):
        m = fhs.measures(c)
        assert m.expected_shortfall >= m.var > 0.0
    # Niveles mas altos dan medidas mas grandes.
    assert fhs.measures(0.99).var > fhs.measures(0.95).var


def test_la_fhs_es_reproducible():
    r = simulate_garch(1_000, omega=3e-6, alpha=0.10, beta=0.88, seed=61)
    a = filtered_historical_simulation(r, n_simulations=5_000, seed=67)
    b = filtered_historical_simulation(r, n_simulations=5_000, seed=67)
    np.testing.assert_array_equal(a.simulated_losses, b.simulated_losses)


def test_el_riesgo_de_la_fhs_crece_con_el_horizonte():
    r = simulate_garch(1_500, omega=3e-6, alpha=0.10, beta=0.88, seed=71)
    uno = filtered_historical_simulation(r, horizon=1, n_simulations=20_000, seed=73)
    diez = filtered_historical_simulation(r, horizon=10, n_simulations=20_000, seed=73)
    assert diez.measures(0.99).var > uno.measures(0.99).var


def test_la_fhs_supera_a_la_simulacion_historica_simple_ante_un_cambio_de_regimen():
    """La comparacion que justifica filtrar.

    Se construye un salto de volatilidad de 4x a mitad de la serie. La
    simulacion historica simple arrastra la ventana entera, asi que subestima el
    riesgo justo cuando aumenta; la filtrada reescala por la volatilidad de hoy y
    se adapta en dias.

    Se compara el estadistico de Kupiec, no solo el conteo: mide cuanto se
    desvia la frecuencia de excepciones de la esperada.
    """
    baja = simulate_garch(1_300, omega=1.0e-6, alpha=0.06, beta=0.90, seed=1)
    alta = simulate_garch(900, omega=2.5e-5, alpha=0.10, beta=0.85, seed=2)
    r = np.concatenate([baja, alta])
    ventana = 500

    rf = rolling_fhs_backtest(
        r, window=ventana, confidence=CONF, n_simulations=2_000, seed=7, refit_every=50
    )
    real = rf.realized_losses
    hs_var = np.array([np.quantile(-r[i : i + ventana], CONF) for i in range(len(r) - ventana)])

    kupiec_fhs = kupiec_pof(real, rf.var_forecast, CONF)
    kupiec_hs = kupiec_pof(real, hs_var, CONF)

    # La frecuencia de excepciones de la FHS esta mucho mas cerca de la esperada.
    assert rf.exception_rate < float(np.mean(real > hs_var))
    # Y el estadistico de Kupiec es sustancialmente menor.
    assert kupiec_fhs.statistic < 0.5 * kupiec_hs.statistic
    # La simulacion historica simple queda en zona roja del semaforo.
    n_exc_hs = int(np.sum(real > hs_var))
    assert traffic_light(n_exc_hs, real.size, CONF) == "roja"


def test_la_fhs_valida_sus_entradas():
    r = simulate_garch(600, omega=3e-6, alpha=0.10, beta=0.88, seed=79)
    with pytest.raises(ValueError, match="horizonte"):
        filtered_historical_simulation(r, horizon=0)
    with pytest.raises(ValueError, match="al menos 100 simulaciones"):
        filtered_historical_simulation(r, n_simulations=10)
    with pytest.raises(ValueError, match="se requieren mas de"):
        rolling_fhs_backtest(r, window=590)
