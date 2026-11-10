"""Demostracion: por que filtrar, y por que backtestear.

Construye un cambio de regimen de volatilidad y compara la simulacion historica
simple contra la filtrada con la bateria completa de backtests.

Ejecutar con:

    python examples/backtest_demo.py
"""

from __future__ import annotations

import warnings

import numpy as np

from marketrisk import (
    euler_contributions,
    evt_expected_shortfall,
    evt_var,
    expected_shortfall,
    fit_garch,
    fit_gpd_pot,
    kupiec_pof,
    risk_measures,
    rolling_fhs_backtest,
    run_all_backtests,
    simulate_garch,
    traffic_light,
    value_at_risk,
)

CONF = 0.99
ES_CONF = 0.975
VENTANA = 500


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 76}\n{titulo}\n{'=' * 76}")


def demo_coherencia() -> None:
    seccion("Por que FRTB cambio de VaR a Expected Shortfall")
    print("Dos bonos independientes, 4 % de probabilidad de impago, perdida 100.")
    print("Al 95 % de confianza cada impago individual queda FUERA de la cola\n")
    rng = np.random.default_rng(97)
    n = 400_000
    a = (rng.random(n) < 0.04) * 100.0
    b = (rng.random(n) < 0.04) * 100.0

    va, vb, vab = (value_at_risk(x, 0.95) for x in (a, b, a + b))
    ea, eb, eab = (expected_shortfall(x, 0.95) for x in (a, b, a + b))

    print(f"  VaR(A) = {va:7.2f}   VaR(B) = {vb:7.2f}   VaR(A+B) = {vab:7.2f}")
    print(f"  VaR(A) + VaR(B) = {va + vb:.2f}  <  VaR(A+B) = {vab:.2f}")
    print("  -> el VaR dice que DIVERSIFICAR aumenta el riesgo. La medida esta mal.\n")
    print(f"  ES(A) = {ea:7.2f}    ES(B) = {eb:7.2f}    ES(A+B) = {eab:7.2f}")
    print(f"  ES(A) + ES(B) = {ea + eb:.2f}  >=  ES(A+B) = {eab:.2f}")
    print("  -> el ES es subaditivo: la diversificacion reduce el riesgo medido.")
    print("\nNota sobre el estimador: la media condicional E[L | L >= VaR] daria")
    print(f"  ES(A) = {float(a[a >= va].mean()):.2f} en lugar de {ea:.2f}. Con VaR = 0 la condicion")
    print("  incluye toda la muestra. Un error de factor 20, y rompe la subaditividad.")


def demo_fhs() -> None:
    seccion("Simulacion historica simple frente a filtrada, ante un cambio de regimen")
    baja = simulate_garch(1_300, omega=1.0e-6, alpha=0.06, beta=0.90, seed=1)
    alta = simulate_garch(900, omega=2.5e-5, alpha=0.10, beta=0.85, seed=2)
    r = np.concatenate([baja, alta])
    print(f"volatilidad diaria: tramo 1 = {baja.std():.4f}, tramo 2 = {alta.std():.4f} "
          f"(salto de {alta.std() / baja.std():.1f}x)\n")

    rf = rolling_fhs_backtest(
        r, window=VENTANA, confidence=CONF, es_confidence=ES_CONF,
        n_simulations=3_000, seed=7, refit_every=25,
    )
    real = rf.realized_losses
    hs_var = np.array([np.quantile(-r[i : i + VENTANA], CONF) for i in range(len(r) - VENTANA)])
    hs_esvar = np.array(
        [np.quantile(-r[i : i + VENTANA], ES_CONF) for i in range(len(r) - VENTANA)]
    )
    hs_es = np.array(
        [
            risk_measures(-r[i : i + VENTANA], ES_CONF).expected_shortfall
            for i in range(len(r) - VENTANA)
        ]
    )

    for nombre, v, ev, e in (
        ("Historica SIMPLE", hs_var, hs_esvar, hs_es),
        ("Historica FILTRADA", rf.var_forecast, rf.es_var_forecast, rf.es_forecast),
    ):
        exc = int(np.sum(real > v))
        print(f"--- {nombre} ---")
        print(f"  excepciones: {exc}/{real.size} = {exc / real.size:.2%} "
              f"(esperado {1 - CONF:.2%})   semaforo de Basilea: {traffic_light(exc, real.size, CONF).upper()}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for res in run_all_backtests(
                real, v, e, es_var_forecast=ev, confidence=CONF, es_confidence=ES_CONF
            ):
                print(f"  {res}")
        print()

    k_fhs = kupiec_pof(real, rf.var_forecast, CONF).statistic
    k_hs = kupiec_pof(real, hs_var, CONF).statistic
    print(f"Estadistico de Kupiec: filtrada {k_fhs:.2f} contra simple {k_hs:.2f} "
          f"-> reduccion de {k_hs / k_fhs:.1f}x")


def demo_clustering() -> None:
    seccion("Lo que Kupiec no puede ver")
    print("Serie con la frecuencia de excepciones EXACTA (1 %) pero agrupadas en")
    print("cuatro rachas de cinco dias, que es como se comportan en una crisis.\n")
    n = 2_000
    var = np.ones(n)
    perdidas = np.zeros(n)
    for inicio in (300, 700, 1_200, 1_700):
        perdidas[inicio : inicio + 5] = 2.0
    for res in run_all_backtests(perdidas, var, confidence=CONF):
        print(f"  {res}")
    print("\nKupiec aprueba el modelo. La prueba de independencia lo rechaza.")
    print("Reportar solo Kupiec deja pasar exactamente el fallo que importa.")


def demo_evt() -> None:
    seccion("Teoria de valores extremos: extrapolar mas alla del dato mas extremo")
    r = simulate_garch(8_000, omega=3e-6, alpha=0.10, beta=0.88, seed=5, innovation_df=5.0)
    perdidas = np.abs(r)
    ajuste = fit_gpd_pot(perdidas, threshold_quantile=0.95)
    print(f"  {ajuste}")
    print(f"  parametro de forma xi = {ajuste.shape:.4f}")
    print(f"  varianza finita (xi < 0.5): {ajuste.has_finite_variance}")
    print(f"  media finita   (xi < 1.0): {ajuste.has_finite_mean}")
    print(f"\n  maxima perdida observada: {perdidas.max():.6f}\n")
    print(f"{'nivel':>10} {'VaR empirico':>15} {'VaR EVT':>12} {'ES EVT':>12}")
    for c in (0.99, 0.999, 0.9999):
        emp = float(np.quantile(perdidas, c))
        print(f"{c:>10.4%} {emp:>15.6f} {evt_var(ajuste, c):>12.6f} "
              f"{evt_expected_shortfall(ajuste, c):>12.6f}")
    print("\nAl 99.99 % el cuantil empirico es un dato suelto; el de EVT se apoya")
    print("en la forma funcional de la cola, que el teorema de Pickands justifica.")


def demo_atribucion() -> None:
    seccion("Atribucion de riesgo: que posicion causa la cola")
    rng = np.random.default_rng(113)
    n = 40_000
    factor = rng.standard_normal(n)
    nucleo = np.column_stack([factor + 0.3 * rng.standard_normal(n) for _ in range(3)])
    correlacionada = factor + 0.3 * rng.standard_normal(n)
    independiente = rng.standard_normal(n) * float(np.std(correlacionada))
    perdidas = np.column_stack([nucleo, correlacionada, independiente])

    c = euler_contributions(perdidas, confidence=ES_CONF)
    nombres = ["nucleo 1", "nucleo 2", "nucleo 3", "correlacionada", "independiente"]
    print(f"{'posicion':>16} {'riesgo propio':>15} {'contribucion':>14} {'% del total':>12}")
    for nombre, ind, contrib, pct in zip(nombres, c.standalone, c.contributions, c.percentage):
        print(f"{nombre:>16} {ind:>15.4f} {contrib:>14.4f} {pct:>11.1%}")
    print(f"\n  ES del portafolio          : {c.total:.4f}")
    print(f"  suma de riesgos propios    : {c.sum_of_standalone:.4f}")
    print(f"  beneficio de diversificacion: {c.diversification_benefit:.1%}")
    print(f"  las contribuciones suman el total: {c.sums_to_total()}")
    print("\nLas dos ultimas posiciones tienen riesgo propio casi identico, pero la")
    print("correlacionada contribuye varias veces mas: Euler mide coincidencia con")
    print("la cola del portafolio, no volatilidad individual.")


def demo_garch() -> None:
    seccion("GARCH: recuperacion de parametros y escalado")
    verdad = dict(omega=5.0e-6, alpha=0.15, beta=0.80)
    print(f"verdad: omega={verdad['omega']:.1e}, alpha={verdad['alpha']}, beta={verdad['beta']}\n")
    print(f"{'n':>8} {'omega':>12} {'alpha':>9} {'beta':>9} {'persistencia':>14}")
    for n in (2_000, 8_000, 24_000):
        f = fit_garch(simulate_garch(n, seed=17, **verdad))
        print(f"{n:>8} {f.omega:>12.3e} {f.alpha:>9.4f} {f.beta:>9.4f} {f.persistence:>14.4f}")

    print("\nInvariancia de escala (prueba de la correccion interna de escalado):")
    r = simulate_garch(6_000, seed=31, **verdad)
    base = fit_garch(r)
    for factor in (0.01, 1.0, 100.0):
        f = fit_garch(r * factor)
        print(f"  serie x{factor:<7} alpha={f.alpha:.6f} beta={f.beta:.6f} "
              f"omega/factor^2={f.omega / factor**2:.4e}")
    print("  alpha y beta son invariantes de escala; omega escala con la varianza.")
    print("  Sin normalizar la serie antes de optimizar, esta invariancia se rompe.")


def main() -> None:
    demo_coherencia()
    demo_garch()
    demo_fhs()
    demo_clustering()
    demo_evt()
    demo_atribucion()


if __name__ == "__main__":
    main()
