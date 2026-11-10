# market-risk-var-es

VaR y Expected Shortfall por simulación histórica filtrada con GARCH y
DCC-GARCH, teoría de valores extremos para la cola, y **suite completa de
backtesting**.

Lo que distingue este repositorio de un calculador de cuantiles es lo último.
Un número de VaR sin backtest es un número sin respaldo, y es exactamente lo que
el diseño original de este proyecto no contemplaba.

```python
from marketrisk import rolling_fhs_backtest, run_all_backtests, traffic_light

rf = rolling_fhs_backtest(retornos, window=500, confidence=0.99)
for res in run_all_backtests(rf.realized_losses, rf.var_forecast, rf.es_forecast,
                             es_var_forecast=rf.es_var_forecast):
    print(res)
print(traffic_light(rf.n_exceptions, len(rf), 0.99))
```

---

## Los tres resultados que justifican el diseño

### 1. Por qué Expected Shortfall y no VaR

Dos bonos independientes, 4 % de probabilidad de impago, pérdida 100, al 95 % de
confianza:

| | A | B | A+B |
|---|---|---|---|
| VaR(95 %) | **0.00** | **0.00** | **100.00** |
| ES(95 %) | 80.02 | 80.45 | 103.25 |

`VaR(A) + VaR(B) = 0 < VaR(A+B) = 100`. El VaR dice que **diversificar aumenta
el riesgo**. No es un artefacto numérico: la medida no es subaditiva, y por eso
Basilea III / FRTB la sustituyó por el ES al 97.5 %.

El ES sí cumple `ES(A+B) = 103.25 ≤ ES(A) + ES(B) = 160.47`.

**Y aquí hay una trampa de implementación que el repositorio documenta porque la
encontró su propia batería de pruebas.** El estimador habitual del ES —la media
de las observaciones que superan el VaR— es **incorrecto** cuando la distribución
tiene átomos. En este ejemplo `VaR = 0`, así que la condición `L ≥ 0` incluye
*toda* la muestra y ese estimador devuelve **4.00** en lugar de 80: un error de
factor 20, en la medida que determina el capital regulatorio. Peor aún, con ese
estimador la subaditividad se rompe (`ES(A+B) = 102 > ES(A) + ES(B) = 8`), es
decir se destruye la propiedad por la que se eligió el ES.

El estimador correcto promedia la peor fracción `1−α` repartiendo el átomo que
cae sobre el cuantil. Para distribuciones continuas coincide exactamente con la
media condicional (verificado a 6 decimales); para distribuciones con átomos, no.

### 2. Por qué filtrar: simulación histórica simple contra filtrada

Serie con un salto de volatilidad de **4×** a mitad del recorrido (0.55 % diaria
→ 2.19 %), ventana de 500 días, VaR al 99 %:

| | Histórica simple | Histórica **filtrada** |
|---|---|---|
| Excepciones | 50/1700 = **2.94 %** | 27/1700 = **1.59 %** |
| Esperadas | 1.00 % | 1.00 % |
| Semáforo de Basilea | **ROJA** | AMARILLA |
| Kupiec | 42.53 (p = 0.0000) | 5.04 (p = 0.0248) |
| Christoffersen condicional | 42.71 (p = 0.0000) | 5.91 (p = 0.0520, **no rechaza**) |
| Acerbi-Szekely Z1 | rechaza | **no rechaza** |

El estadístico de Kupiec baja **8.4×**. La simulación simple arrastra la ventana
completa, así que subestima el riesgo justo cuando aumenta; la filtrada divide
por la volatilidad condicional, remuestrea los residuos y reescala por la
volatilidad de **hoy**, con lo que se adapta en días.

Nótese lo que **no** se afirma: la filtrada sigue siendo rechazada por Kupiec.
Un salto de 4× con ventana de 500 días sigue siendo exigente. Mejora mucho; no
resuelve.

### 3. Lo que Kupiec no puede ver

Serie construida con la frecuencia de excepciones **exacta** (1 %) pero agrupadas
en cuatro rachas de cinco días, que es como se comportan en una crisis:

| Prueba | Estadístico | p-valor | Veredicto |
|---|---|---|---|
| Kupiec (cobertura incondicional) | −0.0000 | **1.0000** | **aprueba** |
| Christoffersen (independencia) | **146.35** | **0.0000** | **rechaza** |
| Christoffersen (cobertura condicional) | 146.35 | 0.0000 | rechaza |

Probabilidad de excepción tras una excepción: **80 %**. Sin excepción previa:
**0.2 %**.

Kupiec aprueba el modelo con un p-valor de 1. Reportar solo Kupiec deja pasar
exactamente el fallo que importa, y es la razón por la que las dos pruebas van
siempre juntas.

---

## Qué hace

| Módulo | Contenido |
|---|---|
| `marketrisk.garch` | GARCH(1,1) y DCC-GARCH por cuasi-máxima verosimilitud, con multiarranque |
| `marketrisk.fhs` | Simulación histórica filtrada y backtest con ventana móvil fuera de muestra |
| `marketrisk.evt` | Picos sobre umbral, Pareto generalizada, función de exceso medio |
| `marketrisk.measures` | VaR y ES con el estimador integral correcto; escalado de horizonte |
| `marketrisk.backtesting` | Kupiec, Christoffersen, Acerbi-Szekely Z1 y Z2, semáforo de Basilea |
| `marketrisk.attribution` | Contribuciones al riesgo por asignación de Euler |

### Validación

55 pruebas. Reproducible con `python examples/backtest_demo.py`.

| Propiedad verificada | Resultado |
|---|---|
| GARCH recupera α=0.15, β=0.80 (n=8000) | 0.1509, 0.8025 |
| El error de estimación decrece con n (promediado sobre 5 semillas) | monótono |
| **Invariancia de escala del ajuste** | α y β idénticos a 6 decimales con la serie ×0.01 y ×100 |
| QML consistente con innovaciones t de Student | sí |
| `ES ≥ VaR` en normal, t de Student y exponencial | siempre |
| ES subaditivo; VaR no (contraejemplo explícito) | confirmado |
| `ES/VaR` de una normal al 99 % | 1.146 (teórico 1.146) |
| GPD recupera ξ = 0.10, 0.25, 0.40 | a ±0.06 |
| `ξ ≥ 1` ⇒ el ES es infinito | lanza excepción |
| Las contribuciones de Euler suman el total | exacto |
| Backtests **rechazan** modelos mal calibrados | sí (potencia verificada) |

Un backtest tiene que cumplir dos cosas para servir: no rechazar un modelo
correcto **y rechazar uno incorrecto**. Las dos se prueban; comprobar solo la
primera deja pasar una prueba sin potencia.

---

## Qué no hace

- **No calcula el capital regulatorio.** Entrega el ES al 97.5 %, que es el
  insumo, no el requerimiento. FRTB exige además agregar por horizontes de
  liquidez, separar factores modelizables de no modelizables (NMRF), el
  *stressed ES*, y el régimen de pruebas a nivel de mesa con PLAT. Eso es un
  proyecto de implementación regulatoria, no una librería de medidas.
- **No implementa PLAT** (*P&L attribution test*), que compara el P&L teórico
  del modelo de riesgo con el del front office. Requiere los dos sistemas.
- **No hay descomposición por factores de riesgo.** La atribución es por
  posición, no por delta de tasa, vega o spread de crédito.
- **DCC es de dos etapas y no corrige los errores estándar.** La estimación en
  dos pasos es consistente pero los errores estándar del segundo paso ignoran la
  incertidumbre del primero. Para inferencia sobre `a` y `b` haría falta la
  corrección de dos etapas; para generar escenarios, que es el uso aquí, no es
  necesaria.
- **Sin cópulas ni modelos de dependencia de cola.** DCC captura correlación
  condicional lineal; la dependencia de cola necesita otra familia.
- **Sin datos de mercado.** Los ejemplos usan series simuladas, lo que hace la
  validación reproducible y verificable, a cambio de no exhibir un caso real.
- **El escalado de horizonte está, pero FRTB no escala.** Se incluye para
  cuantificar el error de la regla de la raíz del tiempo, no como sustituto de
  simular directamente a cada horizonte de liquidez.

---

## Alcance y decisiones de diseño

**Las pérdidas son positivas en todo el módulo.** Es la convención de gestión de
riesgos y evita el error de signo más frecuente del área.

**El GARCH se estima sobre la serie normalizada a varianza unitaria.** No es
cosmético: en retornos diarios `ω ≈ 1e-6` mientras `α` y `β` son del orden de
`0.1` y `0.9`, cuatro órdenes de magnitud de diferencia. Un optimizador que
estima el gradiente por diferencias finitas con paso único no maneja esa
disparidad y **devuelve el punto de arranque disfrazado de óptimo**. Es un fallo
silencioso y traicionero: si el arranque está cerca de los valores verdaderos, el
ajuste parece excelente. La prueba de invariancia de escala existe precisamente
para detectarlo.

**Los backtests de ES exigen el VaR al mismo nivel que el ES.** Acerbi-Szekely
define las excepciones con el VaR de nivel α y compara su magnitud contra el ES
del mismo α. Pasar un VaR al 99 % con un ES al 97.5 % —la combinación natural si
se toman los valores por omisión de cada uno— produce un diagnóstico en el que Z1
y Z2 se contradicen. Como la coherencia de niveles no se puede verificar desde
los vectores, se comprueba la condición que sí es observable (`ES ≥ VaR` punto
por punto) y se lanza una excepción explicativa si falla. `rolling_fhs_backtest`
devuelve **dos** vectores de VaR por esta razón.

**El backtest usa ventana móvil con reestimación fuera de muestra.** Estimar con
toda la serie y luego «backtestear» sobre ella produce resultados inservibles, y
es el error más común del área. Entre reestimaciones se reutilizan los parámetros
y solo se refiltra la varianza, que es lo que de verdad mueve el pronóstico.

**El escalado de horizonte exige la varianza actual y la de largo plazo, no solo
la persistencia.** Con la persistencia sola el problema está indeterminado: no se
sabe de qué nivel se parte, y el signo del sesgo de la regla de la raíz del tiempo
depende precisamente de eso. Si la volatilidad de hoy está por debajo de su nivel
de largo plazo, la raíz del tiempo subestima; si está por encima, sobreestima.

**`ξ ≥ 1` lanza una excepción en lugar de devolver un número.** Si el parámetro de
forma de la Pareto generalizada alcanza 1, la media de la cola no existe y el ES
es infinito. Devolver un valor ahí sería devolver una ficción.

**El repositorio es autocontenido.** No depende de una librería núcleo compartida
con los demás del conjunto.

---

## Marco regulatorio

- **Basilea III / FRTB** (BCBS d457, *Minimum capital requirements for market
  risk*): sustituye el VaR al 99 % a 10 días por el **Expected Shortfall al
  97.5 %** con horizontes de liquidez escalados, y exige backtesting y PLAT a
  nivel de mesa. Los valores por omisión del módulo reflejan este marco.
- **BCBS (1996)**, marco de backtesting y semáforo de tres zonas.
- Para México, la **CUB de la CNBV** en materia de administración de riesgos de
  mercado y validación de modelos internos.

---

## Instalación

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python examples/backtest_demo.py
```

Requiere Python 3.11 o superior, NumPy, SciPy y pandas.

---

## Referencias

- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). *Coherent measures of
  risk*. Mathematical Finance 9(3).
- Acerbi, C. & Tasche, D. (2002). *On the coherence of expected shortfall*.
  Journal of Banking & Finance 26(7). Contiene el estimador integral correcto.
- Acerbi, C. & Szekely, B. (2014). *Backtesting expected shortfall*. Risk
  Magazine, diciembre.
- Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). *VaR without
  correlations for portfolios of derivative securities*. Journal of Futures
  Markets 19(5).
- Bollerslev, T. (1986). *Generalized autoregressive conditional
  heteroskedasticity*. Journal of Econometrics 31(3).
- Bollerslev, T. & Wooldridge, J. (1992). *Quasi-maximum likelihood estimation
  and inference in dynamic models with time-varying covariances*. Econometric
  Reviews 11(2).
- Engle, R. (2002). *Dynamic conditional correlation*. Journal of Business &
  Economic Statistics 20(3).
- Kupiec, P. (1995). *Techniques for verifying the accuracy of risk measurement
  models*. Journal of Derivatives 3(2).
- Christoffersen, P. (1998). *Evaluating interval forecasts*. International
  Economic Review 39(4).
- McNeil, A. & Frey, R. (2000). *Estimation of tail-related risk measures for
  heteroscedastic financial time series: an extreme value approach*. Journal of
  Empirical Finance 7(3-4).
- Tasche, D. (1999). *Risk contributions and performance measurement*. Working
  paper, TU München.
- BCBS (2019). *Minimum capital requirements for market risk* (d457).

## Licencia

MIT. Ver [LICENSE](LICENSE).
