# vgc-leviathan-bot — bot de VGC Reg M-C (Champions) con búsqueda tipo Nessie123

Bot de Pokémon Showdown hecho con **poke-env** + red Transformer pequeña + búsqueda *Double-Oracle*
(equilibrio de Nash por turno) + simulador rápido propio. Código del documento
"Ingeniería de Sistemas de IA para Pokémon VGC", **corregido para que funcione** (ver `CAMBIOS_RESPECTO_AL_DOCX.md`).

## Qué hay en la carpeta

| Archivo | Para qué sirve |
|---|---|
| `teams_reg_mc.py` | Tus 2 equipos (`TEAM_1_REG_MC`, `TEAM_2_REG_MC`) + mutador de Stat Points/objetos |
| `vgc_baseline_agent.py` | Bot simple (tipos + potencia) y prueba de velocidad contra `SimpleHeuristicsPlayer` |
| `nessie_network.py` | Red neuronal (13 tokens, cabeza de valor, políticas, cabezas auxiliares) |
| `double_oracle_search.py` | Solver de Nash (programación lineal), rastreador Bo3 y motor de búsqueda |
| `vgc_state_encoder.py` | Convierte la batalla de poke-env en un estado rápido y en el tensor (1,13,128) |
| `vgc_fast_simulator.py` | Simulador de turnos en memoria + evaluador híbrido (heurística + red) |
| `leviathan_master_bot.py` | **El bot completo**, con panel "coach" y barra de victoria |
| `training_pipeline.py` | Función de pérdida multitarea (⚠ solo la pérdida, ver abajo) |
| `play_vs_me.py` | *Añadido:* el bot espera tus retos para que practiques contra él |
| `check_setup.py` | *Añadido:* comprueba que todo está instalado |
| `requirements.txt` | *Añadido:* dependencias de Python |

## Instalación desde cero

Necesitas: **Git**, **Node.js >= 22.18** (https://nodejs.org) y **Python >= 3.10**. Probado en Linux (Python 3.13, Node 22.22);
en Windows/macOS los comandos son equivalentes, pero no los he podido probar yo.

```bash
# 1) Entra en esta carpeta
cd vgc-leviathan-bot

# 2) Simulador oficial de Showdown
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js      # Windows: copy config\config-example.js config\config.js
cd ..

# 3) Entorno de Python
python3 -m venv .venv                              # Windows: python -m venv .venv
source .venv/bin/activate                          # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Arrancar el bot (necesitas 2 terminales)

**Terminal 1 — servidor (déjala abierta):**
```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```
Debe poner `Worker 1 now listening on 0.0.0.0:8000`.

**Terminal 2 — bot** (con el venv activado, dentro de `vgc-leviathan-bot`):
```bash
python check_setup.py            # debe acabar en "TODO LISTO"
python vgc_baseline_agent.py     # prueba rápida: 20 combates vs bot heurístico (~segundos)
python leviathan_master_bot.py   # Team 1 vs Team 2, Bo3, con panel coach (~2-3 min)
```

## Practicar tú contra el bot (tu objetivo)

```bash
python play_vs_me.py                          # el bot juega TEAM_1
python play_vs_me.py --team 2                 # el bot juega TEAM_2
python play_vs_me.py --team rival.txt         # el bot juega un equipo que pegues en rival.txt
```
Luego abre http://localhost:8000 → pon un nombre → **Teambuilder**, importa tu equipo →
formato **[Gen 9 Champions] VGC 2026 Reg M-C** → reta al usuario **LeviathanBot** y acepta las hojas de equipo abiertas.
En la terminal verás la barra de victoria y el % de cada jugada del equilibrio de Nash (solo los 8 primeros turnos).

**Poner tus equipos:** edita `TEAM_1_REG_MC` / `TEAM_2_REG_MC` en `teams_reg_mc.py` (formato Showdown; en Champions la línea `EVs:`
son *Stat Points*: máx. 32 por stat y 66 en total). Valida un equipo con:
`cd pokemon-showdown && node pokemon-showdown pack-team < equipo.txt | node pokemon-showdown validate-team gen9championsvgc2026regmc`

## Problemas típicos
* **El combate no empieza / se queda colgado:** los dos jugadores deben tener `accept_open_team_sheet=True`.
* **`ModuleNotFoundError: poke_env.environment`:** tienes código antiguo del docx; usa los archivos de esta carpeta.
* **`Connection refused`:** el servidor de la Terminal 1 no está arrancado.
* **Avisos `CRITICAL ... open team sheets` en consola:** son inofensivos (poke-env responde dos veces); el combate sigue.
* **Va lento:** baja `inference_budget` (p. ej. 80). Cada turno tarda ~1,3 s con 180.

## Estado real (importante)
Lo que ya funciona: servidor, conexión, team preview, búsqueda Nash con simulador rápido, Mega-Evolución, panel coach, Bo3, modo práctica.
Lo que **todavía no existe** y hace falta para que "mejore sola": bucle de auto-juego + entrenamiento + guardado de pesos (`training_pipeline.py`
solo trae la pérdida), y conectar `Bo3OpponentTracker` al bot. Sin eso, la red tiene pesos aleatorios y el bot juega con la heurística + búsqueda.
