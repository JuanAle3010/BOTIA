# Qué he cambiado respecto al código del docx (y por qué)

El código del docx **no arrancaba tal cual**. Lo probé contra un servidor Showdown local y
poke-env 0.16.1 y estos fueron los fallos. Todo lo demás está tal cual viene en el documento.

| # | Archivo | Problema del docx | Arreglo |
|---|---------|-------------------|---------|
| 1 | `vgc_baseline_agent.py`, `leviathan_master_bot.py` | Formato `gen9vgc2026regmc`: **no existe** en Showdown | `gen9championsvgc2026regmc` (el real, mod "champions") |
| 2 | `vgc_baseline_agent.py`, `vgc_state_encoder.py`, `leviathan_master_bot.py` | `from poke_env.environment.…` — ese módulo ya no existe | `from poke_env.battle.…` |
| 3 | `vgc_baseline_agent.py` | `battle._data.type_chart` no existe | `GenData.from_gen(battle.gen).type_chart` |
| 4 | `vgc_state_encoder.py` | Los ataques de zona nunca se detectaban (`str(mv.target)` ya no devuelve el nombre) | `mv.target.name in ("ALL_ADJACENT_FOES", "ALL_ADJACENT")` |
| 5 | `vgc_baseline_agent.py` | El rival `SimpleHeuristicsPlayer` no aceptaba las hojas de equipo abiertas (OTS) y el combate **se quedaba colgado para siempre** | `accept_open_team_sheet=True` también en el rival |
| 6 | `vgc_baseline_agent.py`, `leviathan_master_bot.py` | Tras un KO, un hueco vacío daba `None` y poke-env petaba (`AttributeError`) y el combate se colgaba | Se usa `PassBattleOrder()`; y en el agente base los cambios forzados los resuelve `choose_random_doubles_move` |
| 7 | (todo) | El docx dice Node v16+ | Showdown actual exige **Node >= 22.18** |

## Archivos que he añadido (no están en el docx)
`README.md`, `requirements.txt`, `check_setup.py` (comprobación de instalación) y
`play_vs_me.py` (el "paso 4 opcional" del docx hecho script: el bot acepta retos tuyos desde el navegador).

## Cosas del docx que NO he tocado pero debes saber
* La red `NessieEvaluatorNet` tiene **1,76 M** parámetros (el docx dice ~1,49 M).
* `training_pipeline.py` solo contiene la **función de pérdida**. No hay bucle de auto-juego, ni optimizador, ni guardado de pesos, aunque el texto del docx diga lo contrario.
* `Bo3OpponentTracker.record_observed_opponent_action()` **nunca se llama** en `leviathan_master_bot.py`: el rastreador de hábitos del rival no aprende nada todavía.
* El simulador rápido tiene bonificaciones de Mega-Evolución inventadas (+25 AtE, +15 Vel, +10 Def/DEE) y solo implementa unas pocas habilidades/objetos/movimientos: es una aproximación, no Showdown.
* Siempre que una Mega-Evolución es posible, el bot genera solo opciones con `mega=True` (nunca "mover sin megaevolucionar").
* Con la red sin entrenar (pesos aleatorios) el evaluador se apoya ~80 % en una heurística de PS/Pokémon vivos y solo ~20 % en la red.
