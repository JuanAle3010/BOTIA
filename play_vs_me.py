"""
Modo práctica: el bot se queda esperando retos en tu Showdown local.

Uso:
    python play_vs_me.py                          # el bot usa TEAM_1_REG_MC
    python play_vs_me.py --team 2                 # el bot usa TEAM_2_REG_MC
    python play_vs_me.py --team mi_equipo.txt     # el bot usa un equipo tuyo (formato Showdown/pokepaste)
    python play_vs_me.py --n 10 --budget 250      # acepta 10 retos, más búsqueda por turno

Luego abre http://localhost:8000, elige un nombre, pulsa "Choose name",
importa TU equipo, elige el formato "[Gen 9 Champions] VGC 2026 Reg M-C"
y reta al usuario "LeviathanBot". En la terminal verás qué jugadas considera óptimas.

(Archivo añadido: no está en el docx. Es el "paso 4 opcional" del docx, ya hecho.)
"""
import argparse
import asyncio
import os

from poke_env import AccountConfiguration

from leviathan_master_bot import LeviathanMasterVGCBot
from teams_reg_mc import TEAM_1_REG_MC, TEAM_2_REG_MC

BATTLE_FORMAT = "gen9championsvgc2026regmc"


def load_team(arg: str) -> str:
    if arg == "1":
        return TEAM_1_REG_MC
    if arg == "2":
        return TEAM_2_REG_MC
    with open(arg, encoding="utf8") as f:
        return f.read()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="1", help="1, 2 o ruta a un .txt con el equipo")
    ap.add_argument("--name", default="LeviathanBot", help="nombre de usuario del bot")
    ap.add_argument("--n", type=int, default=10, help="número de retos a aceptar")
    ap.add_argument("--budget", type=int, default=180, help="evaluaciones de búsqueda por turno")
    ap.add_argument("--checkpoint", default=None, help="pesos .pt entrenados (opcional)")
    ap.add_argument("--no-coach", action="store_true", help="no imprimir el panel de coach")
    args = ap.parse_args()

    bot = LeviathanMasterVGCBot(
        model_checkpoint=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None,
        inference_budget=args.budget,
        verbose_coach=not args.no_coach,
        account_configuration=AccountConfiguration(args.name, None),
        battle_format=BATTLE_FORMAT,
        team=load_team(args.team),
        max_concurrent_battles=1,
    )
    print(f"[Leviathan] Esperando retos de '{BATTLE_FORMAT}' para el usuario '{args.name}' ...")
    await bot.accept_challenges(None, args.n)
    print(f"Resultado: el bot ganó {bot.n_won_battles} de {bot.n_finished_battles}")


if __name__ == "__main__":
    asyncio.run(main())
