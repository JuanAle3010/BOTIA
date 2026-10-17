import asyncio
import time
from poke_env.data import GenData
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder
from teams_reg_mc import TEAM_1_REG_MC, TEAM_2_REG_MC


class VGCStarterAgent(Player):
    """
    Agente base para VGC Regulación M-C (Combates Dobles con Mega-Evolución)
    con soporte para Open Team Sheets (OTS) y selección de Team Preview (4 de 6 Pokémon).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, accept_open_team_sheet=True, **kwargs)

    def teampreview(self, battle: DoubleBattle) -> str:
        """
        Evalúa la ventaja de tipos de los 6 Pokémon propios frente a los 6 del rival
        y selecciona el orden óptimo de 4 Pokémon (2 leads + 2 reservas).
        """
        mon_scores = []
        opp_mons = list(battle.opponent_team.values())

        for i, mon in enumerate(battle.team.values()):
            score = 0.0
            for opp in opp_mons:
                off_mult = max([t.damage_multiplier(*opp.types, type_chart=GenData.from_gen(battle.gen).type_chart)
                                for t in mon.types if t is not None], default=1.0)
                def_mult = max([t.damage_multiplier(*mon.types, type_chart=GenData.from_gen(battle.gen).type_chart)
                                for t in opp.types if t is not None], default=1.0)
                score += (off_mult - 0.8 * def_mult)
            mon_scores.append((score, i + 1))

        mon_scores.sort(key=lambda x: x[0], reverse=True)
        chosen_four = [str(idx) for _, idx in mon_scores[:4]]
        return "/team " + "".join(chosen_four)

    def choose_move(self, battle: DoubleBattle) -> DoubleBattleOrder:
        """
        Selecciona un par de acciones conjuntas válidas para los dos Pokémon activos,
        activando la Mega-Evolución si el Pokémon dispone de Mega Piedra.
        """
        # Cambio forzado tras un KO: dejamos que poke-env elija un reemplazo válido.
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        orders = [None, None]
        mega_already_queued = False

        for slot in range(2):
            active_mon = battle.active_pokemon[slot]
            if active_mon is None or active_mon.fainted:
                continue

            available_moves = battle.available_moves[slot]
            available_switches = battle.available_switches[slot]
            can_mega = bool(battle.can_mega_evolve[slot]) and not mega_already_queued

            best_order = None
            best_score = -999.0

            for move in available_moves:
                targets = battle.get_possible_showdown_targets(move, active_mon)
                for target in targets:
                    if target in (-1, -2) and move.base_power > 0:
                        continue

                    score = float(move.base_power)
                    if target in (1, 2):
                        opp_idx = target - 1
                        opp_mon = battle.opponent_active_pokemon[opp_idx]
                        if opp_mon is not None and not opp_mon.fainted:
                            mult = move.type.damage_multiplier(
                                *opp_mon.types, type_chart=GenData.from_gen(battle.gen).type_chart
                            )
                            score *= mult
                            if move.type in active_mon.types:
                                score *= 1.5

                    if score > best_score:
                        best_score = score
                        best_order = self.create_order(move, move_target=target, mega=can_mega)

            if best_order is not None and can_mega:
                mega_already_queued = True
            elif best_order is None and available_switches:
                best_order = self.create_order(available_switches[0])

            orders[slot] = best_order

        if orders[0] is None and orders[1] is None:
            return self.choose_random_doubles_move(battle)

        # Un hueco vacío (Pokémon debilitado sin reemplazo) debe ser "pass", no None.
        first = orders[0] if orders[0] is not None else PassBattleOrder()
        second = orders[1] if orders[1] is not None else PassBattleOrder()
        return DoubleBattleOrder(first_order=first, second_order=second)


async def main():
    # Formato VGC Reg M-C en el servidor Showdown
    battle_format = "gen9championsvgc2026regmc"

    agent_team1 = VGCStarterAgent(
        battle_format=battle_format,
        team=TEAM_1_REG_MC,
        max_concurrent_battles=10
    )
    # IMPORTANTE: el rival TAMBIÉN debe aceptar las hojas de equipo abiertas (OTS);
    # si no, Showdown se queda esperando y la batalla nunca empieza.
    opponent_team2 = SimpleHeuristicsPlayer(
        battle_format=battle_format,
        team=TEAM_2_REG_MC,
        max_concurrent_battles=10,
        accept_open_team_sheet=True
    )

    start_time = time.time()
    await agent_team1.battle_against(opponent_team2, n_battles=20)
    elapsed = time.time() - start_time

    print(f"Batallas completadas: {agent_team1.n_finished_battles} en {elapsed:.2f}s")
    print(f"Victorias de TEAM_1_REG_MC: {agent_team1.n_won_battles} / 20")


if __name__ == "__main__":
    asyncio.run(main())
