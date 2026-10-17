import os
import time
import asyncio
import numpy as np
import torch
from typing import List, Tuple
from poke_env.player import Player
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder

from nessie_network import NessieEvaluatorNet
from double_oracle_search import EntropyGuidedDoubleOracleEngine, Bo3OpponentTracker
from vgc_state_encoder import VGCStateEncoder, FastVGCState
from vgc_fast_simulator import VGCFastSimulator, HybridNessieEvaluator, JointSlotAction, JointDoubleAction
from teams_reg_mc import TEAM_1_REG_MC, TEAM_2_REG_MC


class LeviathanMasterVGCBot(Player):
    """
    Agente VGC de arquitectura Nessie123 adaptado a Regulación M-C (Megas + Stat Points):
    - Open Team Sheet (OTS) activo
    - Búsqueda simultánea Double-Oracle + Profundización por Entropía de Hoja
    - Red Neuronal de 1.5M de parámetros con Cabezas Auxiliares
    - Rastreador de explotación Bo3 + Panel de Análisis en Vivo para entrenamiento humano
    """
    def __init__(
        self,
        model_checkpoint: str = None,
        inference_budget: int = 180,
        verbose_coach: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, accept_open_team_sheet=True, **kwargs)
        self.device = "cpu"
        self.net = NessieEvaluatorNet().to(self.device)

        if model_checkpoint and os.path.exists(model_checkpoint):
            self.net.load_state_dict(torch.load(model_checkpoint, map_location=self.device))
            net_trust = 0.85
            print(f"[Leviathan] Pesos neuronales cargados desde {model_checkpoint}")
        else:
            net_trust = 0.20

        self.net.eval()
        self.evaluator = HybridNessieEvaluator(self.net, device=self.device, net_trust_alpha=net_trust)
        self.bo3_tracker = Bo3OpponentTracker()
        self.search_engine = EntropyGuidedDoubleOracleEngine(
            evaluator_fn=self.evaluator.evaluate,
            simulator_step_fn=VGCFastSimulator.step,
            bo3_tracker=self.bo3_tracker
        )
        self.inference_budget = inference_budget
        self.verbose_coach = verbose_coach

    def teampreview(self, battle: DoubleBattle) -> str:
        fast_state = VGCStateEncoder.from_double_battle(battle)
        all_p1_mons = [m for m in fast_state.p1_active + fast_state.p1_reserves if m is not None]

        scores = []
        for idx, mon in enumerate(all_p1_mons):
            test_state = fast_state.clone()
            test_state.p1_active = [mon, all_p1_mons[(idx + 1) % len(all_p1_mons)]]
            win_prob, _ = self.evaluator.evaluate(test_state)
            scores.append((win_prob, idx + 1))

        scores.sort(key=lambda x: x[0], reverse=True)
        chosen = [str(i) for _, i in scores[:4]]
        while len(chosen) < 4:
            chosen.append(str(len(chosen) + 1))
        return "/team " + "".join(chosen[:4])

    def _generate_player_joint_actions(
        self,
        battle: DoubleBattle,
        fast_state: FastVGCState
    ) -> Tuple[List[JointDoubleAction], List[str]]:
        slot_candidates = [[], []]

        for slot in range(2):
            active_mon = battle.active_pokemon[slot]
            fast_mon = fast_state.p1_active[slot] if slot < len(fast_state.p1_active) else None
            if active_mon is None or active_mon.fainted or fast_mon is None:
                slot_candidates[slot].append((None, "none"))
                continue

            can_mega = bool(battle.can_mega_evolve[slot]) and not fast_state.p1_mega_used

            for mv in battle.available_moves[slot]:
                fast_mv = next((fm for fm in fast_mon.moves if fm.id == mv.id), None)
                if fast_mv is None:
                    continue
                targets = battle.get_possible_showdown_targets(mv, active_mon)
                for tgt in targets:
                    if tgt in (-1, -2) and mv.base_power > 0:
                        continue
                    order = self.create_order(mv, move_target=tgt, mega=can_mega)
                    target_idx = 1 if tgt == 2 else 0
                    mega_tag = "[MEGA]+" if can_mega else ""
                    label = f"{mega_tag}{mv.id}" + (f"->{target_idx+1}" if tgt in (1, 2) and not fast_mv.is_spread else "")
                    slot_candidates[slot].append((
                        JointSlotAction(
                            action_type="move",
                            move=fast_mv,
                            target_slot=target_idx,
                            mega_evolve=can_mega,
                            raw_order=order
                        ),
                        label
                    ))

            for sw_idx, sw_mon in enumerate(battle.available_switches[slot][:2]):
                order = self.create_order(sw_mon)
                slot_candidates[slot].append((
                    JointSlotAction(action_type="switch", switch_idx=sw_idx, raw_order=order),
                    f"Switch:{sw_mon.species}"
                ))

            if not slot_candidates[slot]:
                slot_candidates[slot].append((None, "pass"))

        joint_actions = []
        labels = []
        for s0, lbl0 in slot_candidates[0][:6]:
            for s1, lbl1 in slot_candidates[1][:6]:
                if s0 and s1 and s0.action_type == "switch" and s1.action_type == "switch":
                    if lbl0 == lbl1:
                        continue
                # Evitar ordenar dos Mega-Evoluciones simultáneas en el mismo turno
                if s0 and s1 and s0.mega_evolve and s1.mega_evolve:
                    continue
                joint_actions.append(JointDoubleAction(slot0=s0, slot1=s1))
                labels.append(f"[{lbl0}  +  {lbl1}]")

        return joint_actions[:16], labels[:16]

    def _generate_opponent_joint_actions(
        self,
        fast_state: FastVGCState
    ) -> Tuple[List[JointDoubleAction], List[str]]:
        opp_slot_cands = [[], []]
        for slot in range(2):
            opp_mon = fast_state.p2_active[slot] if slot < len(fast_state.p2_active) else None
            if opp_mon is None or opp_mon.fainted:
                opp_slot_cands[slot].append((None, "single_target"))
                continue

            opp_can_mega = bool(opp_mon.can_mega and not fast_state.p2_mega_used)

            for fm in opp_mon.moves[:4]:
                cat = "protect_turn" if fm.is_protect else ("spread_attack" if fm.is_spread else "single_target")
                if fm.id in ("tailwind", "trickroom", "icywind"):
                    cat = "speed_control"
                if fm.is_spread or fm.is_protect or fm.base_power == 0:
                    opp_slot_cands[slot].append((
                        JointSlotAction(action_type="move", move=fm, target_slot=0, mega_evolve=opp_can_mega), cat
                    ))
                else:
                    for t_slot in (0, 1):
                        opp_slot_cands[slot].append((
                            JointSlotAction(action_type="move", move=fm, target_slot=t_slot, mega_evolve=opp_can_mega), cat
                        ))

            if fast_state.p2_reserves:
                opp_slot_cands[slot].append((
                    JointSlotAction(action_type="switch", switch_idx=0), "switch_action"
                ))

            if not opp_slot_cands[slot]:
                opp_slot_cands[slot].append((None, "single_target"))

        opp_joints = []
        opp_categories = []
        for s0, c0 in opp_slot_cands[0][:5]:
            for s1, c1 in opp_slot_cands[1][:5]:
                cat = "protect_turn" if "protect_turn" in (c0, c1) else c0
                opp_joints.append(JointDoubleAction(slot0=s0, slot1=s1, category_tag=cat))
                opp_categories.append(cat)

        return opp_joints[:14], opp_categories[:14]

    def choose_move(self, battle: DoubleBattle) -> DoubleBattleOrder:
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        t0 = time.perf_counter()
        fast_state = VGCStateEncoder.from_double_battle(battle)
        p1_actions, p1_labels = self._generate_player_joint_actions(battle, fast_state)
        p2_actions, p2_categories = self._generate_opponent_joint_actions(fast_state)

        if not p1_actions:
            return self.choose_random_doubles_move(battle)

        nash_probs, expected_winrate, stats = self.search_engine.search(
            root_state=fast_state,
            p1_actions=p1_actions,
            p2_actions=p2_actions,
            p2_categories=p2_categories,
            inference_budget_evals=self.inference_budget,
            max_seeds_per_leaf=64
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        chosen_idx = int(np.random.choice(len(p1_actions), p=nash_probs))
        chosen_joint = p1_actions[chosen_idx]

        if self.verbose_coach and battle.turn <= 8:
            self._print_coach_dashboard(
                battle, expected_winrate, nash_probs, p1_labels, chosen_idx, stats, elapsed_ms
            )

        order0 = chosen_joint.slot0.raw_order if chosen_joint.slot0 else None
        order1 = chosen_joint.slot1.raw_order if chosen_joint.slot1 else None

        if order0 is None and order1 is None:
            return self.choose_random_doubles_move(battle)
        if order0 is None:
            order0 = PassBattleOrder()
        if order1 is None:
            order1 = PassBattleOrder()
        return DoubleBattleOrder(first_order=order0, second_order=order1)

    @staticmethod
    def _print_coach_dashboard(battle, winrate, nash_probs, labels, chosen_idx, stats, elapsed_ms):
        bar_len = 24
        filled = int(round(winrate * bar_len))
        eval_bar = "█" * filled + "░" * (bar_len - filled)
        p1_names = [m.species if m and not m.fainted else "---" for m in battle.active_pokemon]
        p2_names = [m.species if m and not m.fainted else "---" for m in battle.opponent_active_pokemon]

        print("\n" + "═" * 78)
        print(f" [LEVIATHAN REG M-C COACH] Turno {battle.turn} | {p1_names} vs {p2_names}")
        print(f" Barra Eval : [{eval_bar}] {winrate*100:5.1f}% Winrate Esperado")
        print(f" Búsqueda   : {stats['total_evaluations']} nodos | Soporte Nash: {stats['support_size']} jugadas | {elapsed_ms:.1f} ms")
        print("─" * 78)
        print(" Distribución de Equilibrio de Nash (Jugadas Óptimas):")

        ranked = sorted(enumerate(nash_probs), key=lambda x: x[1], reverse=True)
        for idx, prob in ranked:
            if prob >= 0.01:
                marker = "  <-- [EJECUTADA]" if idx == chosen_idx else ""
                print(f"   • {prob*100:5.1f}%  {labels[idx]}{marker}")
        print("═" * 78)


async def run_live_benchmark():
    battle_format = "gen9championsvgc2026regmc"

    print("Iniciando serie Bo3 en Reg M-C: TEAM_1_REG_MC vs TEAM_2_REG_MC...")
    leviathan_t1 = LeviathanMasterVGCBot(
        battle_format=battle_format,
        team=TEAM_1_REG_MC,
        inference_budget=180,
        verbose_coach=True,
        max_concurrent_battles=1
    )
    leviathan_t2 = LeviathanMasterVGCBot(
        battle_format=battle_format,
        team=TEAM_2_REG_MC,
        inference_budget=180,
        verbose_coach=False,
        max_concurrent_battles=1
    )

    await leviathan_t1.battle_against(leviathan_t2, n_battles=3)
    print(f"\nResultado de la serie Bo3 (Team 1 vs Team 2): {leviathan_t1.n_won_battles} - {leviathan_t2.n_won_battles}")


if __name__ == "__main__":
    asyncio.run(run_live_benchmark())
