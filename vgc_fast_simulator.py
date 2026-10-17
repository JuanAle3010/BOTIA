import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from vgc_state_encoder import FastVGCState, FastMon, FastMove, VGCStateEncoder

TYPE_CHART = {
    "fire": {"grass": 2.0, "ice": 2.0, "bug": 2.0, "steel": 2.0, "fire": 0.5, "water": 0.5, "rock": 0.5, "dragon": 0.5},
    "water": {"fire": 2.0, "ground": 2.0, "rock": 2.0, "water": 0.5, "grass": 0.5, "dragon": 0.5},
    "grass": {"water": 2.0, "ground": 2.0, "rock": 2.0, "fire": 0.5, "grass": 0.5, "poison": 0.5, "flying": 0.5, "bug": 0.5, "dragon": 0.5, "steel": 0.5},
    "electric": {"water": 2.0, "flying": 2.0, "electric": 0.5, "grass": 0.5, "dragon": 0.5, "ground": 0.0},
    "ice": {"grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "fire": 0.5, "water": 0.5, "ice": 0.5, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "rock": 2.0, "dark": 2.0, "steel": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "fairy": 0.5, "ghost": 0.0},
    "ground": {"fire": 2.0, "electric": 2.0, "poison": 2.0, "rock": 2.0, "steel": 2.0, "grass": 0.5, "bug": 0.5, "flying": 0.0},
    "flying": {"grass": 2.0, "fighting": 2.0, "bug": 2.0, "electric": 0.5, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "steel": 0.5, "dark": 0.0},
    "ghost": {"psychic": 2.0, "ghost": 2.0, "dark": 0.5, "normal": 0.0},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"psychic": 2.0, "ghost": 2.0, "fighting": 0.5, "dark": 0.5, "fairy": 0.5},
    "steel": {"ice": 2.0, "rock": 2.0, "fairy": 2.0, "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
    "fairy": {"fighting": 2.0, "dragon": 2.0, "dark": 2.0, "fire": 0.5, "poison": 0.5, "steel": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0, "fighting": 0.5, "ground": 0.5, "steel": 0.5},
    "poison": {"grass": 2.0, "fairy": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0},
    "bug": {"grass": 2.0, "psychic": 2.0, "dark": 2.0, "fire": 0.5, "fighting": 0.5, "flying": 0.5, "ghost": 0.5, "steel": 0.5, "fairy": 0.5},
    "normal": {"rock": 0.5, "steel": 0.5, "ghost": 0.0}
}


@dataclass
class JointSlotAction:
    action_type: str        # 'move' o 'switch'
    move: Optional[FastMove] = None
    target_slot: int = 0
    switch_idx: int = 0
    mega_evolve: bool = False
    raw_order: object = None


@dataclass
class JointDoubleAction:
    slot0: Optional[JointSlotAction]
    slot1: Optional[JointSlotAction]
    category_tag: str = "single_target"


class VGCFastSimulator:
    DAMAGE_ROLLS = np.linspace(0.85, 1.00, 16)

    @staticmethod
    def type_effectiveness(move_type: str, defender_types: List[str]) -> float:
        mult = 1.0
        chart_row = TYPE_CHART.get(move_type, {})
        for dt in defender_types:
            mult *= chart_row.get(dt, 1.0)
        return mult

    @staticmethod
    def stat_stage_multiplier(stage: int) -> float:
        stage = max(-6, min(6, stage))
        return (2.0 + stage) / 2.0 if stage >= 0 else 2.0 / (2.0 - stage)

    @classmethod
    def effective_speed(cls, mon: FastMon, tailwind: bool) -> float:
        spe = mon.spe * cls.stat_stage_multiplier(mon.boosts.get("spe", 0))
        if mon.item == "choicescarf":
            spe *= 1.5
        if tailwind:
            spe *= 2.0
        if mon.status == "par":
            spe *= 0.5
        return spe

    @staticmethod
    def apply_mega_evolution(mon: FastMon):
        if not mon.can_mega or mon.is_mega:
            return
        mon.is_mega = True
        mon.can_mega = False
        mon.spa += 25
        mon.spe += 15
        mon.def_ += 10
        mon.spd += 10

    @classmethod
    def compute_vgc_damage_fraction(
        cls,
        attacker: FastMon,
        defender: FastMon,
        move: FastMove,
        state: FastVGCState,
        roll_idx: int = 15,
        is_crit: bool = False
    ) -> float:
        if move.base_power <= 0 or move.category == "status" or defender.fainted:
            return 0.0

        if move.category == "physical":
            atk_stage = attacker.boosts.get("atk", 0)
            def_stage = defender.boosts.get("def", 0)
            atk_stat = attacker.atk * cls.stat_stage_multiplier(max(0, atk_stage) if is_crit else atk_stage)
            def_stat = defender.def_ * cls.stat_stage_multiplier(min(0, def_stage) if is_crit else def_stage)
            if attacker.status == "brn":
                atk_stat *= 0.5
        else:
            spa_stage = attacker.boosts.get("spa", 0)
            spd_stage = defender.boosts.get("spd", 0)
            atk_stat = attacker.spa * cls.stat_stage_multiplier(max(0, spa_stage) if is_crit else spa_stage)
            def_stat = defender.spd * cls.stat_stage_multiplier(min(0, spd_stage) if is_crit else spd_stage)

        base_dmg = ((22.0 * move.base_power * (atk_stat / max(1.0, def_stat))) / 50.0) + 2.0

        if move.is_spread:
            base_dmg *= 0.75

        if state.terrain and "grassy" in state.terrain and move.type_name == "grass":
            base_dmg *= 1.3
        if attacker.item == "miracleseed" and move.type_name == "grass":
            base_dmg *= 1.2
        if attacker.item == "lifeorb":
            base_dmg *= 1.3
        if attacker.ability == "fairyaura" and move.type_name == "fairy":
            base_dmg *= 1.33

        if is_crit:
            base_dmg *= 1.5

        base_dmg *= cls.DAMAGE_ROLLS[roll_idx % 16]

        if move.type_name in attacker.types:
            base_dmg *= 2.0 if attacker.ability == "adaptability" else 1.5

        type_mult = cls.type_effectiveness(move.type_name, defender.types)
        base_dmg *= type_mult

        return float(base_dmg / max(1, defender.max_hp))

    @classmethod
    def step(
        cls,
        state: FastVGCState,
        p1_joint: JointDoubleAction,
        p2_joint: JointDoubleAction,
        seed: int = 0
    ) -> Tuple[FastVGCState, bool, float]:
        next_s = state.clone()
        rng = np.random.RandomState(seed + 1337 * state.turn)

        for m in next_s.p1_active + next_s.p2_active:
            if m is not None:
                m.protect_active = False

        # 1. Resolver Mega-Evoluciones al inicio del turno
        for side_idx, (active_list, joint_act) in enumerate([
            (next_s.p1_active, p1_joint),
            (next_s.p2_active, p2_joint)
        ]):
            for slot_idx, slot_act in enumerate([joint_act.slot0, joint_act.slot1]):
                if slot_act and slot_act.mega_evolve and slot_idx < len(active_list):
                    mon = active_list[slot_idx]
                    if mon and not mon.fainted:
                        cls.apply_mega_evolution(mon)
                        if side_idx == 0:
                            next_s.p1_mega_used = True
                        else:
                            next_s.p2_mega_used = True

        # 2. Ordenar acciones por prioridad (Grassy Glide +1 en Grassy Terrain) y velocidad efectiva
        execution_queue = []
        for side_idx, (active_list, joint_act, tw) in enumerate([
            (next_s.p1_active, p1_joint, next_s.tailwind_p1),
            (next_s.p2_active, p2_joint, next_s.tailwind_p2)
        ]):
            for slot_idx, slot_act in enumerate([joint_act.slot0, joint_act.slot1]):
                if slot_act is None or slot_idx >= len(active_list):
                    continue
                mon = active_list[slot_idx]
                if mon is None or mon.fainted:
                    continue

                if slot_act.action_type == "switch":
                    execution_queue.append((6, 9999.0, side_idx, slot_idx, slot_act))
                elif slot_act.move is not None:
                    prio = slot_act.move.priority
                    if slot_act.move.id == "grassyglide" and next_s.terrain and "grassy" in next_s.terrain:
                        prio = 1
                    spe = cls.effective_speed(mon, tw)
                    eff_spe = -spe if next_s.trick_room else spe
                    execution_queue.append((prio, eff_spe, side_idx, slot_idx, slot_act))

        execution_queue.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # 3. Ejecutar acciones
        for prio, _, side_idx, slot_idx, slot_act in execution_queue:
            own_active = next_s.p1_active if side_idx == 0 else next_s.p2_active
            opp_active = next_s.p2_active if side_idx == 0 else next_s.p1_active
            own_reserves = next_s.p1_reserves if side_idx == 0 else next_s.p2_reserves

            actor = own_active[slot_idx]
            if actor is None or actor.fainted:
                continue

            if slot_act.action_type == "switch":
                if 0 <= slot_act.switch_idx < len(own_reserves):
                    replacement = own_reserves[slot_act.switch_idx]
                    if not replacement.fainted:
                        own_reserves[slot_act.switch_idx] = actor
                        own_active[slot_idx] = replacement
                        if replacement.ability == "intimidate":
                            for opp in opp_active:
                                if opp and not opp.fainted:
                                    opp.boosts["atk"] = max(-6, opp.boosts.get("atk", 0) - 1)
                        elif replacement.ability == "grassysurge":
                            next_s.terrain = "grassyterrain"
                continue

            mv = slot_act.move
            if mv.is_protect:
                actor.protect_active = True
                continue

            if mv.id == "tailwind":
                if side_idx == 0:
                    next_s.tailwind_p1 = True
                else:
                    next_s.tailwind_p2 = True
                continue
            if mv.id == "quiverdance":
                actor.boosts["spa"] = min(6, actor.boosts.get("spa", 0) + 1)
                actor.boosts["spd"] = min(6, actor.boosts.get("spd", 0) + 1)
                actor.boosts["spe"] = min(6, actor.boosts.get("spe", 0) + 1)
                continue
            if mv.id == "nastyplot":
                actor.boosts["spa"] = min(6, actor.boosts.get("spa", 0) + 2)
                continue
            if mv.id == "calmmind":
                actor.boosts["spa"] = min(6, actor.boosts.get("spa", 0) + 1)
                actor.boosts["spd"] = min(6, actor.boosts.get("spd", 0) + 1)
                continue

            if mv.accuracy < 1.0 and rng.rand() > mv.accuracy:
                continue

            targets = [0, 1] if mv.is_spread else [min(max(0, slot_act.target_slot), 1)]
            roll_idx = seed % 16
            is_crit = (seed >= 16 and rng.rand() < (1.0 / 24.0))

            for t_idx in targets:
                if t_idx >= len(opp_active):
                    continue
                target_mon = opp_active[t_idx]
                if target_mon is None or target_mon.fainted:
                    alt_idx = 1 - t_idx
                    if alt_idx < len(opp_active) and opp_active[alt_idx] and not opp_active[alt_idx].fainted:
                        target_mon = opp_active[alt_idx]
                    else:
                        continue

                if target_mon.protect_active:
                    continue

                dmg_frac = cls.compute_vgc_damage_fraction(actor, target_mon, mv, next_s, roll_idx, is_crit)
                target_mon.hp_frac = max(0.0, target_mon.hp_frac - dmg_frac)

                if mv.id == "zapcannon" and target_mon.hp_frac > 0 and "electric" not in target_mon.types:
                    target_mon.status = "par"

                if 0.0 < target_mon.hp_frac <= 0.50 and target_mon.item == "sitrusberry":
                    target_mon.hp_frac = min(1.0, target_mon.hp_frac + 0.25)
                    target_mon.item = None

                if target_mon.hp_frac <= 0.0:
                    target_mon.fainted = True

            if mv.id == "makeitrain":
                actor.boosts["spa"] = max(-6, actor.boosts.get("spa", 0) - 1)
            elif mv.id == "overheat":
                actor.boosts["spa"] = max(-6, actor.boosts.get("spa", 0) - 2)
            if actor.item == "lifeorb" and mv.base_power > 0:
                actor.hp_frac = max(0.0, actor.hp_frac - 0.10)
                if actor.hp_frac <= 0.0:
                    actor.fainted = True

        for active_list, res_list in [(next_s.p1_active, next_s.p1_reserves), (next_s.p2_active, next_s.p2_reserves)]:
            for idx in range(len(active_list)):
                if active_list[idx] is None or active_list[idx].fainted:
                    for r_idx, r_mon in enumerate(res_list):
                        if not r_mon.fainted:
                            active_list[idx] = r_mon
                            res_list.pop(r_idx)
                            break

        next_s.turn += 1
        p1_alive = sum(1 for m in next_s.p1_active + next_s.p1_reserves if m and not m.fainted)
        p2_alive = sum(1 for m in next_s.p2_active + next_s.p2_reserves if m and not m.fainted)

        if p1_alive == 0 and p2_alive == 0:
            return next_s, True, 0.5
        if p2_alive == 0:
            return next_s, True, 1.0
        if p1_alive == 0:
            return next_s, True, 0.0
        return next_s, False, 0.5


class HybridNessieEvaluator:
    def __init__(self, net: torch.nn.Module, device: str = "cpu", net_trust_alpha: float = 0.25):
        self.net = net.to(device)
        self.device = device
        self.net_trust_alpha = net_trust_alpha

    @torch.no_grad()
    def evaluate(self, state: FastVGCState) -> Tuple[float, float]:
        tensor_in = VGCStateEncoder.encode_fast_state_to_tensor(state, self.device)
        out = self.net(tensor_in, return_aux=False)
        net_winrate = float(out["winrate"].item())
        leaf_entropy = float(out["leaf_entropy"].item())

        p1_mons = [m for m in state.p1_active + state.p1_reserves if m and not m.fainted]
        p2_mons = [m for m in state.p2_active + state.p2_reserves if m and not m.fainted]

        p1_hp_sum = sum(m.hp_frac for m in p1_mons)
        p2_hp_sum = sum(m.hp_frac for m in p2_mons)
        raw_adv = 1.4 * (p1_hp_sum - p2_hp_sum) + 0.85 * (len(p1_mons) - len(p2_mons))
        if state.tailwind_p1 and not state.tailwind_p2:
            raw_adv += 0.25
        elif state.tailwind_p2 and not state.tailwind_p1:
            raw_adv -= 0.25

        tactical_winrate = float(1.0 / (1.0 + np.exp(-raw_adv)))
        blended_winrate = (1.0 - self.net_trust_alpha) * tactical_winrate + self.net_trust_alpha * net_winrate
        uncertainty = 4.0 * blended_winrate * (1.0 - blended_winrate)
        effective_entropy = 0.5 * leaf_entropy + 1.5 * uncertainty

        return blended_winrate, effective_entropy
