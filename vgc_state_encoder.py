import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition

POKEMON_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

MEGA_STONE_IDS = {"raichunitey", "floettite", "garchompitez"}


@dataclass
class FastMove:
    id: str
    base_power: float
    type_name: str
    category: str       # 'physical', 'special', 'status'
    priority: int
    accuracy: float
    is_spread: bool     # True si golpea a ambos rivales (0.75x en VGC)
    is_protect: bool


@dataclass
class FastMon:
    name: str
    hp_frac: float
    max_hp: int
    atk: int
    def_: int
    spa: int
    spd: int
    spe: int
    types: List[str]
    is_mega: bool
    can_mega: bool
    status: Optional[str]
    item: Optional[str]
    ability: Optional[str]
    boosts: dict = field(default_factory=dict)
    moves: List[FastMove] = field(default_factory=list)
    fainted: bool = False
    protect_active: bool = False

    def clone(self) -> "FastMon":
        return FastMon(
            name=self.name,
            hp_frac=self.hp_frac,
            max_hp=self.max_hp,
            atk=self.atk,
            def_=self.def_,
            spa=self.spa,
            spd=self.spd,
            spe=self.spe,
            types=list(self.types),
            is_mega=self.is_mega,
            can_mega=self.can_mega,
            status=self.status,
            item=self.item,
            ability=self.ability,
            boosts=dict(self.boosts),
            moves=self.moves,
            fainted=self.fainted,
            protect_active=self.protect_active
        )


@dataclass
class FastVGCState:
    p1_active: List[Optional[FastMon]]
    p2_active: List[Optional[FastMon]]
    p1_reserves: List[FastMon]
    p2_reserves: List[FastMon]
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    tailwind_p1: bool = False
    tailwind_p2: bool = False
    p1_mega_used: bool = False
    p2_mega_used: bool = False
    turn: int = 1

    def clone(self) -> "FastVGCState":
        return FastVGCState(
            p1_active=[m.clone() if m else None for m in self.p1_active],
            p2_active=[m.clone() if m else None for m in self.p2_active],
            p1_reserves=[m.clone() for m in self.p1_reserves],
            p2_reserves=[m.clone() for m in self.p2_reserves],
            weather=self.weather,
            terrain=self.terrain,
            trick_room=self.trick_room,
            tailwind_p1=self.tailwind_p1,
            tailwind_p2=self.tailwind_p2,
            p1_mega_used=self.p1_mega_used,
            p2_mega_used=self.p2_mega_used,
            turn=self.turn
        )


class VGCStateEncoder:
    TOKEN_DIM = 128

    @staticmethod
    def extract_fast_mon(mon: Optional[Pokemon], is_opponent: bool = False) -> Optional[FastMon]:
        if mon is None:
            return None

        base = mon.base_stats
        # Imputación a Nivel 50 con ~15-20 Stat Points medios para el rival en Reg M-C
        max_hp = int(((2 * base["hp"] + 31) * 50) / 100 + 60 + 20) if base else 180
        atk = int(((2 * base["atk"] + 31) * 50) / 100 + 5 + 15) if base else 135
        def_ = int(((2 * base["def"] + 31) * 50) / 100 + 5 + 15) if base else 115
        spa = int(((2 * base["spa"] + 31) * 50) / 100 + 5 + 15) if base else 135
        spd = int(((2 * base["spd"] + 31) * 50) / 100 + 5 + 15) if base else 115
        spe = int(((2 * base["spe"] + 31) * 50) / 100 + 5 + 15) if base else 120

        if not is_opponent and mon.stats:
            atk = mon.stats.get("atk") or atk
            def_ = mon.stats.get("def") or def_
            spa = mon.stats.get("spa") or spa
            spd = mon.stats.get("spd") or spd
            spe = mon.stats.get("spe") or spe
            max_hp = mon.max_hp or max_hp

        fast_moves = []
        for mv in mon.moves.values():
            target_name = mv.target.name if mv.target else ""
            is_spread = target_name in ("ALL_ADJACENT_FOES", "ALL_ADJACENT")
            is_protect = mv.id in ("protect", "detect", "spikyshield", "kingsshield", "burningbulwark")
            cat_str = mv.category.name.lower() if mv.category else "status"
            type_str = mv.type.name.lower() if mv.type else "normal"
            prio = int(mv.priority or 0)
            fast_moves.append(FastMove(
                id=mv.id,
                base_power=float(mv.base_power or 0),
                type_name=type_str,
                category=cat_str,
                priority=prio,
                accuracy=float(mv.accuracy if mv.accuracy is not True else 1.0),
                is_spread=is_spread,
                is_protect=is_protect
            ))

        types = [t.name.lower() for t in mon.types if t is not None]
        status_str = mon.status.name.lower() if mon.status else None
        item_id = (mon.item or "").lower().replace(" ", "")
        species_lower = (mon.species or "").lower()
        is_mega = "mega" in species_lower
        can_mega = (item_id in MEGA_STONE_IDS or item_id.endswith("ite") or item_id.endswith("itey") or item_id.endswith("itez")) and not is_mega

        return FastMon(
            name=mon.species,
            hp_frac=float(mon.current_hp_fraction if not mon.fainted else 0.0),
            max_hp=max_hp,
            atk=atk,
            def_=def_,
            spa=spa,
            spd=spd,
            spe=spe,
            types=types,
            is_mega=is_mega,
            can_mega=can_mega,
            status=status_str,
            item=item_id,
            ability=(mon.ability or "").lower().replace(" ", ""),
            boosts=dict(mon.boosts) if mon.boosts else {},
            moves=fast_moves,
            fainted=bool(mon.fainted)
        )

    @classmethod
    def from_double_battle(cls, battle: DoubleBattle) -> FastVGCState:
        p1_act = [cls.extract_fast_mon(m, False) for m in battle.active_pokemon]
        p2_act = [cls.extract_fast_mon(m, True) for m in battle.opponent_active_pokemon]

        active_p1_ids = {id(m) for m in battle.active_pokemon if m is not None}
        active_p2_ids = {id(m) for m in battle.opponent_active_pokemon if m is not None}

        p1_res = [
            cls.extract_fast_mon(m, False)
            for m in battle.team.values()
            if id(m) not in active_p1_ids and not m.fainted
        ]
        p2_res = [
            cls.extract_fast_mon(m, True)
            for m in battle.opponent_team.values()
            if id(m) not in active_p2_ids and not m.fainted
        ]

        weather = next(iter(battle.weather.keys())).name.lower() if battle.weather else None
        terrain = None
        trick_room = False
        for f in battle.fields.keys():
            fname = f.name.lower()
            if "terrain" in fname:
                terrain = fname
            if "trick_room" in fname or "trickroom" in fname:
                trick_room = True

        tw_p1 = SideCondition.TAILWIND in battle.side_conditions
        tw_p2 = SideCondition.TAILWIND in battle.opponent_side_conditions

        p1_mega_used = any(m.is_mega for m in p1_act + p1_res if m is not None)
        p2_mega_used = any(m.is_mega for m in p2_act + p2_res if m is not None)

        return FastVGCState(
            p1_active=p1_act,
            p2_active=p2_act,
            p1_reserves=p1_res,
            p2_reserves=p2_res,
            weather=weather,
            terrain=terrain,
            trick_room=trick_room,
            tailwind_p1=tw_p1,
            tailwind_p2=tw_p2,
            p1_mega_used=p1_mega_used,
            p2_mega_used=p2_mega_used,
            turn=battle.turn
        )

    @classmethod
    def encode_fast_state_to_tensor(cls, state: FastVGCState, device: str = "cpu") -> torch.Tensor:
        tokens = np.zeros((13, cls.TOKEN_DIM), dtype=np.float32)

        # Token 0: Estado Global del Campo en Reg M-C
        tokens[0, 0] = min(state.turn / 15.0, 1.0)
        tokens[0, 1] = 1.0 if state.trick_room else 0.0
        tokens[0, 2] = 1.0 if state.tailwind_p1 else 0.0
        tokens[0, 3] = 1.0 if state.tailwind_p2 else 0.0
        tokens[0, 4] = 1.0 if state.p1_mega_used else 0.0
        tokens[0, 5] = 1.0 if state.p2_mega_used else 0.0
        tokens[0, 8] = 1.0 if state.terrain and "electric" in state.terrain else 0.0
        tokens[0, 9] = 1.0 if state.terrain and "grassy" in state.terrain else 0.0
        tokens[0, 10] = 1.0 if state.terrain and "psychic" in state.terrain else 0.0
        tokens[0, 11] = 1.0 if state.terrain and "misty" in state.terrain else 0.0

        p1_all = list(state.p1_active) + list(state.p1_reserves)
        p2_all = list(state.p2_active) + list(state.p2_reserves)

        for idx in range(6):
            if idx < len(p1_all) and p1_all[idx] is not None:
                cls._write_mon_token(tokens[1 + idx], p1_all[idx], is_active=(idx < 2))
            if idx < len(p2_all) and p2_all[idx] is not None:
                cls._write_mon_token(tokens[7 + idx], p2_all[idx], is_active=(idx < 2))

        return torch.from_numpy(tokens).unsqueeze(0).to(device)

    @staticmethod
    def _write_mon_token(vec: np.ndarray, mon: FastMon, is_active: bool):
        vec[0] = 1.0
        vec[1] = 1.0 if is_active else 0.0
        vec[2] = mon.hp_frac
        vec[3] = 0.0 if mon.fainted else 1.0
        vec[4] = 1.0 if mon.is_mega else (0.5 if mon.can_mega else 0.0)
        vec[5] = 1.0 if mon.status is not None else 0.0

        vec[6] = mon.max_hp / 250.0
        vec[7] = mon.atk / 220.0
        vec[8] = mon.def_ / 220.0
        vec[9] = mon.spa / 220.0
        vec[10] = mon.spd / 220.0
        vec[11] = mon.spe / 220.0

        for b_idx, stat_key in enumerate(["atk", "def", "spa", "spd", "spe"]):
            vec[12 + b_idx] = mon.boosts.get(stat_key, 0) / 6.0

        for t in mon.types:
            if t in POKEMON_TYPES:
                vec[18 + POKEMON_TYPES.index(t)] = 1.0

        for m_idx, mv in enumerate(mon.moves[:4]):
            base_offset = 40 + m_idx * 10
            vec[base_offset] = mv.base_power / 150.0
            vec[base_offset + 1] = (mv.priority + 7.0) / 14.0
            vec[base_offset + 2] = 1.0 if mv.is_spread else 0.0
            vec[base_offset + 3] = 1.0 if mv.is_protect else 0.0
            vec[base_offset + 4] = 1.0 if mv.category == "physical" else (0.5 if mv.category == "special" else 0.0)
