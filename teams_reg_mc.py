import random

TEAM_1_REG_MC = """
Incineroar @ Sitrus Berry  
Ability: Intimidate  
Level: 50  
EVs: 32 HP / 14 Def / 20 SpD  
Careful Nature  
- Fake Out  
- Flare Blitz  
- Roar  
- Parting Shot  

Basculegion (M) @ Choice Scarf  
Ability: Adaptability  
Level: 50  
EVs: 26 Atk / 13 Def / 27 Spe  
Jolly Nature  
- Wave Crash  
- Aqua Jet  
- Last Respects  
- Flip Turn  

Rillaboom @ Miracle Seed  
Ability: Grassy Surge  
Level: 50  
EVs: 32 HP / 1 Atk / 10 Def / 23 SpD  
Impish Nature  
- Fake Out  
- Wood Hammer  
- Grassy Glide  
- High Horsepower  

Raichu-Mega-Y @ Raichunite Y  
Ability: Lightning Rod  
Level: 50  
EVs: 27 HP / 14 Def / 25 Spe  
Timid Nature  
- Protect  
- Encore  
- Zap Cannon  
- Focus Blast  

Volcarona @ Grassy Seed  
Ability: Flame Body  
Level: 50  
EVs: 16 HP / 27 Def / 5 SpA / 1 SpD / 17 Spe  
Modest Nature  
- Protect  
- Quiver Dance  
- Heat Wave  
- Giga Drain  

Floette-Mega (F) @ Floettite  
Ability: Fairy Aura  
Level: 50  
EVs: 16 HP / 25 Def / 3 SpA / 22 Spe  
Timid Nature  
- Protect  
- Calm Mind  
- Dazzling Gleam  
- Draining Kiss  
"""

TEAM_2_REG_MC = """
Gholdengo @ Life Orb  
Ability: Good as Gold  
Level: 50  
EVs: 17 HP / 7 SpA / 22 SpD / 20 Spe  
Modest Nature  
- Protect  
- Shadow Ball  
- Nasty Plot  
- Make It Rain  

Volcarona @ Rocky Helmet  
Ability: Flame Body  
Level: 50  
EVs: 13 HP / 30 Def / 23 Spe  
Timid Nature  
- Overheat  
- Struggle Bug  
- Rage Powder  
- Tailwind  

Garchomp @ Garchompite Z  
Ability: Rough Skin  
Level: 50  
EVs: 4 Def / 30 SpA / 32 Spe  
Modest Nature  
- Dragon Pulse  
- Earth Power  
- Power Gem  
- Protect  

Incineroar @ Sitrus Berry  
Ability: Intimidate  
Level: 50  
EVs: 30 HP / 20 Def / 14 SpD / 2 Spe  
Impish Nature  
- Parting Shot  
- Fake Out  
- Flare Blitz  
- Darkest Lariat  

Rillaboom @ Miracle Seed  
Ability: Grassy Surge  
Level: 50  
EVs: 26 HP / 30 Atk / 1 Def / 4 SpD / 5 Spe  
Adamant Nature  
- Grassy Glide  
- Wood Hammer  
- Fake Out  
- U-turn  

Raichu @ Raichunite Y  
Ability: Lightning Rod  
Level: 50  
EVs: 32 HP / 11 Def / 23 Spe  
Timid Nature  
- Zap Cannon  
- Focus Blast  
- Fake Out  
- Protect  
"""

REG_MC_ITEMS_POOL = [
    "Sitrus Berry", "Safety Goggles", "Assault Vest", "Focus Sash",
    "Life Orb", "Leftovers", "Clear Amulet", "Covert Cloak",
    "Choice Scarf", "Rocky Helmet", "Miracle Seed", "Grassy Seed"
]

MEGA_STONES = {"Raichunite Y", "Floettite", "Garchompite Z"}


def mutate_reg_mc_pokepaste(pokepaste: str, mutation_rate: float = 0.50) -> str:
    """
    Mutador de equipos de Nessie123 adaptado a Regulación M-C (Sin Teras, con Megas y Stat Points 0-32).
    En el 50% de las partidas altera ligeramente Stat Points o intercambia objetos no-Mega
    respetando la Item Clause.
    """
    if random.random() > mutation_rate:
        return pokepaste

    blocks = [b.strip() for b in pokepaste.strip().split("\n\n") if b.strip()]
    used_items = set()
    for b in blocks:
        first_line = b.splitlines()[0]
        if "@" in first_line:
            used_items.add(first_line.split("@")[1].strip())

    idx_to_mutate = random.randrange(len(blocks))
    lines = blocks[idx_to_mutate].splitlines()

    # 1. Mutar Stat Points dentro de la escala de Champions (máx 32 por stat, suma <= 66)
    for i, line in enumerate(lines):
        if line.strip().startswith("EVs:"):
            parts = [p.strip() for p in line.replace("EVs:", "").split("/")]
            stats_dict = {}
            for p in parts:
                val_str, stat_name = p.split()
                stats_dict[stat_name] = int(val_str)

            stat_keys = list(stats_dict.keys())
            if len(stat_keys) >= 2:
                k1, k2 = random.sample(stat_keys, 2)
                delta = random.choice([1, 2, 3])
                if stats_dict[k1] - delta >= 1 and stats_dict[k2] + delta <= 32:
                    stats_dict[k1] -= delta
                    stats_dict[k2] += delta
                    lines[i] = "EVs: " + " / ".join(f"{v} {k}" for k, v in stats_dict.items())
            break

    # 2. Con 30% de probabilidad adicional, rotar objeto si NO es una Mega Piedra
    if "@" in lines[0] and random.random() < 0.30:
        mon_part, item_part = lines[0].split("@")
        current_item = item_part.strip()
        if current_item not in MEGA_STONES:
            candidates = [it for it in REG_MC_ITEMS_POOL if it not in used_items]
            if candidates:
                new_item = random.choice(candidates)
                lines[0] = f"{mon_part.rstrip()} @ {new_item}"

    blocks[idx_to_mutate] = "\n".join(lines)
    return "\n\n".join(blocks)
