"""
Comprobación rápida de la instalación. Ejecútalo ANTES de lanzar el bot:
    python check_setup.py
(Archivo añadido: no está en el docx.)
"""
import os
import shutil
import socket
import subprocess
import sys

FORMAT = "gen9championsvgc2026regmc"
HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "pokemon-showdown")
ok_all = True


def report(ok, msg):
    global ok_all
    ok_all &= ok
    print(("[ OK ] " if ok else "[FALLO] ") + msg)


report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (necesitas >= 3.10)")

for mod in ("poke_env", "numpy", "scipy", "torch"):
    try:
        __import__(mod)
        report(True, f"módulo {mod} instalado")
    except ImportError:
        report(False, f"falta el módulo {mod} -> pip install -r requirements.txt (con el venv activo)")

node = shutil.which("node")
if node:
    v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    major = int(v.lstrip("v").split(".")[0])
    report(major >= 22, f"Node {v} (Showdown actual pide >= 22.18)")
else:
    report(False, "Node.js no encontrado (instálalo desde https://nodejs.org, versión 22 o superior)")

report(os.path.isdir(SD), "carpeta pokemon-showdown/ presente (si no: git clone https://github.com/smogon/pokemon-showdown.git && cd pokemon-showdown && npm install && cp config/config-example.js config/config.js)")

if node and os.path.isdir(SD):
    sys.path.insert(0, HERE)
    from teams_reg_mc import TEAM_1_REG_MC, TEAM_2_REG_MC
    for name, team in (("TEAM_1_REG_MC", TEAM_1_REG_MC), ("TEAM_2_REG_MC", TEAM_2_REG_MC)):
        try:
            packed = subprocess.run([node, "pokemon-showdown", "pack-team"], input=team, capture_output=True,
                                    text=True, cwd=SD, timeout=120).stdout
            r = subprocess.run([node, "pokemon-showdown", "validate-team", FORMAT], input=packed,
                               capture_output=True, text=True, cwd=SD, timeout=120)
            report(r.returncode == 0, f"{name} es legal en {FORMAT}" + ("" if r.returncode == 0 else "\n" + r.stderr))
        except Exception as e:  # noqa
            report(False, f"no se pudo validar {name}: {e}")

try:
    with socket.create_connection(("localhost", 8000), timeout=2):
        report(True, "servidor Showdown escuchando en localhost:8000")
except OSError:
    report(False, "no hay servidor en localhost:8000 -> en otra terminal: cd pokemon-showdown && node pokemon-showdown start --no-security")

print("\n" + ("TODO LISTO. Ya puedes ejecutar: python leviathan_master_bot.py" if ok_all else "Arregla los puntos marcados como FALLO y vuelve a ejecutar este script."))
sys.exit(0 if ok_all else 1)
