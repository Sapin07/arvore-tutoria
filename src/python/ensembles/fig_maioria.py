"""Gera figures/maioria.dat (usado pelo pgfplots no slide da lei dos grandes numeros)."""
from pathlib import Path
import numpy as np
from scipy.stats import binom

AQUI = Path(__file__).resolve().parent
Bs = np.unique(np.round(np.logspace(0, 4, 400)).astype(int))
Bs = Bs[Bs % 2 == 1]                       # B impar: sem empate
with open(AQUI / "figures" / "maioria.dat", "w") as f:
    f.write("B p51 p55 p60\n")
    for B in Bs:
        row = [1 - binom.cdf(B // 2, B, p) for p in (0.51, 0.55, 0.60)]
        f.write(f"{B} " + " ".join(f"{v:.5f}" for v in row) + "\n")
print("ok", len(Bs), "pontos")
