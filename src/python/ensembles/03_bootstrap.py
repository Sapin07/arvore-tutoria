import numpy as np

rng = np.random.default_rng(0)
for n in (10, 150, 1000, 100_000):
    idx = rng.integers(0, n, size=n)      # n sorteios com reposicao
    print(f"n = {n:>6}: distintas = {len(np.unique(idx)) / n:.3f}")
print(f"limite 1 - 1/e = {1 - np.exp(-1):.3f}")
