from scipy.stats import binom

def prob_maioria(B, p):
    """P(mais da metade de B votos independentes estar certa)."""
    return 1 - binom.cdf(B // 2, B, p)

for B in (1, 11, 101, 1000, 10000):
    print(f"B = {B:>5}:  p=0.51 -> {prob_maioria(B, 0.51):.3f}   "
          f"p=0.60 -> {prob_maioria(B, 0.60):.3f}")
