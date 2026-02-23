#!/usr/bin/env python3
"""Bayesian posterior P(Δ≥9,529) for BSW miscount."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
SEP = "=" * 60
DEFICIT = 9_529
N_PREC = 95_046
N_RECOUNT = 50
S_RECOUNT = 15  # 50 × 0.3
CORR_OBS = 4_277  # prelim→final BSW gain
N_SIMS = 50_000

PRIORS = {
    "informative": (2, 10),
    "weakly_inf": (1, 1),
    "jeffreys": (0.5, 0.001),
}


def model_a(prior_a, prior_b, rng):
    """Uniform rate: all precincts same θ."""
    a = prior_a + S_RECOUNT
    b = prior_b + N_RECOUNT
    thetas = rng.gamma(a, 1/b, N_SIMS)
    totals = np.array([rng.poisson(t*N_PREC)
                       for t in thetas])
    return totals


def model_b(prior_a, prior_b, rng):
    """Mixture: π problem precincts (high rate),
    rest normal (low rate)."""
    a = prior_a + S_RECOUNT
    b = prior_b + N_RECOUNT
    totals = np.empty(N_SIMS)
    for i in range(N_SIMS):
        pi = rng.beta(1, 9)  # ~10% problem
        n_hi = rng.binomial(N_PREC, pi)
        th_hi = rng.gamma(a, 1/b)
        th_lo = rng.gamma(0.5, 1/10)
        totals[i] = (rng.poisson(th_hi*n_hi)
                     + rng.poisson(th_lo*(N_PREC-n_hi)))
    return totals


def model_c(prior_a, prior_b, rng):
    """Correction-informed: +4,277 observed,
    model remaining error."""
    a = prior_a + S_RECOUNT
    b = prior_b + N_RECOUNT
    thetas = rng.gamma(a, 1/b, N_SIMS)
    n_uncorr = N_PREC - 500
    remaining = np.array([rng.poisson(t*n_uncorr)
                          for t in thetas])
    return CORR_OBS + remaining


def _stats(totals):
    p5, p50, p95 = np.percentile(totals, [5,50,95])
    pc = (totals >= DEFICIT).mean()
    return p50, p5, p95, pc


def run_all():
    """Run all models × priors."""
    rng = np.random.RandomState(42)
    models = {"A_uniform": model_a,
              "B_mixture": model_b,
              "C_correction": model_c}
    rows = []
    for mn, mf in models.items():
        for pn, (pa, pb) in PRIORS.items():
            t = mf(pa, pb, rng)
            med, lo, hi, pc = _stats(t)
            rows.append({"model": mn, "prior": pn,
                "median": med, "ci5": lo, "ci95": hi,
                "p_cross": pc})
    return pd.DataFrame(rows)


def main():
    print(f"{SEP}\nBAYESIAN POSTERIOR: P(Δ≥9,529)")
    print(f"N={N_PREC:,}, recount:{N_RECOUNT}×"
          f"{S_RECOUNT/N_RECOUNT:.1f}, "
          f"corr:+{CORR_OBS:,}\n{SEP}")
    df = run_all()
    for _, r in df.iterrows():
        print(f"  {r['model']:<14} {r['prior']:<12}"
              f" med={r['median']:>7,.0f}"
              f" P={r['p_cross']:.1%}")
    df.to_csv(DATA/"bayesian_posterior.csv",
              index=False)
    print(f"\nSaved bayesian_posterior.csv")


if __name__ == "__main__":
    main()
