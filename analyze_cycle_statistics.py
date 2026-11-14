"""
Cycle statistics: lag-1 autocorrelation + reversal-rate-vs-null (point 2)
=========================================================================
The reversal rate alone cannot distinguish a genuine cycle from i.i.d. noise,
because for ANY i.i.d. sequence the expected reversal rate is 2/3 = 0.667 (the
middle of any three consecutive draws is a local peak or valley with prob 2/3).
A true alternating cycle pushes the reversal rate toward 1.0; i.i.d. noise sits
at 0.667.

This script reads the K=20 convergence logs you already have and, per cell:
  - reports the reversal rate against the 0.667 i.i.d. null and its 95% band,
  - computes the LAG-1 AUTOCORRELATION of the fresh-attacker detection
    trajectory. rho_1 strongly NEGATIVE  => genuine round-to-round oscillation
    (overshoot/undershoot, a real cycle). rho_1 NEAR ZERO => each round's fresh
    attacker lands at a random point of a stationary distribution (no cycle,
    just non-convergence with fluctuation).
  - a permutation p-value for rho_1 (is it more negative than chance?).

The lag-1 autocorrelation is the statistic the reviewer recommends and is the
one that actually decides "persistent cycle" vs "fluctuates at best-response
variance scale."

Usage:
    python analyze_cycle_statistics.py                      # results/convergence
    python analyze_cycle_statistics.py results/convergence  # explicit dir
"""

import os
import sys
import json
import numpy as np

CONV_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/convergence"

# i.i.d. null for the reversal rate of a continuous sequence.
IID_REVERSAL_MEAN = 2.0 / 3.0


def reversal_rate(traj):
    if len(traj) < 3:
        return float("nan")
    d = np.diff(traj)
    s = np.sign(d)
    flips = np.sum(s[1:] * s[:-1] < 0)
    return flips / (len(d) - 1)


def iid_reversal_band(n_interior, n_boot=20000, seed=0):
    """95% band for the reversal rate of an i.i.d. sequence of this length."""
    rng = np.random.default_rng(seed)
    n_points = n_interior + 2  # interior reversals need n_points-2 interior pts
    rates = []
    for _ in range(n_boot):
        x = rng.standard_normal(n_points)
        rates.append(reversal_rate(x))
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def lag1_autocorr(traj):
    """Pearson lag-1 autocorrelation of the trajectory (mean-centered)."""
    x = np.asarray(traj, dtype=float)
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[1:] * x[:-1]) / denom)


def perm_pvalue_rho1(traj, n_perm=20000, seed=0):
    """
    One-sided permutation p-value: P(rho_1 of a random reordering <= observed).
    Small p => the trajectory is MORE negatively autocorrelated (more cyclic)
    than a random reshuffling of the same values would be.
    """
    rng = np.random.default_rng(seed)
    obs = lag1_autocorr(traj)
    if not np.isfinite(obs):
        return float("nan"), obs
    x = np.asarray(traj, dtype=float)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(x)
        if lag1_autocorr(x) <= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1), obs


def load_cells(conv_dir):
    if not os.path.isdir(conv_dir):
        print(f"No convergence dir at {conv_dir}. Point at your K=20 logs.")
        return None
    cells = {}
    for f in sorted(os.listdir(conv_dir)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(conv_dir, f)) as fh:
            log = json.load(fh)
        rounds = log.get("rounds", [])
        traj = [r["fresh_attacker_det"] for r in rounds]
        traj = [x * 100 if x <= 1.0 else x for x in traj]  # -> pp
        key = (log["agent"], log["noise"], f)
        cells[key] = traj
    return cells


def main():
    cells = load_cells(CONV_DIR)
    if cells is None:
        return
    if not cells:
        print(f"No .json logs in {CONV_DIR}.")
        return

    # Reversal-rate i.i.d. band for a typical K=20 trajectory (18 interior pts).
    any_traj = next(iter(cells.values()))
    n_int = max(1, len(any_traj) - 2)
    lo, hi = iid_reversal_band(n_int)

    print("=" * 94)
    print("CYCLE STATISTICS on the fresh-attacker detection trajectory (K=20 logs)")
    print(f"  i.i.d. reversal-rate null = {IID_REVERSAL_MEAN:.3f}, "
          f"95% band for ~{n_int} interior points = [{lo:.2f}, {hi:.2f}]")
    print("  rho_1 << 0 => genuine cycle ; rho_1 ~ 0 => i.i.d.-like fluctuation")
    print("=" * 94)
    print(f"  {'file':38s} {'len':>3s} {'revRate':>8s} {'vs null':>9s} "
          f"{'rho_1':>7s} {'perm p':>8s}  reading")
    print("  " + "-" * 90)

    rows = []
    for (agent, noise, fname), traj in cells.items():
        rr = reversal_rate(traj)
        rho1 = lag1_autocorr(traj)
        p, _ = perm_pvalue_rho1(traj)
        if rr < lo:
            vs = "BELOW"
        elif rr > hi:
            vs = "ABOVE"
        else:
            vs = "in band"
        # reading: a real cycle needs rho_1 clearly negative (and p small)
        if np.isfinite(rho1) and rho1 <= -0.3 and p <= 0.1:
            reading = "cycle-like"
        elif np.isfinite(rho1) and rho1 >= -0.1:
            reading = "iid-like (no cycle)"
        else:
            reading = "weak/ambiguous"
        print(f"  {fname:38s} {len(traj):>3d} {rr:>8.2f} {vs:>9s} "
              f"{rho1:>7.2f} {p:>8.3f}  {reading}")
        rows.append((agent, noise, rr, rho1, p, reading))

    print("=" * 94)
    print("\n  Per (agent, noise) means:")
    seen = {}
    for agent, noise, rr, rho1, p, _ in rows:
        seen.setdefault((agent, noise), []).append((rr, rho1, p))
    for (agent, noise), vals in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rrs = np.mean([v[0] for v in vals])
        rhos = np.mean([v[1] for v in vals])
        ps = np.mean([v[2] for v in vals])
        print(f"    {agent:10s} mu={noise*100:>2.0f}%: "
              f"reversal {rrs:.2f} (null {IID_REVERSAL_MEAN:.2f}), "
              f"rho_1 {rhos:+.2f}, mean perm p {ps:.3f}")

    print("\n  HEADLINE DECISION:")
    print("  - If rho_1 is near zero across cells, the reversal rate is a noise")
    print("    artifact and 'persistent cycle' is NOT supported: report the result")
    print("    as 'no convergence; round-to-round fluctuation at best-response scale'.")
    print("  - If rho_1 is clearly negative with small perm p (esp. mu=3%), the")
    print("    oscillation is real and 'cycle' is earned for those cells.")


if __name__ == "__main__":
    main()
