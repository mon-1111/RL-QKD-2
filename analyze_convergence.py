"""
Convergence analysis for the learning-defender game (Stage 4)
=============================================================
Answers the keystone question from the 45-run sweep: is the attacker-defender
best-response game a genuine non-converging CYCLE, or was it slow convergence
that just hadn't settled in K=5 rounds?

This reads the sweep logs you ALREADY HAVE (results/sweep/*.json) -- no new
compute. Each log has the per-round fresh-attacker detection trajectory. We ask,
per cell:

  - Cycle amplitude by half: std of the fresh-attacker detection over the FIRST
    half of rounds vs the SECOND half. If the second-half amplitude is as large
    as the first, the game is NOT settling (persistent cycle). If it shrinks
    toward zero, it is converging.
  - Round-to-round reversal rate: how often the direction of change flips sign.
    A cycling game reverses direction most rounds (~>50%); a converging one
    stops reversing.

Interpretation printed per noise regime, since the behavior is noise-dependent.

Usage:  python analyze_convergence.py
(Optional) point at a different dir: python analyze_convergence.py results/sweep
"""

import os
import sys
import json
import numpy as np

SWEEP_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/sweep"
AGENTS = ["qlearning", "sarsa", "doubleq"]
NOISES = [0.01, 0.03, 0.05]


def reversal_rate(traj):
    """Fraction of interior points where the step direction flips sign."""
    if len(traj) < 3:
        return float("nan")
    diffs = np.diff(traj)
    signs = np.sign(diffs)
    flips = np.sum(signs[1:] * signs[:-1] < 0)
    return flips / (len(diffs) - 1)


def half_amplitudes(traj):
    """Std of the first half vs second half of the trajectory (in pp)."""
    n = len(traj)
    if n < 4:
        return float("nan"), float("nan")
    mid = n // 2
    first = np.std(traj[:mid]) * 100 if max(traj) <= 1.0 else np.std(traj[:mid])
    second = np.std(traj[mid:]) * 100 if max(traj) <= 1.0 else np.std(traj[mid:])
    return float(first), float(second)


def load_cells():
    if not os.path.isdir(SWEEP_DIR):
        print(f"No sweep dir at {SWEEP_DIR}. Upload/point at your saved logs.")
        return None
    cells = {}
    for f in os.listdir(SWEEP_DIR):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(SWEEP_DIR, f)) as fh:
            log = json.load(fh)
        # fresh-attacker detection trajectory across rounds
        rounds = log.get("rounds", [])
        traj = [r["fresh_attacker_det"] for r in rounds]
        # detection values are stored as fractions (0-1); normalize to pp here
        traj = [x * 100 if x <= 1.0 else x for x in traj]
        key = (log["agent"], log["noise"])
        cells.setdefault(key, []).append(traj)
    return cells


def main():
    cells = load_cells()
    if cells is None:
        return
    print("=" * 76)
    print("CONVERGENCE vs CYCLING analysis (from existing sweep logs, no new compute)")
    print("  fresh-attacker detection trajectory across best-response rounds")
    print("=" * 76)
    print(f"  {'agent':10s} {'mu':>4s} "
          f"{'1st-half amp':>13s} {'2nd-half amp':>13s} {'reversal rate':>14s}  verdict")
    print("  " + "-" * 72)

    regime = {}
    for agent in AGENTS:
        for noise in NOISES:
            trajs = cells.get((agent, noise), [])
            if not trajs:
                continue
            a1 = np.nanmean([half_amplitudes(t)[0] for t in trajs])
            a2 = np.nanmean([half_amplitudes(t)[1] for t in trajs])
            rr = np.nanmean([reversal_rate(t) for t in trajs])
            # per-cell verdict: cycle persists if 2nd-half amplitude is still
            # large (not shrinking toward 0) AND reversals stay frequent.
            persists = (a2 > 10.0) and (rr >= 0.4)
            tag = "CYCLE persists" if persists else (
                  "damping" if a2 < a1 * 0.5 else "flat/low")
            print(f"  {agent:10s} {noise*100:>3.0f}% "
                  f"{a1:>11.1f}pp {a2:>11.1f}pp {rr:>13.2f}  {tag}")
            regime.setdefault(noise, []).append((a1, a2, rr, persists))

    print("=" * 76)
    print("\n  Per-regime summary:")
    for noise in NOISES:
        rows = regime.get(noise, [])
        if not rows:
            continue
        a1 = np.mean([r[0] for r in rows])
        a2 = np.mean([r[1] for r in rows])
        rr = np.mean([r[2] for r in rows])
        n_persist = sum(r[3] for r in rows)
        shrink = "shrinks" if a2 < a1 * 0.6 else "holds"
        print(f"    mu={noise*100:>2.0f}%: amplitude {a1:.1f}pp -> {a2:.1f}pp "
              f"({shrink}), reversal {rr:.2f}, "
              f"{n_persist}/{len(rows)} cells cycle-persistent")

    print("\n  READING:")
    print("  - If 2nd-half amplitude stays HIGH and reversal rate ~>=0.5,")
    print("    the game does NOT converge in K=5 -> motivates a longer-K run")
    print("    to confirm the cycle is persistent (the 'no stable equilibrium'")
    print("    claim). If amplitude SHRINKS, it was slow convergence.")
    print("  - K=5 gives only ~2-3 points per half, so this is SUGGESTIVE.")
    print("    The definitive test is a longer-K run (K=15-20) on a few cells.")


if __name__ == "__main__":
    main()