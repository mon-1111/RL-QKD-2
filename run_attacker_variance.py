"""
Attacker-variance control (Reviewer point 3 -- the decisive experiment)
=======================================================================
Question: are the round-to-round swings in fresh-attacker detection a GAME
dynamic (the defender and attacker genuinely chasing each other with no fixed
point), or just VARIANCE in the best-response oracle (retraining an attacker
against ONE UNCHANGED defender already lands at very different detection rates)?

The control isolates the second effect. We FREEZE a single final defender and
train N fresh attackers against it with N different seeds, then measure the
spread of their detection rates. We compare that spread to the round-to-round
"cycle" amplitude from the K=20 convergence runs.

  spread  >=  cycle amplitude   =>  much of the "non-convergence" is oracle
                                     noise, not game dynamics. Soften the
                                     headline to "fluctuates at the scale of
                                     best-response variance."
  spread  <<  cycle amplitude   =>  the swings are real game dynamics; the
                                     "no stable equilibrium" cycle survives.

HOW THE FROZEN DEFENDER IS OBTAINED
-----------------------------------
run_game is deterministic given its seeds: round r's defender is produced by
train_defender(noise, <frozen attacker from round r-1>, ..., seed+r,
defender_seed=seed+r). We reproduce exactly that defender by replaying the game
up to and including round r's defender step, using the SAME master seed as the
saved K=20 run. We then discard the rest of that run and instead fan out fresh
attackers (seeds A_SEEDS) against this one frozen defender.

We default to the LAST round's defender of each cell (the "final" defender the
exploitability metric already probes), which is the fairest comparison: the
exploitability number in Table II is exactly one draw from this same
distribution.

COMPUTE
-------
Per cell: replay to the final defender (= K defender-trainings + K attacker-
trainings of 10k episodes) then N_ATTACKERS fresh attacker trainings (10k each).
With K=20 that replay is the expensive part. To keep it affordable we expose
--k to shorten the replay (the frozen defender then comes from an earlier round;
set --round to pick which). For the paper, run the default (final defender of
the K=20 cells) on the cells that drive the headline: Q@1%, Q@3%, DQ@1%, DQ@3%,
SARSA@3%.

Usage:
    python run_attacker_variance.py run            # headline cells, 5 attackers
    python run_attacker_variance.py run --full     # all 5 cells x 8 attackers
    python run_attacker_variance.py run_lean       # Q@3% only, fast sanity
    python run_attacker_variance.py analyze        # summarize saved results
"""

import os
import json
import time
import argparse
import numpy as np

from game import train_attacker, train_defender, eval_attacker
from detectors import StaticThresholdDetector

OUT_DIR = "results/attacker_variance"

# Cells that drive the headline (agent, noise). Match the K=20 convergence set.
HEADLINE_CELLS = [
    ("qlearning", 0.01),
    ("qlearning", 0.03),
    ("doubleq",   0.01),
    ("doubleq",   0.03),
    ("sarsa",     0.03),
]
LEAN_CELLS = [("qlearning", 0.03)]

# Fresh-attacker seeds for the fan-out (independent of the game's own seeds).
A_SEEDS_DEFAULT = [10001, 10002, 10003, 10004, 10005]
A_SEEDS_FULL    = [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]

# The master seed of the saved K=20 runs we reconstruct the defender from.
# run_convergence.py used seeds [42, 123]; we reconstruct from 42 by default.
MASTER_SEED = 42


def reconstruct_frozen_defender(noise, agent_type, K, master_seed,
                                round_idx, atk_episodes, def_episodes,
                                honest_frac=0.5, verbose=True):
    """
    Replay run_game's deterministic schedule up to the defender of `round_idx`
    (1-based) and return that frozen defender, reproducing the saved K=20 run
    bit-for-bit (same seed arithmetic as game.run_game).
    """
    if verbose:
        print(f"    reconstructing frozen defender: {agent_type} mu={noise:.0%} "
              f"K={K} master_seed={master_seed} -> round {round_idx} defender")
    # Round 0: attacker vs static baseline (same as run_game).
    static = StaticThresholdDetector()
    attacker = train_attacker(noise, static, atk_episodes, master_seed, agent_type)

    defender = None
    for rnd in range(1, round_idx + 1):
        # defender best-responds to the frozen attacker (same seeds as run_game)
        defender = train_defender(noise, attacker, agent_type, def_episodes,
                                  master_seed + rnd, honest_frac=honest_frac,
                                  defender_seed=master_seed + rnd)
        defender.set_eval()
        if rnd < round_idx:
            # advance: fresh attacker best-responds, exactly as run_game does,
            # so the next round's defender sees the same frozen attacker it would
            attacker = train_attacker(noise, defender,
                                      atk_episodes, master_seed + 100 + rnd,
                                      agent_type)
    defender.set_eval()
    return defender


def run_cell(noise, agent_type, K, round_idx, a_seeds,
             atk_episodes, def_episodes, honest_frac=0.5):
    """Freeze one defender, train len(a_seeds) fresh attackers, record spread."""
    defender = reconstruct_frozen_defender(
        noise, agent_type, K, MASTER_SEED, round_idx,
        atk_episodes, def_episodes, honest_frac=honest_frac)

    dets, kfs = [], []
    for i, aseed in enumerate(a_seeds, 1):
        atk = train_attacker(noise, defender, atk_episodes, aseed, agent_type)
        d, c, kf = eval_attacker(noise, defender, atk, agent_type)
        dets.append(d * 100)
        kfs.append(kf)
        print(f"      attacker {i}/{len(a_seeds)} (seed {aseed}): "
              f"det={d*100:.1f}%  keyfrac={kf:.3f}")

    result = {
        "agent": agent_type, "noise": noise, "K": K,
        "frozen_round": round_idx, "master_seed": MASTER_SEED,
        "attacker_seeds": list(a_seeds),
        "detections_pp": dets, "keyfracs": kfs,
        "det_mean": float(np.mean(dets)), "det_std": float(np.std(dets)),
        "det_min": float(np.min(dets)), "det_max": float(np.max(dets)),
        "det_range": float(np.max(dets) - np.min(dets)),
        "kf_mean": float(np.mean(kfs)), "kf_std": float(np.std(kfs)),
    }
    return result


def run(cells, a_seeds, K=20, round_idx=None, atk_episodes=10000,
        def_episodes=10000, honest_frac=0.5):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    ridx = round_idx if round_idx is not None else K   # default: final defender
    print("=" * 74)
    print(f"ATTACKER-VARIANCE CONTROL (reviewer point 3)")
    print(f"  {len(cells)} cells x {len(a_seeds)} fresh attackers, "
          f"frozen defender = round {ridx} of K={K} (master seed {MASTER_SEED})")
    print(f"  saving to {os.path.abspath(OUT_DIR)} (resumable)")
    print("=" * 74)

    for ci, (agent, noise) in enumerate(cells, 1):
        fname = f"{agent}_noise{int(round(noise*100)):02d}_round{ridx}_K{K}.json"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            print(f"[{ci}/{len(cells)}] SKIP (exists): {fname}")
            continue
        print(f"\n[{ci}/{len(cells)}] {agent} | mu={noise:.0%}")
        res = run_cell(noise, agent, K, ridx, a_seeds,
                       atk_episodes, def_episodes, honest_frac)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"    -> spread: det {res['det_mean']:.1f} +/-{res['det_std']:.1f}pp "
              f"(range {res['det_range']:.1f}pp)   saved {fname}")

    print(f"\nControl complete in {(time.time()-t0)/60:.1f} min.")
    print(f"Now run:  python run_attacker_variance.py analyze")


def analyze():
    if not os.path.isdir(OUT_DIR):
        print(f"No results at {OUT_DIR}. Run the control first.")
        return
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".json"))
    if not files:
        print("No saved results yet.")
        return
    print("=" * 82)
    print("ATTACKER-VARIANCE CONTROL -- spread of fresh attackers vs ONE frozen defender")
    print("  Compare 'attacker-only std' to the round-to-round cycle amplitude (Table III)")
    print("=" * 82)
    print(f"  {'agent':10s} {'mu':>4s} {'n':>3s} "
          f"{'det mean':>9s} {'att-only std':>13s} {'range':>8s} "
          f"{'keyfrac':>9s}")
    print("  " + "-" * 76)
    for f in files:
        with open(os.path.join(OUT_DIR, f)) as fh:
            r = json.load(fh)
        print(f"  {r['agent']:10s} {r['noise']*100:>3.0f}% {len(r['attacker_seeds']):>3d} "
              f"{r['det_mean']:>7.1f}% {r['det_std']:>12.1f}pp "
              f"{r['det_range']:>6.1f}pp {r['kf_mean']:>8.3f}")
    print("=" * 82)
    print("\n  READING:")
    print("  - 'att-only std' is the spread from RETRAINING the attacker against an")
    print("    UNCHANGED defender. The K=20 half-amplitudes (Table III) are the spread")
    print("    across GAME ROUNDS. If att-only std is comparable to the half-amplitude,")
    print("    the swings are largely best-response-oracle noise -> soften 'cycle' to")
    print("    'fluctuates at best-response-variance scale'. If att-only std is much")
    print("    smaller, the round-to-round swings are genuine game dynamics.")
    print("  - Cross-check: Table II exploitability std for each cell is ONE draw of")
    print("    this same attacker-retraining distribution (seed+777), so it should sit")
    print("    inside [det_min, det_max] here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="analyze",
                    choices=["run", "run_lean", "analyze"])
    ap.add_argument("--full", action="store_true",
                    help="all headline cells x 8 attacker seeds")
    ap.add_argument("--k", type=int, default=20, help="K of the run to reconstruct")
    ap.add_argument("--round", type=int, default=None,
                    help="which round's defender to freeze (default: final = K)")
    ap.add_argument("--atk-episodes", type=int, default=10000)
    ap.add_argument("--def-episodes", type=int, default=10000)
    args = ap.parse_args()

    if args.mode == "analyze":
        analyze()
    elif args.mode == "run_lean":
        run(LEAN_CELLS, A_SEEDS_DEFAULT, K=args.k, round_idx=args.round,
            atk_episodes=args.atk_episodes, def_episodes=args.def_episodes)
    else:  # run
        cells = HEADLINE_CELLS
        a_seeds = A_SEEDS_FULL if args.full else A_SEEDS_DEFAULT
        run(cells, a_seeds, K=args.k, round_idx=args.round,
            atk_episodes=args.atk_episodes, def_episodes=args.def_episodes)
