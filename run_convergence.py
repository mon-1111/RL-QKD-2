"""
Longer-K convergence confirmation (Stage 4 keystone)
====================================================
The cheap K=5 analysis suggested the best-response game does not converge --
strongest at mu=3% (all agents cycle), agent-dependent at mu=1% (Double-Q
cycles, Q-Learning settles). K=5 is too short to be definitive (2-3 points per
half). This runner re-runs the targeted cells out to K=20, where each
trajectory has ~10 points per half -- enough to tell a PERSISTENT cycle (large
amplitude through round 20) from SLOW CONVERGENCE (amplitude decays toward 0).

Targeted cells (the "thorough" set):
    mu=3%  qlearning, sarsa, doubleq   x 2 seeds  -> confirm unanimous cycling
    mu=1%  doubleq                     x 2 seeds  -> confirm its cycle is real
    mu=1%  qlearning                   x 2 seeds  -> confirm it genuinely settles
                                                      (the clean contrast)
= 10 runs at K=20. Each K=20 game is ~4x a K=5 game (~40-50 min on a laptop),
so budget ~4-6 hours. RESUMABLE: finished runs are skipped on restart.

Saves to results/convergence/<agent>_noise<NN>_seed<S>_K20.json with the full
per-round trajectory. Then:  python analyze_convergence.py results/convergence

Usage:
    python run_convergence.py run       # the 10-run K=20 confirmation (local)
    python run_convergence.py run_lean  # just mu=3% qlearning x2 (fast check)
"""

import os
import json
import time
from game import run_game

CONV_DIR = "results/convergence"
K_LONG = 20
SEEDS2 = [42, 123, 456, 789, 1337]      # five seeds per targeted cell

# (agent, noise) cells to confirm, each run over SEEDS2
THOROUGH_CELLS = [
    ("qlearning", 0.03),   # mu=3% unanimous cycling
    ("sarsa",     0.03),
    ("doubleq",   0.03),
    ("doubleq",   0.01),   # mu=1% Double-Q: confirm cycle is real
    ("qlearning", 0.01),   # mu=1% Q-Learning: confirm it settles (contrast)
]
LEAN_CELLS = [("qlearning", 0.03)]


def _fname(agent, noise, seed):
    return f"{agent}_noise{int(round(noise*100)):02d}_seed{seed}_K{K_LONG}.json"


def run_confirmation(cells, seeds=SEEDS2):
    os.makedirs(CONV_DIR, exist_ok=True)
    total = len(cells) * len(seeds)
    done = 0
    t0 = time.time()
    print("=" * 66)
    print(f"Longer-K convergence confirmation: K={K_LONG}, "
          f"{len(cells)} cells x {len(seeds)} seeds = {total} runs")
    print(f"  saving to {os.path.abspath(CONV_DIR)} (resumable)")
    print("=" * 66)

    for agent, noise in cells:
        for seed in seeds:
            done += 1
            path = os.path.join(CONV_DIR, _fname(agent, noise, seed))
            if os.path.exists(path):
                print(f"[{done}/{total}] SKIP (exists): {_fname(agent,noise,seed)}")
                continue
            print(f"\n[{done}/{total}] {agent} | mu={noise:.0%} | seed={seed} "
                  f"| K={K_LONG}")
            log = run_game(noise=noise, agent_type=agent, K=K_LONG,
                           atk_episodes=10000, def_episodes=10000,
                           honest_frac=0.5, seed=seed, verbose=True)
            with open(path, "w") as f:
                json.dump(log, f, indent=2)
            print(f"    saved -> {path}")

    print(f"\nConfirmation complete in {(time.time()-t0)/60:.1f} min.")
    print(f"Now run:  python analyze_convergence.py {CONV_DIR}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run_confirmation(THOROUGH_CELLS)
    elif mode == "run_lean":
        run_confirmation(LEAN_CELLS)
    else:
        print("Usage: python run_convergence.py [run|run_lean]")
