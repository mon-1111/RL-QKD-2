"""
Multi-seed sweep for the learning-defender game (Stage 4)
========================================================
Runs the alternating best-response game across the full factorial
    3 agents x 3 noise x 5 seeds = 45 game runs
and saves each run's log to disk so the sweep is resumable (finished runs are
skipped on restart). Then `analyze` aggregates exploitability (detection AND
key fraction) as mean +/- std per agent x noise, which is the table that shows
whether the window-open result holds across all conditions or is a seed
artifact.

This is the compute-heavy step. Each game run is ~11 trainings of 10k episodes
(K=5 rounds x 2 players + 1 exploitability), ~10-15 min, so the full sweep is
hours. RUN IT LOCALLY. It is resumable: re-run after an interruption and it
continues from where it stopped.

Usage:
    python run_sweep.py run        # the full 45-run sweep (resumable), local
    python run_sweep.py run_quick  # 1 seed only (9 runs) for a faster check
    python run_sweep.py analyze    # aggregate saved logs into the summary table
"""

import os
import json
import time
import numpy as np
from game import run_game

AGENTS = ["qlearning", "sarsa", "doubleq"]
NOISES = [0.01, 0.03, 0.05]
SEEDS  = [42, 123, 456, 789, 1337]
SWEEP_DIR = "results/sweep"


def _fname(agent, noise, seed):
    return f"{agent}_noise{int(round(noise*100)):02d}_seed{seed}.json"


def run_sweep(seeds=SEEDS, K=5, atk_episodes=10000, def_episodes=10000,
              honest_frac=0.5):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    total = len(AGENTS) * len(NOISES) * len(seeds)
    done = 0
    t0 = time.time()
    print("=" * 64)
    print(f"Learning-defender sweep: {len(AGENTS)} agents x {len(NOISES)} "
          f"noise x {len(seeds)} seeds = {total} runs")
    print(f"  saving to {os.path.abspath(SWEEP_DIR)} (resumable)")
    print("=" * 64)

    for agent in AGENTS:
        for noise in NOISES:
            for seed in seeds:
                done += 1
                path = os.path.join(SWEEP_DIR, _fname(agent, noise, seed))
                if os.path.exists(path):
                    print(f"[{done}/{total}] SKIP (exists): {_fname(agent,noise,seed)}")
                    continue
                print(f"\n[{done}/{total}] {agent} | mu={noise:.0%} | seed={seed}")
                log = run_game(noise=noise, agent_type=agent, K=K,
                               atk_episodes=atk_episodes,
                               def_episodes=def_episodes,
                               honest_frac=honest_frac, seed=seed, verbose=True)
                with open(path, "w") as f:
                    json.dump(log, f, indent=2)
                print(f"    saved -> {path}")

    print(f"\nSweep complete in {(time.time()-t0)/60:.1f} min.")


def analyze():
    """Aggregate saved sweep logs into exploitability mean +/- std per cell."""
    if not os.path.isdir(SWEEP_DIR):
        print(f"No sweep directory at {SWEEP_DIR}. Run the sweep first.")
        return
    files = [f for f in os.listdir(SWEEP_DIR) if f.endswith(".json")]
    if not files:
        print("No saved runs yet.")
        return

    # cell[(agent,noise)] = list of dicts with exploit + convergence info
    cell = {}
    for f in files:
        with open(os.path.join(SWEEP_DIR, f)) as fh:
            log = json.load(fh)
        key = (log["agent"], log["noise"])
        e = log["exploitability"]
        conv = log.get("convergence", {})
        cell.setdefault(key, []).append({
            "det": e["det"], "kf": e["keyfrac"], "cor": e["correct"],
            "settled": conv.get("settled"), "tail_std": conv.get("tail_std"),
        })

    print("=" * 78)
    print("EXPLOITABILITY of the final learned defender (mean +/- std over seeds)")
    print("  fresh attacker vs final defender: low det + high keyfrac = WINDOW OPEN")
    print("=" * 78)
    print(f"  {'agent':12s} {'noise':6s} {'n':>3s} "
          f"{'detection %':>15s} {'key fraction':>15s} {'settled?':>10s}")
    print("  " + "-" * 72)
    for agent in AGENTS:
        for noise in NOISES:
            vals = cell.get((agent, noise))
            if not vals:
                print(f"  {agent:12s} {noise:<6.0%} {'-':>3s}  (no data)")
                continue
            dets = np.array([v["det"] for v in vals]) * 100
            kfs  = np.array([v["kf"] for v in vals])
            n_settled = sum(1 for v in vals if v["settled"])
            print(f"  {agent:12s} {noise:<6.0%} {len(vals):>3d} "
                  f"{dets.mean():>7.2f} +/-{dets.std():>4.2f} "
                  f"{kfs.mean():>8.3f} +/-{kfs.std():>4.3f} "
                  f"{n_settled:>5d}/{len(vals):<4d}")
    print("=" * 78)

    # Per-REGIME verdict (averaging across regimes is meaningless; μ=5% is
    # degenerate, so the window question only makes sense per noise level).
    print("\n  Verdict by noise regime (window = can a fresh attacker still"
          " evade?):")
    for noise in NOISES:
        ds, ks, settled = [], [], []
        for agent in AGENTS:
            for v in cell.get((agent, noise), []):
                ds.append(v["det"]); ks.append(v["kf"])
                settled.append(v["settled"])
        if not ds:
            continue
        md, mk = np.mean(ds) * 100, np.mean(ks)
        frac_settled = (sum(1 for s in settled if s) / len(settled)
                        if settled else 0)
        if mk < 0.05:
            tag = "DEGENERATE (channel too noisy for any attacker)"
        elif md < 20 and mk > 0.5:
            tag = "WINDOW OPEN (learning defender does not close it)"
        elif md > 50:
            tag = "window closed (defender contains fresh attacker)"
        else:
            tag = "PARTIAL / unsettled (inspect convergence)"
        print(f"    mu={noise:<4.0%}: det {md:5.1f}%, keyfrac {mk:.3f}, "
              f"{frac_settled*100:.0f}% of runs settled  => {tag}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "run":
        run_sweep()
    elif mode == "run_quick":
        run_sweep(seeds=[42])
    elif mode == "analyze":
        analyze()
    else:
        print("Usage: python run_sweep.py [run|run_quick|analyze]")
