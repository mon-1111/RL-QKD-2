"""
Defender-policy check (reviewer point 2 -- decides the title)
=============================================================
Question: does the defender actually keep changing across best-response rounds,
or has it settled to one policy while only the attacker's retraining noise makes
the exploitability wobble?

  defenders KEEP CHANGING across rounds  -> genuine non-convergence of the
                                            strategy profile. "No Stable
                                            Equilibrium" is earned.
  defenders have SETTLED (same greedy policy round to round) while exploitability
  still swings -> the swing is attacker training noise around a fixed defender.
                  The title should soften to "... Do Not Settle".

The defender's greedy policy is a length-160 vector of chosen threshold-action
indices, one per (QBER bin x block index) state. We reconstruct each round's
defender deterministically (identical seed arithmetic to game.run_game, same as
run_attacker_variance.py), read off its greedy policy, and measure how much
successive policies differ.

Metrics per cell:
  - mean round-to-round Hamming distance between successive greedy policies,
    as a fraction of the 160 states (0 = identical policy every round, i.e.
    settled; large = the policy keeps changing).
  - the same restricted to REACHABLE states only (states the greedy attacker of
    that round actually visits), since unreached states carry an arbitrary
    argmax and inflate the raw Hamming distance.
  - number of distinct policies across the K rounds.

COMPUTE: per cell this replays the K=20 schedule once (the expensive part, same
as one K=20 game), with no extra attacker fan-out, so it is cheaper than the
attacker-variance control. Run the five targeted cells.

Usage:
    python run_defender_policy_check.py run       # 5 targeted cells, K=20
    python run_defender_policy_check.py run_lean  # Q@3% only
    python run_defender_policy_check.py analyze
"""

import os
import json
import time
import argparse
import numpy as np

from game import train_attacker, train_defender, eval_attacker, _state
from detectors import StaticThresholdDetector
from environment import BB84Environment
from learning_defender import (N_STATES_D, encode_state_d, THRESHOLDS,
                               QBER_BINS_D, BLOCK_BINS_D)


def _state_qber_value(state_id):
    """Representative QBER for a defender state's QBER bin (bin lower edge).

    State encodes qber_bin * BLOCK_BINS_D + block_bin, and _bin_qber maps a
    qber to int(qber*100) capped at 15. We invert to the bin's lower edge in
    QBER units, e.g. bin 9 -> 0.09, which is the value at which the
    abort/continue decision is evaluated.
    """
    qbin = state_id // BLOCK_BINS_D
    return qbin / 100.0


def _decision_vector(policy):
    """Map a length-N greedy threshold-index policy to a boolean abort decision
    per state, evaluated at that state's representative QBER. This is what the
    defender actually DOES, collapsing threshold labels that give the same
    abort/continue outcome (reviewer point 2)."""
    dec = np.zeros(len(policy), dtype=bool)
    for s, a in enumerate(policy):
        q = _state_qber_value(s)
        dec[s] = q >= THRESHOLDS[int(a)]
    return dec

OUT_DIR = "results/defender_policy"

HEADLINE_CELLS = [
    ("qlearning", 0.01),
    ("qlearning", 0.03),
    ("doubleq",   0.01),
    ("doubleq",   0.03),
    ("sarsa",     0.03),
]
LEAN_CELLS = [("qlearning", 0.03)]

MASTER_SEED = 42
K_LONG = 20


def greedy_policy(defender):
    """Length-N_STATES_D vector of greedy threshold-action indices."""
    agent = defender.agent
    # Double-Q uses Q+Q2 for greedy; others use Q. greedy_action handles both,
    # so call it per state for correctness across agent types.
    return np.array([agent.greedy_action(s) for s in range(N_STATES_D)],
                    dtype=int)


def reachable_states(noise, defender, attacker, agent_type, n_eval=300,
                     rng_seed=4242):
    """Set of defender states visited by the greedy attacker vs this defender."""
    env = BB84Environment(channel_error_rate=noise, detector=defender,
                          min_blocks=10, max_blocks=10, rng_seed=rng_seed)
    saved = attacker.epsilon
    attacker.epsilon = 0.0
    visited = set()
    for _ in range(n_eval):
        env.reset()
        while not env.done:
            # the defender decides on each completed block; mirror its state key
            s_att = _state(env)
            a = attacker.greedy_action(s_att)
            # record the defender state for the current block/qber
            visited.add(encode_state_d(env.last_qber,
                                       min(env.block_index, env.n_blocks - 1)))
            env.step(a)
    attacker.epsilon = saved
    return visited


def run_cell(noise, agent_type, K, master_seed, atk_episodes, def_episodes,
             honest_frac=0.5):
    """Replay the schedule, collecting each round's defender greedy policy."""
    static = StaticThresholdDetector()
    attacker = train_attacker(noise, static, atk_episodes, master_seed, agent_type)

    policies = []
    reach_union = set()
    for rnd in range(1, K + 1):
        defender = train_defender(noise, attacker, agent_type, def_episodes,
                                  master_seed + rnd, honest_frac=honest_frac,
                                  defender_seed=master_seed + rnd)
        defender.set_eval()
        pol = greedy_policy(defender)
        policies.append(pol)
        # reachable states for THIS round (attacker is the one about to be
        # replaced, which is the frozen attacker this defender saw)
        reach_union |= reachable_states(noise, defender, attacker, agent_type)
        # advance attacker exactly as run_game does
        attacker = train_attacker(noise, defender, atk_episodes,
                                  master_seed + 100 + rnd, agent_type)

    P = np.vstack(policies)                       # K x 160 threshold labels
    D = np.vstack([_decision_vector(p) for p in policies])  # K x 160 abort bool
    reach = sorted(reach_union)
    # decision-relevant reachable states: those whose abort decision is NOT the
    # same under all five thresholds (i.e. QBER bin in the 8-11% band). In the
    # others every threshold gives the same outcome, so a label flip is inert.
    def _decision_relevant(s):
        q = _state_qber_value(s)
        outcomes = {q >= t for t in THRESHOLDS}
        return len(outcomes) > 1
    reach_relevant = [s for s in reach if _decision_relevant(s)]

    label_frac, dec_frac_reach, dec_frac_relevant = [], [], []
    for i in range(1, len(policies)):
        lbl_diff = (P[i] != P[i - 1])
        dec_diff = (D[i] != D[i - 1])
        if reach:
            label_frac.append(float(lbl_diff[reach].mean()))
            dec_frac_reach.append(float(dec_diff[reach].mean()))
        if reach_relevant:
            dec_frac_relevant.append(float(dec_diff[reach_relevant].mean()))

    # distinct policies by DECISION over reachable states (the meaningful count)
    if reach:
        sigs_dec = {tuple(D[i][reach]) for i in range(len(policies))}
    else:
        sigs_dec = {tuple(D[i]) for i in range(len(policies))}

    return {
        "agent": agent_type, "noise": noise, "K": K, "master_seed": master_seed,
        "n_reachable_states": len(reach),
        "n_reachable_decision_relevant": len(reach_relevant),
        "mean_label_change_reachable": float(np.mean(label_frac)) if label_frac else 0.0,
        "mean_decision_change_reachable": float(np.mean(dec_frac_reach)) if dec_frac_reach else 0.0,
        "mean_decision_change_relevant": float(np.mean(dec_frac_relevant)) if dec_frac_relevant else 0.0,
        "distinct_decision_policies_reachable": len(sigs_dec),
        "decision_change_relevant_per_round": dec_frac_relevant,
    }


def run_defender_variance(cells, n_def=5, atk_episodes=10000,
                          def_episodes=10000, honest_frac=0.5):
    """
    Defender-side analogue of the attacker-variance control (Table IV).
    Freeze ONE attacker (the round-0 attacker vs the static baseline, same for
    all seeds) and retrain the defender n_def times with different seeds. Measure
    the spread of the resulting defenders' ABORT DECISIONS over decision-relevant
    reachable states. A large spread would mean the defender's best-response is
    itself noisy, so that retraining it against an UNCHANGED attacker already
    moves its decisions -- the same 'oracle noise' caveat Table IV answered for
    the attacker, now answered for the defender.
    """
    out_dir = OUT_DIR + "_variance"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("DEFENDER-VARIANCE CONTROL (reviewer point 2, second half)")
    print(f"  {len(cells)} cells x {n_def} defender re-seeds vs one frozen attacker")
    print(f"  saving to {os.path.abspath(out_dir)} (resumable)")
    print("=" * 74)
    for ci, (agent, noise) in enumerate(cells, 1):
        fname = f"{agent}_noise{int(round(noise*100)):02d}_defvar.json"
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            print(f"[{ci}/{len(cells)}] SKIP (exists): {fname}")
            continue
        print(f"\n[{ci}/{len(cells)}] {agent} | mu={noise:.0%}")
        static = StaticThresholdDetector()
        attacker = train_attacker(noise, static, atk_episodes, MASTER_SEED, agent)
        decisions = []
        reach_union = set()
        for j in range(n_def):
            dseed = MASTER_SEED + 2000 + j
            defn = train_defender(noise, attacker, agent, def_episodes,
                                  dseed, honest_frac=honest_frac,
                                  defender_seed=dseed)
            defn.set_eval()
            decisions.append(_decision_vector(greedy_policy(defn)))
            reach_union |= reachable_states(noise, defn, attacker, agent)
        reach = sorted(reach_union)
        reach_relevant = [s for s in reach
                          if len({(s // BLOCK_BINS_D)/100.0 >= t
                                  for t in THRESHOLDS}) > 1]
        Dd = np.vstack(decisions)
        # pairwise mean decision disagreement over decision-relevant states
        import itertools
        pdiffs = []
        for a, b in itertools.combinations(range(n_def), 2):
            if reach_relevant:
                pdiffs.append(float((Dd[a][reach_relevant]
                                     != Dd[b][reach_relevant]).mean()))
        res = {
            "agent": agent, "noise": noise, "n_def": n_def,
            "n_reachable_decision_relevant": len(reach_relevant),
            "mean_pairwise_decision_disagreement": float(np.mean(pdiffs)) if pdiffs else 0.0,
        }
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"    decision-relevant states={len(reach_relevant)}, "
              f"mean pairwise decision disagreement="
              f"{res['mean_pairwise_decision_disagreement']*100:.1f}%")
    print(f"\nDefender-variance control complete in {(time.time()-t0)/60:.1f} min.")


def analyze_variance():
    out_dir = OUT_DIR + "_variance"
    if not os.path.isdir(out_dir):
        print(f"No results at {out_dir}. Run run_defvar first.")
        return
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json"))
    print("=" * 72)
    print("DEFENDER-VARIANCE CONTROL -- spread of decisions when the attacker is FIXED")
    print("=" * 72)
    print(f"  {'agent':10s} {'mu':>4s} {'relevant states':>16s} "
          f"{'pairwise dec disagreement':>26s}")
    print("  " + "-" * 60)
    for f in files:
        with open(os.path.join(out_dir, f)) as fh:
            r = json.load(fh)
        print(f"  {r['agent']:10s} {r['noise']*100:>3.0f}% "
              f"{r['n_reachable_decision_relevant']:>16d} "
              f"{r['mean_pairwise_decision_disagreement']*100:>24.1f}%")
    print("=" * 72)
    print("\n  READING: compare this to the round-to-round DECISION change from the")
    print("  main check. If the round-to-round change is clearly LARGER than this")
    print("  re-seed spread, the defender genuinely moves with the evolving attacker")
    print("  (not just oracle noise). If comparable, the defender's motion is largely")
    print("  its own best-response variance.")


def run(cells, K=K_LONG, atk_episodes=10000, def_episodes=10000,
        honest_frac=0.5):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("DEFENDER-POLICY CHECK (reviewer point 2 -- decides the title)")
    print(f"  {len(cells)} cells, K={K}, master seed {MASTER_SEED}")
    print(f"  saving to {os.path.abspath(OUT_DIR)} (resumable)")
    print("=" * 74)
    for ci, (agent, noise) in enumerate(cells, 1):
        fname = f"{agent}_noise{int(round(noise*100)):02d}_K{K}.json"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            print(f"[{ci}/{len(cells)}] SKIP (exists): {fname}")
            continue
        print(f"\n[{ci}/{len(cells)}] {agent} | mu={noise:.0%}")
        res = run_cell(noise, agent, K, MASTER_SEED, atk_episodes, def_episodes,
                       honest_frac)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"    reachable={res['n_reachable_states']} "
              f"(decision-relevant {res['n_reachable_decision_relevant']}) | "
              f"label change={res['mean_label_change_reachable']*100:.1f}% | "
              f"DECISION change (relevant states)="
              f"{res['mean_decision_change_relevant']*100:.1f}% | "
              f"distinct decision policies="
              f"{res['distinct_decision_policies_reachable']}/{K}")
    print(f"\nCheck complete in {(time.time()-t0)/60:.1f} min.")
    print("Now run:  python run_defender_policy_check.py analyze")


def analyze():
    if not os.path.isdir(OUT_DIR):
        print(f"No results at {OUT_DIR}. Run the check first.")
        return
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".json"))
    if not files:
        print("No saved results yet.")
        return
    print("=" * 96)
    print("DEFENDER-POLICY CHECK -- does the defender's ABORT DECISION keep changing across rounds?")
    print("  (decision-level, reviewer point 2: collapses threshold labels that give the same outcome)")
    print("=" * 96)
    print(f"  {'agent':10s} {'mu':>4s} {'reach':>6s} {'relevant':>9s} "
          f"{'label chg':>10s} {'DECISION chg':>13s} {'distinct dec':>13s}  verdict")
    print("  " + "-" * 86)
    stale = []
    for f in files:
        with open(os.path.join(OUT_DIR, f)) as fh:
            r = json.load(fh)
        if "mean_decision_change_relevant" not in r:
            stale.append(f)
            continue
        lbl = r["mean_label_change_reachable"] * 100
        dec = r["mean_decision_change_relevant"] * 100
        dist = r["distinct_decision_policies_reachable"]
        # verdict keyed on the DECISION-level change over decision-relevant states
        if dec < 5.0 and dist <= 2:
            verdict = "SETTLED (soften title)"
        elif dec < 15.0:
            verdict = "mostly stable (inspect)"
        else:
            verdict = "KEEPS CHANGING (title earned)"
        print(f"  {r['agent']:10s} {r['noise']*100:>3.0f}% "
              f"{r['n_reachable_states']:>6d} "
              f"{r['n_reachable_decision_relevant']:>9d} "
              f"{lbl:>9.1f}% {dec:>12.1f}% {dist:>10d}/{r['K']:<3d}  {verdict}")
    if stale:
        print("  " + "-" * 86)
        print(f"  {len(stale)} file(s) are from the OLD label-only metric and were skipped:")
        for f in stale:
            print(f"      {f}")
        print(f"  Delete this folder and rerun to recompute with the decision metric:")
        print(f"      rm -rf {OUT_DIR} && python3 run_defender_policy_check.py run")
    print("=" * 96)
    print("\n  READING:")
    print("  - 'label chg' is the inflated metric (any threshold-label flip, incl. inert ones).")
    print("  - 'DECISION chg' is the fraction of DECISION-RELEVANT reachable states (QBER bin in")
    print("    the 8-11% band, where the threshold actually matters) whose ABORT/CONTINUE decision")
    print("    changes round to round. This is the honest measure. If it is substantial and")
    print("    sustained, the defender genuinely keeps changing what it does -> 'No Stable")
    print("    Equilibrium' earned. If it is small, the defender is behaviorally close to settled")
    print("    and the title should soften.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="analyze",
                    choices=["run", "run_lean", "analyze",
                             "run_defvar", "analyze_defvar"])
    ap.add_argument("--atk-episodes", type=int, default=10000)
    ap.add_argument("--def-episodes", type=int, default=10000)
    args = ap.parse_args()
    if args.mode == "analyze":
        analyze()
    elif args.mode == "analyze_defvar":
        analyze_variance()
    elif args.mode == "run_defvar":
        run_defender_variance(HEADLINE_CELLS, atk_episodes=args.atk_episodes,
                              def_episodes=args.def_episodes)
    elif args.mode == "run_lean":
        run(LEAN_CELLS, atk_episodes=args.atk_episodes,
            def_episodes=args.def_episodes)
    else:
        run(HEADLINE_CELLS, atk_episodes=args.atk_episodes,
            def_episodes=args.def_episodes)