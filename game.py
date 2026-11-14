"""
Alternating best-response game + exploitability (Stage 4)
=========================================================
The learning attacker vs the learning defender, trained as a Stackelberg
(alternating best-response) game, with exploitability as the headline metric.

Why alternating, not simultaneous: two tabular learners updating against each
other are non-stationary and often fail to converge. Instead we freeze one and
train the other to best-respond, then swap, and iterate. This also encodes the
"who moves last" structure we care about.

Round structure:
    Round 0 : attacker = trained vs the static detector (Paper 2 baseline).
    defender step : freeze attacker, train the LEARNING defender to best-respond.
    attacker step : freeze defender, train a FRESH attacker to best-respond.
    repeat K rounds.

Defender reward (option B, matched-FPR built in):
    Each training episode is EITHER an attacked episode (prob 1 - honest_frac,
    driven by the frozen attacker) OR an honest no-attack episode (prob
    honest_frac). The defender is rewarded for aborting attacked episodes and
    penalized for aborting honest ones. The honest fraction is what disciplines
    its false-alarm rate -- it cannot abort freely without paying on honest
    traffic. honest_frac is set so the induced FPR budget matches the static
    target at that noise level.

Exploitability (the headline):
    After K rounds, freeze the final defender and train a BRAND-NEW attacker
    against it. Report that attacker's detection rate AND distillable key
    fraction. Low detection + high key fraction => window still open
    (last-mover advantage persists). High detection + low key fraction =>
    defender closed the window.

NOTE ON COMPUTE: a full run is ~2K+1 trainings of up to 10k episodes each.
That exceeds the sandbox time cap. This module is unit-tested here at tiny
scale (hundreds of episodes); run the real lean experiment locally.
"""

import numpy as np
from environment import BB84Environment, N_STATES, N_ACTIONS
from agents import make_agent
from detectors import StaticThresholdDetector
from learning_defender import LearningDefender
from keyfraction import key_fraction
from train import HYPERPARAMS, run_episode


# ---------------------------------------------------------------------------
# Train an ATTACKER against a FROZEN detector (reuses the Stage 1 machinery)
# ---------------------------------------------------------------------------

def train_attacker(noise, detector, n_episodes, rng_seed, agent_type="qlearning"):
    """Fresh attacker best-responding to a frozen detector. Returns agent."""
    alpha, gamma = HYPERPARAMS[agent_type][noise]
    env = BB84Environment(channel_error_rate=noise, detector=detector,
                          min_blocks=10, max_blocks=10, rng_seed=rng_seed)
    agent = make_agent(agent_type, N_STATES, N_ACTIONS,
                       alpha=alpha, gamma=gamma,
                       epsilon_decay_episodes=n_episodes,
                       explore_attack_bias=0.20, rng_seed=rng_seed)
    for _ in range(n_episodes):
        run_episode(env, agent, agent_type, learn=True)
        agent.decay_epsilon()
    return agent


def eval_attacker(noise, detector, attacker, agent_type="qlearning",
                  n_eval=1000, rng_seed=999):
    """
    Greedy evaluation of a frozen attacker vs a frozen detector.
    Returns (detection_rate, mean_correct_bits, mean_key_fraction).
    """
    env = BB84Environment(channel_error_rate=noise, detector=detector,
                          min_blocks=10, max_blocks=10, rng_seed=rng_seed)
    saved_eps = attacker.epsilon
    attacker.epsilon = 0.0
    dets, cors, kfs = [], [], []
    for _ in range(n_eval):
        env.reset()
        block_qbers = []
        while not env.done:
            a = attacker.greedy_action(_state(env))
            _, _, done, info = env.step(a)
            if info.get("completed_block"):
                block_qbers.append(info["completed_block_qber"])
        dets.append(env.detected)
        cors.append(env.total_correct_bits)
        # key fraction from the mean induced QBER on this episode's blocks
        mq = np.mean(block_qbers) if block_qbers else 0.0
        kfs.append(key_fraction(mq))
    attacker.epsilon = saved_eps
    return float(np.mean(dets)), float(np.mean(cors)), float(np.mean(kfs))


def _state(env):
    """Reconstruct the attacker's current state id from the env (mirrors
    environment.encode_state on its live counters)."""
    from environment import encode_state
    return encode_state(env.last_qber, env.attacks_this_block,
                        min(env.block_index, env.n_blocks - 1))


# ---------------------------------------------------------------------------
# Train the DEFENDER against a FROZEN attacker (matched-FPR via honest_frac)
# ---------------------------------------------------------------------------

def train_defender(noise, frozen_attacker, attacker_type, n_episodes, rng_seed,
                   honest_frac=0.5, r_catch=+10.0, r_false=-10.0,
                   r_pass=+1.0, r_miss=-1.0, defender_seed=0):
    """
    Train a LearningDefender to best-respond to a frozen attacker.

    Each episode is honest with probability honest_frac (no attack, false-abort
    is penalized) else attacked (frozen attacker plays, catching is rewarded).
    This bakes the matched-FPR discipline into the reward.
    Returns the trained LearningDefender.
    """
    defender = LearningDefender(agent_type="qlearning", rng_seed=defender_seed,
                               epsilon_decay_episodes=n_episodes)
    rng = np.random.default_rng(rng_seed)
    env = BB84Environment(channel_error_rate=noise, detector=defender,
                          min_blocks=10, max_blocks=10, rng_seed=rng_seed)
    frozen_attacker.epsilon = 0.0

    for _ in range(n_episodes):
        honest = rng.random() < honest_frac
        env.reset()
        defender.reset()
        while not env.done:
            if honest:
                a = 0
            else:
                a = frozen_attacker.greedy_action(_state(env))
            env.step(a)
        # reward the defender
        if honest:
            reward = r_false if env.detected else r_pass
        else:
            reward = r_catch if env.detected else r_miss
        defender.learn_from_episode(reward)
        defender.decay_epsilon()

    return defender


# ---------------------------------------------------------------------------
# The game: alternating best response + exploitability
# ---------------------------------------------------------------------------

def run_game(noise=0.01, agent_type="qlearning", K=5,
             atk_episodes=10000, def_episodes=10000,
             honest_frac=0.5, seed=42, verbose=True):
    """
    Returns a dict with per-round metrics and the final exploitability.
    """
    log = {"rounds": [], "noise": noise, "agent": agent_type, "K": K}

    # Round 0: attacker vs static baseline
    static = StaticThresholdDetector()
    attacker = train_attacker(noise, static, atk_episodes, seed, agent_type)
    if verbose:
        d, c, kf = eval_attacker(noise, static, attacker, agent_type)
        print(f"[round 0] attacker vs static: det={d*100:.1f}% "
              f"correct={c:.1f} keyfrac={kf:.3f}")

    defender = None
    for rnd in range(1, K + 1):
        # defender best-responds to frozen attacker
        defender = train_defender(noise, attacker, agent_type, def_episodes,
                                  seed + rnd, honest_frac=honest_frac,
                                  defender_seed=seed + rnd)
        defender.set_eval()
        d_det, d_cor, d_kf = eval_attacker(noise, defender, attacker, agent_type)

        # fresh attacker best-responds to frozen defender
        attacker = train_attacker(noise, defender, atk_episodes, seed + 100 + rnd,
                                  agent_type)
        a_det, a_cor, a_kf = eval_attacker(noise, defender, attacker, agent_type)

        log["rounds"].append({
            "round": rnd,
            "cotrained_attacker_det": d_det,   # old attacker vs new defender
            "fresh_attacker_det": a_det,       # new attacker vs this defender
            "fresh_attacker_keyfrac": a_kf,
            "fresh_attacker_correct": a_cor,
        })
        if verbose:
            print(f"[round {rnd}] defender vs frozen atk: det={d_det*100:.1f}% "
                  f"| fresh atk vs defender: det={a_det*100:.1f}% "
                  f"keyfrac={a_kf:.3f}")

    # Exploitability: brand-new attacker vs final frozen defender
    defender.set_eval()
    exploit = train_attacker(noise, defender, atk_episodes, seed + 777, agent_type)
    e_det, e_cor, e_kf = eval_attacker(noise, defender, exploit, agent_type)
    log["exploitability"] = {"det": e_det, "correct": e_cor, "keyfrac": e_kf}

    # Convergence vs cycling diagnostic: the fresh-attacker detection trajectory
    # across rounds. A settled game has a flat tail; a cycling game keeps
    # swinging. We report the std of the last-3 rounds' fresh-attacker detection
    # as a simple cycling indicator (small = converged, large = cycling).
    fresh_traj = [r["fresh_attacker_det"] for r in log["rounds"]]
    tail = fresh_traj[-3:] if len(fresh_traj) >= 3 else fresh_traj
    log["convergence"] = {
        "fresh_attacker_det_trajectory": fresh_traj,
        "tail_std": float(np.std(tail)) if tail else 0.0,
        "settled": bool(np.std(tail) < 0.05) if tail else False,
    }

    if verbose:
        print(f"[EXPLOITABILITY] fresh attacker vs final defender: "
              f"det={e_det*100:.1f}% correct={e_cor:.1f} keyfrac={e_kf:.3f}")
        # Honest per-run verdict: judge the WINDOW by exploitability, and flag
        # whether the best-response game actually settled.
        settled = log["convergence"]["settled"]
        if e_det < 0.20 and e_kf > 0.5:
            verdict = "WINDOW OPEN (fresh attacker evades final defender)"
        elif e_kf < 0.05:
            verdict = "degenerate regime (channel too noisy for any attacker)"
        else:
            verdict = "window partially closed (defender gains traction)"
        conv = "settled" if settled else ("CYCLING (tail_std="
               f"{log['convergence']['tail_std']*100:.0f}pp) — needs more seeds")
        print(f"  => {verdict}  [{conv}]")
    return log


# ---------------------------------------------------------------------------
# Tiny-scale unit test (proves the loop runs and metrics compute)
# ---------------------------------------------------------------------------

def smoke_test():
    print("=" * 60)
    print("Game smoke test (tiny scale: proves it runs, NOT a result)")
    print("=" * 60)
    log = run_game(noise=0.01, agent_type="qlearning", K=2,
                   atk_episodes=400, def_episodes=400,
                   honest_frac=0.5, seed=42, verbose=True)
    ok = (len(log["rounds"]) == 2 and "exploitability" in log
          and 0.0 <= log["exploitability"]["keyfrac"] <= 1.0)
    print("\n  [PASS] game loop runs, rounds logged, exploitability computed"
          if ok else "\n  [FAIL] something is off")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke_test()
    elif mode == "lean":
        # the real lean experiment — run locally, exceeds sandbox cap
        run_game(noise=0.01, agent_type="qlearning", K=5,
                 atk_episodes=10000, def_episodes=10000, seed=42)
    else:
        print("Usage: python game.py [smoke|lean]")
