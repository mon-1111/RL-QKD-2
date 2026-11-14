"""
Training loop + evaluation
==========================
Drives one agent through the environment for N episodes, then evaluates its
greedy policy. This is where learning actually happens and where we reproduce
Paper 1's headline numbers.

The loop is agent-agnostic with one wrinkle: SARSA is on-policy, so it needs the
NEXT action (a') to update, while Q-Learning and Double-Q only need (s,a,r,s').
We therefore pick the next action before calling update() and pass it through;
the non-SARSA agents simply ignore it.

Hyperparameters are the Paper 1 Table 1 values, keyed by (agent, mu), so a
reproduction run uses exactly what the accepted paper used.

Metrics (per episode)
---------------------
    detected      : did the detector abort this episode?
    correct_bits  : raw bits Eve learned (right-basis interceptions)
    reward        : total episode reward
We report means over the final `eval_window` episodes, matching Paper 1's
"final 500 episodes" convention.
"""

import numpy as np
from dataclasses import dataclass, field
from environment import BB84Environment, N_STATES, N_ACTIONS
from agents import make_agent
from detectors import StaticThresholdDetector


# Paper 1 Table 1: (alpha, gamma) by agent and noise level.
HYPERPARAMS = {
    "qlearning": {0.01: (0.50, 0.50), 0.03: (0.50, 0.50), 0.05: (0.50, 0.50)},
    "sarsa":     {0.01: (0.01, 0.99), 0.03: (0.10, 0.99), 0.05: (0.50, 0.99)},
    "doubleq":   {0.01: (0.50, 0.50), 0.03: (0.50, 0.50), 0.05: (0.50, 0.75)},
}


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    detected: bool
    correct_bits: int
    attacks_per_block: list = field(default_factory=list)


@dataclass
class RunResult:
    agent_type: str
    noise: float
    detector_name: str
    logs: list = field(default_factory=list)

    def tail(self, window=500):
        return self.logs[-window:]

    def detection_rate(self, window=500):
        t = self.tail(window)
        return float(np.mean([e.detected for e in t])) if t else float("nan")

    def mean_correct(self, window=500):
        t = self.tail(window)
        return float(np.mean([e.correct_bits for e in t])) if t else float("nan")

    def mean_reward(self, window=500):
        t = self.tail(window)
        return float(np.mean([e.reward for e in t])) if t else float("nan")


def run_episode(env, agent, agent_type, learn=True):
    """
    One episode. Returns (reward, detected, correct_bits, attacks_per_block).

    The loop structure must match Paper 1 EXACTLY, because the agent's
    exploration RNG stream depends on how many times select_action is called
    per step. SARSA needs the next action to update (so it selects once, then
    reuses that selection as the next step's action). Q-Learning and Double-Q
    select fresh each step. Mixing these — e.g. pre-selecting next_action for
    all agents — draws an extra random per step and desynchronizes the stream,
    changing the learned policy even when the physics is identical.
    """
    s = env.reset()
    total_r = 0.0
    info_detected = False

    if learn and agent_type == "sarsa":
        a = agent.select_action(s)
        done = False
        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = agent.select_action(s_next)   # also used as next action
            agent.update(s, a, r, s_next, done, next_action=a_next)
            s, a = s_next, a_next
            total_r += r
    elif learn:
        done = False
        while not done:
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total_r += r
    else:
        # greedy evaluation
        done = False
        while not done:
            a = agent.greedy_action(s)
            s_next, r, done, _ = env.step(a)
            s = s_next
            total_r += r

    return total_r, env.detected, env.total_correct_bits, env.attacks_per_block


def train(agent_type, noise, detector=None,
          n_episodes=10000, min_blocks=10, max_blocks=10,
          rng_seed=42, verbose=False, log_interval=2000):
    """Train one agent against one detector and return a RunResult."""
    alpha, gamma = HYPERPARAMS[agent_type][noise]
    detector = detector if detector is not None else StaticThresholdDetector()
    env = BB84Environment(channel_error_rate=noise, detector=detector,
                          min_blocks=min_blocks, max_blocks=max_blocks,
                          rng_seed=rng_seed)
    agent = make_agent(agent_type, N_STATES, N_ACTIONS,
                       alpha=alpha, gamma=gamma,
                       epsilon_decay_episodes=n_episodes,
                       explore_attack_bias=0.20, rng_seed=rng_seed)
    result = RunResult(agent_type=agent_type, noise=noise,
                       detector_name=type(detector).__name__)

    for ep in range(n_episodes):
        r, det, correct, apb = run_episode(env, agent, agent_type, learn=True)
        agent.decay_epsilon()
        result.logs.append(EpisodeLog(ep, r, det, correct, apb))
        if verbose and (ep + 1) % log_interval == 0:
            w = result.logs[-log_interval:]
            print(f"  ep {ep+1:6d} | det={np.mean([e.detected for e in w]):.3f} "
                  f"| correct={np.mean([e.correct_bits for e in w]):.1f} "
                  f"| reward={np.mean([e.reward for e in w]):+.1f} "
                  f"| eps={agent.epsilon:.3f}")
    return result, agent


# ---------------------------------------------------------------------------
# Stage 1 reproduction: Q-Learning vs static, mu=1% -> Paper 1 ~0.28%
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, time
    noise = 0.01
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "qlearning"

    print("=" * 60)
    print(f"Stage 1 reproduction: {agent_type} vs static, mu={noise:.0%}")
    print(f"  Paper 1 target (mu=1%): Q 0.28% | SARSA 1.04% | DQ 3.52%")
    print("=" * 60)
    t0 = time.time()
    result, _ = train(agent_type, noise, n_episodes=10000,
                      rng_seed=42, verbose=True)
    print("-" * 60)
    print(f"  FINAL (last 500 eps): "
          f"detection={result.detection_rate()*100:.2f}% | "
          f"correct bits={result.mean_correct():.1f} | "
          f"reward={result.mean_reward():+.1f}")
    print(f"  ({(time.time()-t0)/60:.1f} min)")
    print("=" * 60)
