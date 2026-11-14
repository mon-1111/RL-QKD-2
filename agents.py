"""
Tabular RL Agents for BB84
==========================
Three agents that share everything except their Q-value update rule:

    QLearningAgent        off-policy, uses max over next actions (aggressive)
    SARSAAgent            on-policy, uses the action actually taken (cautious)
    DoubleQLearningAgent  two tables cancel overestimation (most conservative)

An agent is just a table Q[state, action]. It acts by epsilon-greedy selection
(mostly greedy, sometimes random), and learns by nudging Q toward observed
return. The three differ ONLY in what they nudge toward -- that single line is
the whole scientific content here, and it is what produces the overestimation
hierarchy (Q-Learning most aggressive, Double-Q most conservative) that Paper 1
reports.

Biased exploration
------------------
During random exploration the agent attacks with probability
`explore_attack_bias` (default 0.20), NOT 0.5. Unbiased 50% exploration drives
QBER to ~12.5%, over the 11% threshold, so the agent is detected in nearly
every early episode and never survives long enough to learn restraint. Biasing
toward 20% keeps early QBER ~5%, giving the agent room to learn. This is the
same fix Paper 1 used.

These agents are generic: the same class is used for Eve (attack/pass) and,
later, for the learning defender (whose actions are threshold choices). The
agent code does not know or care what the actions mean -- that is the
environment's job.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Common infrastructure: the Q-table, epsilon-greedy selection with biased
    exploration, and linear epsilon decay. Subclasses implement update().

    Parameters
    ----------
    n_states, n_actions : int
        shape of the Q-table.
    alpha : float
        learning rate (how big each nudge is).
    gamma : float
        discount (how much future reward counts vs immediate).
    epsilon_start, epsilon_end : float
        exploration rate, decayed linearly from start to end.
    epsilon_decay_episodes : int
        number of episodes over which epsilon decays.
    explore_attack_bias : float
        during exploration, probability of choosing action 1 (attack).
        0.20 by default; set 0.5 for unbiased. (Only meaningful for the
        2-action attacker; harmless for other action counts, where it just
        biases toward action 1.)
    rng_seed : optional int
    """

    def __init__(self, n_states, n_actions,
                 alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_episodes=10000,
                 explore_attack_bias=0.20,
                 rng_seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes
        self.explore_attack_bias = explore_attack_bias
        self.rng = np.random.default_rng(rng_seed)
        self.Q = np.zeros((n_states, n_actions))
        self.episode_count = 0

    def select_action(self, state):
        """Epsilon-greedy with biased exploration."""
        if self.rng.random() < self.epsilon:
            # Explore. For the 2-action case this biases toward pass (0).
            if self.n_actions == 2:
                return 1 if self.rng.random() < self.explore_attack_bias else 0
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state):
        """No exploration -- used at evaluation time."""
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.episode_count += 1

    def reset(self):
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.epsilon = self.epsilon_start
        self.episode_count = 0


# ---------------------------------------------------------------------------
# Q-Learning: off-policy, nudge toward the BEST next action
# ---------------------------------------------------------------------------

class QLearningAgent(BaseAgent):
    def update(self, s, a, r, s_next, done, **kwargs):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# ---------------------------------------------------------------------------
# SARSA: on-policy, nudge toward the action ACTUALLY TAKEN next
# ---------------------------------------------------------------------------

class SARSAAgent(BaseAgent):
    def update(self, s, a, r, s_next, done, next_action=None, **kwargs):
        assert done or next_action is not None, "SARSA needs next_action"
        target = r if done else r + self.gamma * self.Q[s_next, next_action]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# ---------------------------------------------------------------------------
# Double Q-Learning: two tables, one picks and the other evaluates
# ---------------------------------------------------------------------------

class DoubleQLearningAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q2 = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            if self.n_actions == 2:
                return 1 if self.rng.random() < self.explore_attack_bias else 0
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state] + self.Q2[state]))

    def greedy_action(self, state):
        return int(np.argmax(self.Q[state] + self.Q2[state]))

    def update(self, s, a, r, s_next, done, **kwargs):
        if done:
            self.Q[s, a]  += self.alpha * (r - self.Q[s, a])
            self.Q2[s, a] += self.alpha * (r - self.Q2[s, a])
            return
        if self.rng.random() < 0.5:
            # Q picks the action, Q2 evaluates it.
            a_star = int(np.argmax(self.Q[s_next]))
            target = r + self.gamma * self.Q2[s_next, a_star]
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        else:
            a_star = int(np.argmax(self.Q2[s_next]))
            target = r + self.gamma * self.Q[s_next, a_star]
            self.Q2[s, a] += self.alpha * (target - self.Q2[s, a])

    def reset(self):
        super().reset()
        self.Q2 = np.zeros((self.n_states, self.n_actions))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agent(agent_type, n_states, n_actions, **kwargs):
    agents = {
        "qlearning": QLearningAgent,
        "sarsa": SARSAAgent,
        "doubleq": DoubleQLearningAgent,
    }
    t = agent_type.lower()
    if t not in agents:
        raise ValueError(f"Unknown agent '{agent_type}'. Choose {list(agents)}.")
    return agents[t](n_states, n_actions, **kwargs)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_agents():
    print("=" * 60)
    print("Agent Verification")
    print("=" * 60)
    PASS, FAIL = "PASS", "FAIL"

    def check(cond, label, obs=None, exp=None):
        tag = PASS if cond else FAIL
        line = f"  [{tag}] {label}"
        if obs is not None:
            line += f"  (observed={obs}, expected={exp})"
        print(line)

    N_STATES, N_ACTIONS = 160, 2

    for name in ["qlearning", "sarsa", "doubleq"]:
        print(f"\n[{name.upper()}]")
        ag = make_agent(name, N_STATES, N_ACTIONS,
                        alpha=0.1, gamma=0.95,
                        epsilon_decay_episodes=1000, rng_seed=42)
        check(ag.epsilon == 1.0, "initial epsilon = 1.0", ag.epsilon, 1.0)
        check(np.all(ag.Q == 0), "Q starts at zeros")
        for _ in range(1000):
            ag.decay_epsilon()
        check(abs(ag.epsilon - 0.01) < 1e-3, "epsilon decays to 0.01",
              round(ag.epsilon, 4), 0.01)
        # Greedy picks the higher Q-value.
        ag.reset(); ag.epsilon = 0.0; ag.Q[0, 1] = 1.0
        check(ag.select_action(0) == 1, "greedy picks best action",
              ag.select_action(0), 1)
        # Terminal update ignores the future.
        ag.reset(); ag.Q[0, 1] = 99.0
        if name == "sarsa":
            ag.update(0, 0, 5.0, 0, done=True, next_action=1)
        else:
            ag.update(0, 0, 5.0, 0, done=True)
        check(abs(ag.Q[0, 0] - 0.5) < 1e-6, "terminal update = alpha*reward",
              round(ag.Q[0, 0], 4), 0.5)

    # Biased exploration attacks ~20%.
    print("\n[BIASED EXPLORATION]")
    ag = make_agent("qlearning", N_STATES, N_ACTIONS,
                    explore_attack_bias=0.20, rng_seed=0)
    ag.epsilon = 1.0
    rate = np.mean([ag.select_action(0) for _ in range(5000)])
    check(abs(rate - 0.20) < 0.03, "exploration attacks ~20%",
          round(rate, 3), 0.20)

    # Q-Learning's optimism exceeds SARSA when the explored action is bad.
    print("\n[OVERESTIMATION: Q-Learning > SARSA]")
    q = QLearningAgent(N_STATES, N_ACTIONS, alpha=1.0, gamma=1.0)
    s = SARSAAgent(N_STATES, N_ACTIONS, alpha=1.0, gamma=1.0)
    for ag in (q, s):
        ag.Q[5, 0] = 10.0; ag.Q[5, 1] = 1.0   # action 0 is best at state 5
    q.update(0, 0, 0.0, 5, done=False)                 # uses max -> 10
    s.update(0, 0, 0.0, 5, done=False, next_action=1)  # uses taken(1) -> 1
    check(q.Q[0, 0] > s.Q[0, 0],
          "Q-Learning target optimistic vs SARSA",
          f"{q.Q[0,0]:.1f} > {s.Q[0,0]:.1f}", "True")

    print("\n" + "=" * 60)
    print("Agent verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_agents()
