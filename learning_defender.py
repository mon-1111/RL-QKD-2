"""
Learning Defender (Stage 4)
===========================
A tabular RL defender that plays the detector role: instead of a fixed abort
rule, it LEARNS which abort threshold to apply at each block, from the QBER it
observes. This is the new contribution of Paper 2.

It implements the same decide()/reset() interface the environment already uses,
so it drops into the existing game unchanged. Internally it wraps a tabular
agent (reusing agents.py) and remembers its (state, action) choices during an
episode; after the episode it receives a reward and updates its Q-table.

Transparency is preserved: the defender is a small tabular agent (160 states x
a handful of threshold actions), so its learned policy can be printed as a
table and inspected -- the same virtue as the transparent fixed defenders.

State  : (QBER bin x block index) = 16 x 10 = 160. Does NOT include Eve's
         attack count -- the information asymmetry under test.
Action : index into THRESHOLDS; abort iff block QBER >= chosen threshold.
Reward : assigned at episode end (see train loop), penalizing BOTH missed
         attacks and false alarms, constrained to the matched-FPR budget.
"""

import numpy as np
from agents import make_agent

# Discrete abort thresholds the defender can choose among.
THRESHOLDS = [0.08, 0.09, 0.10, 0.11, 0.12]

QBER_BINS_D  = 16    # [0,15%] step 1%, same binning as the attacker env
BLOCK_BINS_D = 10    # fixed-horizon 10 blocks
N_STATES_D   = QBER_BINS_D * BLOCK_BINS_D     # 160
N_ACTIONS_D  = len(THRESHOLDS)


def _bin_qber(qber):
    return int(min(QBER_BINS_D - 1, max(0, int(qber * 100))))


def _bin_block(idx):
    return int(min(BLOCK_BINS_D - 1, idx))


def encode_state_d(qber, block_index):
    return _bin_qber(qber) * BLOCK_BINS_D + _bin_block(block_index)


class LearningDefender:
    """
    Detector-interface wrapper around a tabular agent.

    Usage in the game:
        d = LearningDefender(agent_type="qlearning", rng_seed=0)
        # ... inside an episode the environment calls d.decide(qber, blk) ...
        # after the episode:
        d.learn_from_episode(reward)      # single terminal reward
        d.decay_epsilon()                 # once per episode during training

    During training the defender explores; call d.set_eval() to freeze it
    (greedy, no learning) when it is the frozen opponent or under exploitability.
    """

    def __init__(self, agent_type="qlearning", alpha=0.5, gamma=0.5,
                 epsilon_decay_episodes=10000, rng_seed=None):
        self.agent = make_agent(agent_type, N_STATES_D, N_ACTIONS_D,
                                alpha=alpha, gamma=gamma,
                                epsilon_decay_episodes=epsilon_decay_episodes,
                                rng_seed=rng_seed)
        self.learning = True
        self.last_reason = ""
        self._transitions = []    # (state, action) chosen this episode

    # --- detector interface ---------------------------------------------
    def reset(self):
        self.last_reason = ""
        self._transitions = []

    def decide(self, block_qber, block_index):
        s = encode_state_d(block_qber, block_index)
        if self.learning:
            a = self.agent.select_action(s)
        else:
            a = self.agent.greedy_action(s)
        self._transitions.append((s, a))
        tau = THRESHOLDS[a]
        abort = block_qber >= tau
        self.last_reason = (f"block {block_index}: qber={block_qber:.4f} "
                            f"{'>=' if abort else '<'} chosen tau={tau:.3f}")
        return abort

    def describe(self):
        return (f"Learned threshold policy over {N_STATES_D} states "
                f"(QBER x block), actions = {THRESHOLDS}.")

    # --- learning -------------------------------------------------------
    def learn_from_episode(self, reward):
        """
        Assign a single terminal reward to the episode's chosen (s,a) pairs.
        We treat each block-decision as receiving the shared episode reward
        (a simple, transparent credit scheme suitable for a short horizon and
        a small state space). Updated as terminal transitions.
        """
        if not self.learning or not self._transitions:
            return
        for (s, a) in self._transitions:
            # terminal update: no bootstrap (done=True)
            self.agent.update(s, a, reward, s, done=True)

    def decay_epsilon(self):
        self.agent.decay_epsilon()

    def set_eval(self):
        """Freeze: greedy actions, no learning (frozen opponent / exploitability)."""
        self.learning = False
        self.agent.epsilon = 0.0

    def set_train(self):
        self.learning = True


# ---------------------------------------------------------------------------
# Verification (tiny scale)
# ---------------------------------------------------------------------------

def verify_learning_defender():
    print("=" * 60)
    print("Learning Defender Verification")
    print("=" * 60)
    PASS, FAIL = "PASS", "FAIL"

    def check(cond, label, obs=None, exp=None):
        tag = PASS if cond else FAIL
        line = f"  [{tag}] {label}"
        if obs is not None:
            line += f"  (observed={obs}, expected={exp})"
        print(line)

    print("\n[V1] State/action space")
    check(N_STATES_D == 160, "160 states", N_STATES_D, 160)
    check(N_ACTIONS_D == 5, "5 threshold actions", N_ACTIONS_D, 5)

    print("\n[V2] decide() implements the detector interface")
    d = LearningDefender(rng_seed=0)
    d.reset()
    r = d.decide(0.05, 0)
    check(isinstance(r, bool), "decide returns bool", type(r).__name__, "bool")
    check(len(d._transitions) == 1, "records transition", len(d._transitions), 1)
    check(len(d.last_reason) > 0, "decision explainable")

    print("\n[V3] A high-QBER block aborts under the tightest threshold")
    # If it happens to pick tau=0.08 and qber=0.12, it must abort.
    d.reset(); d.agent.Q[encode_state_d(0.12, 0)] = [1,0,0,0,0]  # prefer tau=0.08
    d.set_eval()
    check(d.decide(0.12, 0) is True, "qber 0.12 >= tau 0.08 -> abort",
          d.decide(0.12,0), True)

    print("\n[V4] Learning changes the policy")
    d = LearningDefender(rng_seed=1, epsilon_decay_episodes=100)
    d.set_train()
    before = d.agent.Q.copy()
    for _ in range(50):
        d.reset()
        d.decide(0.10, 0); d.decide(0.04, 1)
        d.learn_from_episode(+10.0)     # reward these choices
        d.decay_epsilon()
    changed = not np.allclose(before, d.agent.Q)
    check(changed, "Q-table updates from episode rewards", changed, True)

    print("\n[V5] set_eval freezes exploration and learning")
    d.set_eval()
    check(d.agent.epsilon == 0.0 and d.learning is False,
          "eval mode: epsilon=0, learning off",
          (d.agent.epsilon, d.learning), (0.0, False))
    q_before = d.agent.Q.copy()
    d.reset(); d.decide(0.10, 0); d.learn_from_episode(5.0)
    check(np.allclose(q_before, d.agent.Q),
          "no Q change while frozen", True, True)

    print("\n" + "=" * 60)
    print("Learning defender verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_learning_defender()
