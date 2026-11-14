"""
BB84 Eavesdropping Environment
==============================
Connects the three pieces into a trainable loop:

    simulator (physics)  +  agent (Eve)  +  detector (abort rule)

This is the MDP Eve lives in. It presents a discrete STATE, accepts a per-qubit
ACTION (attack/pass), runs blocks through the simulator, asks the plugged-in
DETECTOR whether to abort, and returns REWARDS. The detector is passed in, so
the same environment serves the static threshold (Stage 1), the calibrated
variants (Stage 3), and the learning defender (Stage 4) without change.

State (discretized), matching Paper 1 v2
---------------------------------------
    QBER announced at last checkpoint : 16 bins over [0, 15%], step 1%
    Eve's attack count in this block  : 21 bins over [0, 100], step 5
    block index                       : 15 bins (supports horizons up to 15)
    => 16 * 21 * 15 = 5040 states

Reward (per Paper 1)
--------------------
    +1  Eve attacks and guesses the basis right (information gained)
    -1  Eve attacks and guesses wrong (disturbance, no gain)
     0  Eve passes
    +2  a checkpoint is survived (not aborted)
   -50  the detector aborts (Eve is "detected")

The -1 makes the immediate expected reward of attacking zero (she is right half
the time), so the agent must learn WHEN to attack, not just whether. The +2
rewards multi-block survival; the -50 dominates everything, so evasion is the
priority. The detector decides abort, so swapping detectors changes the game Eve
faces while leaving her state, actions and rewards identical.
"""

import numpy as np
from bb84_simulator import BB84Simulator
from detectors import Detector, StaticThresholdDetector

# --- State discretization constants ---
QBER_BINS   = 16     # [0,15%] step 1%
ATTACK_BINS = 21     # [0,100] step 5
BLOCK_BINS  = 15     # horizons up to 15
N_STATES    = QBER_BINS * ATTACK_BINS * BLOCK_BINS   # 5040
N_ACTIONS   = 2      # 0 = pass, 1 = attack

CHECKPOINT_INTERVAL = 100


def _bin_qber(qber: float) -> int:
    # 1%-wide bins, clamped to [0,15].
    return int(min(QBER_BINS - 1, max(0, int(qber * 100))))


def _bin_attacks(n: int) -> int:
    # 5-wide bins over [0,100].
    return int(min(ATTACK_BINS - 1, n // 5))


def _bin_block(idx: int) -> int:
    return int(min(BLOCK_BINS - 1, idx))


def encode_state(qber: float, attacks_this_block: int, block_index: int) -> int:
    """Flatten the three binned coordinates into a single state id."""
    qb = _bin_qber(qber)
    ab = _bin_attacks(attacks_this_block)
    bb = _bin_block(block_index)
    return (qb * ATTACK_BINS + ab) * BLOCK_BINS + bb


class BB84Environment:
    """
    Gym-style environment. One episode = one BB84 session.

    Parameters
    ----------
    channel_error_rate : mu_ch
    detector : a Detector instance (defaults to the static 0.11 threshold)
    min_blocks, max_blocks : horizon range (equal => fixed, Paper 1 setting)
    reward_* : the five reward constants above
    rng_seed : reproducibility
    """

    def __init__(self, channel_error_rate=0.03, detector: Detector = None,
                 min_blocks=10, max_blocks=10,
                 reward_correct=+1.0, reward_wrong=-1.0,
                 reward_survive=+2.0, reward_detected=-50.0, reward_pass=0.0,
                 rng_seed=None):
        self.sim = BB84Simulator(
            channel_error_rate=channel_error_rate,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            min_blocks=min_blocks, max_blocks=max_blocks,
            rng_seed=rng_seed,
        )
        self.detector = detector if detector is not None else StaticThresholdDetector()
        self.r_correct = reward_correct
        self.r_wrong = reward_wrong
        self.r_survive = reward_survive
        self.r_detected = reward_detected
        self.r_pass = reward_pass
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.n_blocks = self.sim.sample_n_blocks()
        self.block_index = 0
        self.qubit_in_block = 0
        self.last_qber = 0.0
        self.attacks_this_block = 0
        self.done = False
        # within-block sifting accumulators (for QBER at block close)
        self._sifted = 0
        self._sift_errors = 0
        # episode tallies
        self.total_correct_bits = 0
        self.total_attacks = 0
        self.detected = False
        self.attacks_per_block = []
        self.detector.reset()

    def reset(self):
        self._reset_episode_state()
        return encode_state(self.last_qber, 0, 0)

    def step(self, action: int):
        """
        Advance one qubit. Most steps just buffer the action and return the
        same state with reward 0. At a block boundary we run the block through
        the simulator, score it, and ask the detector whether to abort.

        Returns (next_state, reward, done, info).
        """
        assert not self.done, "step() called on a finished episode"

        # --- process THIS qubit through the physics ---
        q = self.sim.process_qubit(int(action))
        self.qubit_in_block += 1
        if action == 1:
            self.attacks_this_block += 1
            self.total_attacks += 1

        # --- per-qubit information reward (the key to credit assignment) ---
        if action == 1:
            if q["eve_correct_basis"]:
                reward = self.r_correct
                self.total_correct_bits += 1
            else:
                reward = self.r_wrong
        else:
            reward = self.r_pass

        # accumulate sifting for this block's QBER
        if q["bases_match"]:
            self._sifted += 1
            if q["alice_bit"] != q["bob_bit"]:
                self._sift_errors += 1

        info = {"completed_block": False}

        if self.qubit_in_block == CHECKPOINT_INTERVAL:
            qber = (self._sift_errors / self._sifted) if self._sifted > 0 else 0.0
            self.last_qber = qber
            self.attacks_per_block.append(self.attacks_this_block)

            # --- detector decides abort ---
            if self.detector.decide(qber, self.block_index):
                reward += self.r_detected
                self.detected = True
                self.done = True
                info.update(completed_block=True,
                            completed_block_qber=qber,
                            completed_block_attacks=self.attacks_this_block)
                return encode_state(qber, 0, self.block_index), reward, True, info

            # survived this checkpoint
            reward += self.r_survive
            info.update(completed_block=True,
                        completed_block_qber=qber,
                        completed_block_attacks=self.attacks_this_block)
            self.block_index += 1
            self.qubit_in_block = 0
            self.attacks_this_block = 0
            self._sifted = 0
            self._sift_errors = 0
            if self.block_index >= self.n_blocks:
                self.done = True   # reached the horizon undetected

        next_state = encode_state(self.last_qber,
                                  self.attacks_this_block,
                                  min(self.block_index, self.n_blocks - 1))
        return next_state, reward, self.done, info


# ---------------------------------------------------------------------------
# Verification -- mirrors Paper 1's environment checks
# ---------------------------------------------------------------------------

def verify_environment():
    print("=" * 60)
    print("Environment Verification")
    print("=" * 60)
    PASS, FAIL = "PASS", "FAIL"

    def check(cond, label, obs=None, exp=None):
        tag = PASS if cond else FAIL
        line = f"  [{tag}] {label}"
        if obs is not None:
            line += f"  (observed={obs}, expected={exp})"
        print(line)

    print("\n[V1] State space size")
    check(N_STATES == 5040, "N_STATES = 5040", N_STATES, 5040)

    print("\n[V2] Fixed mode: exactly 1000 steps for 10 blocks")
    env = BB84Environment(channel_error_rate=0.0, min_blocks=10, max_blocks=10,
                          rng_seed=0)
    env.reset()
    steps = 0
    while not env.done:
        env.step(0); steps += 1
    check(steps == 1000, "1000 steps in fixed mode", steps, 1000)

    print("\n[V3] Random mode: horizon in [500, 1500]")
    env = BB84Environment(channel_error_rate=0.0, min_blocks=5, max_blocks=15,
                          rng_seed=1)
    lengths = []
    for _ in range(300):
        env.reset(); s = 0
        while not env.done:
            env.step(0); s += 1
        lengths.append(s)
    check(min(lengths) == 500 and max(lengths) == 1500,
          "length spans [500,1500]", (min(lengths), max(lengths)), (500, 1500))
    check(abs(np.mean(lengths) - 1000) < 60,
          "mean length ~1000", round(np.mean(lengths), 1), 1000)

    print("\n[V4] Full attack always detected (static detector)")
    env = BB84Environment(channel_error_rate=0.0, min_blocks=10, max_blocks=10,
                          rng_seed=2)
    detected = 0
    for _ in range(200):
        env.reset()
        while not env.done:
            _, _, done, _ = env.step(1)
        if env.detected:
            detected += 1
    check(detected == 200, "full attack detected every time", detected, 200)

    print("\n[V5] No attack, no noise -> never detected, full horizon")
    env = BB84Environment(channel_error_rate=0.0, min_blocks=10, max_blocks=10,
                          rng_seed=3)
    env.reset()
    while not env.done:
        env.step(0)
    check(not env.detected and env.block_index == 10,
          "survives all 10 blocks", (env.detected, env.block_index),
          (False, 10))

    print("\n" + "=" * 60)
    print("Environment verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_environment()
