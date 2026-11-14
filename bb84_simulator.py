"""
BB84 Quantum Key Distribution Simulator
========================================
Simulates the physics of one BB84 session, qubit by qubit:
    Alice prepares -> Eve may intercept -> channel adds noise -> Bob measures
    -> sift -> compute QBER per block.

This is the trusted physics core. It contains NO reinforcement learning and
NO detection policy. Detection (the abort rule) is deliberately left OUT of
this file so that different detectors can be plugged in by the environment
layer. All this file knows how to do is run the quantum protocol and report,
per block, what the QBER was and what Eve learned.

Physics conventions
-------------------
    basis : 0 = rectilinear (+),  1 = diagonal (x)
    bit   : 0 or 1
    - Eve measuring in the WRONG basis yields a random bit (50/50). This is
      the origin of the 25% QBER under full intercept-resend.
    - The channel flips the transmitted bit with probability channel_error_rate.
    - Alice and Bob each choose a basis independently and uniformly, so they
      agree ~50% of the time (the sifting rate).

Variable-length episodes
------------------------
    n_blocks per episode is drawn uniformly from [min_blocks, max_blocks].
    When min_blocks == max_blocks (the default, both 10) the length is fixed,
    reproducing the Paper 1 setting exactly. When they differ, each episode
    has a random horizon -- the randomized-checkpoint condition.

The four sanity checks at the bottom are the ones Paper 1 verified. If any of
them drifts, the physics is wrong and nothing downstream can be trusted.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Per-block result: what the parties observe at one checkpoint
# ---------------------------------------------------------------------------

@dataclass
class BlockResult:
    """Everything known after one 100-qubit block is processed."""
    block_index: int
    qber: float               # observed error rate on sifted bits
    n_sifted: int             # how many qubits survived sifting
    eve_attacks: int          # how many qubits Eve intercepted this block
    eve_correct_bits: int     # of those, how many she learned correctly
                              # (right basis -> she knows Alice's bit)


# ---------------------------------------------------------------------------
# Full-episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Outcome of one full session. Detection is applied by the environment,
    not here -- this object just carries the per-block facts needed to decide
    detection and to score Eve."""
    n_blocks: int
    blocks: list = field(default_factory=list)   # list[BlockResult]

    @property
    def total_eve_correct_bits(self) -> int:
        return sum(b.eve_correct_bits for b in self.blocks)

    @property
    def total_eve_attacks(self) -> int:
        return sum(b.eve_attacks for b in self.blocks)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------

class BB84Simulator:
    """
    Runs BB84 sessions. Stateless between episodes except for its RNG.

    Parameters
    ----------
    channel_error_rate : float
        mu_ch, the background bit-flip probability. Default 0.03.
    checkpoint_interval : int
        qubits per block. Default 100.
    min_blocks, max_blocks : int
        episode horizon drawn uniformly from this inclusive range.
        Equal values (default 10, 10) = fixed horizon = Paper 1 setting.
    rng_seed : optional int
        reproducibility.
    """

    def __init__(
        self,
        channel_error_rate: float = 0.03,
        checkpoint_interval: int = 100,
        min_blocks: int = 10,
        max_blocks: int = 10,
        rng_seed: Optional[int] = None,
    ):
        assert min_blocks >= 1
        assert max_blocks >= min_blocks
        self.mu = channel_error_rate
        self.checkpoint_interval = checkpoint_interval
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.rng = np.random.default_rng(rng_seed)

    def sample_n_blocks(self) -> int:
        """Horizon for one episode. Fixed when min==max."""
        if self.min_blocks == self.max_blocks:
            return self.min_blocks
        return int(self.rng.integers(self.min_blocks, self.max_blocks + 1))

    # ------------------------------------------------------------------
    # Per-qubit processing (for training: the environment needs a reward
    # and a state after EVERY qubit, not just at block close, or credit
    # assignment breaks and the agent cannot learn attack timing).
    #
    # process_qubit returns a dict for one qubit:
    #   eve_attacked, eve_correct_basis, alice_bit, bob_bit, bases_match
    # The environment turns eve_correct_basis into the +1/-1 reward and
    # accumulates (alice_bit!=bob_bit on sifted qubits) into the block QBER.
    # ------------------------------------------------------------------
    def process_qubit(self, attack: int) -> dict:
        # RNG draw ORDER must match Paper 1 exactly for seed-for-seed
        # reproducibility: alice_bit, alice_basis, then eve (basis, and a
        # resend bit only on wrong basis), then the channel flip, then
        # bob_basis, then a bob bit only when the bases mismatch. Same
        # physics as any ordering, but the identical draw sequence makes
        # this simulator a faithful twin of the accepted Paper 1 code.
        alice_bit   = int(self.rng.integers(0, 2))      # _alice_prepare
        alice_basis = int(self.rng.integers(0, 2))

        transmitted = alice_bit
        eve_attacked = bool(attack)
        eve_correct_basis = False
        if eve_attacked:                                # _eve_intercept
            eve_basis = int(self.rng.integers(0, 2))
            eve_correct_basis = (eve_basis == alice_basis)
            if eve_correct_basis:
                eve_measured_bit = alice_bit
            else:
                eve_measured_bit = int(self.rng.integers(0, 2))
            transmitted = eve_measured_bit

        if self.rng.random() < self.mu:                 # _channel_noise
            transmitted = 1 - transmitted

        bob_basis = int(self.rng.integers(0, 2))        # _bob_measure
        bases_match = (bob_basis == alice_basis)
        if bases_match:
            bob_bit = transmitted
        else:
            bob_bit = int(self.rng.integers(0, 2))       # discarded in sifting

        return {
            "eve_attacked": eve_attacked,
            "eve_correct_basis": eve_correct_basis,
            "alice_bit": alice_bit,
            "bob_bit": bob_bit,
            "bases_match": bases_match,
        }

    # ------------------------------------------------------------------
    # One block of `checkpoint_interval` qubits.
    # attack_flags[i] == 1 means Eve intercepts qubit i of this block.
    # ------------------------------------------------------------------
    def process_block(self, block_index: int, attack_flags) -> BlockResult:
        n = self.checkpoint_interval
        assert len(attack_flags) == n

        # Alice's random bits and bases for the whole block (vectorized).
        alice_bit   = self.rng.integers(0, 2, size=n)
        alice_basis = self.rng.integers(0, 2, size=n)
        bob_basis   = self.rng.integers(0, 2, size=n)

        attacks = np.asarray(attack_flags, dtype=int)

        # --- Eve ---
        # She picks a basis per intercepted qubit. If it matches Alice's, she
        # learns the true bit and resends it faithfully. If not, she measures
        # a random bit and resends THAT (the disturbance).
        eve_basis = self.rng.integers(0, 2, size=n)
        eve_right_basis = (eve_basis == alice_basis) & (attacks == 1)
        eve_wrong_basis = (eve_basis != alice_basis) & (attacks == 1)

        # The bit travelling down the channel after Eve (if any):
        transmitted = alice_bit.copy()
        # Where Eve measured in the wrong basis, the resent bit is random:
        rand_bits = self.rng.integers(0, 2, size=n)
        transmitted[eve_wrong_basis] = rand_bits[eve_wrong_basis]
        # (Where Eve had the right basis, transmitted == alice_bit already.)

        # --- Channel noise: flip with prob mu ---
        flips = self.rng.random(size=n) < self.mu
        transmitted = np.where(flips, 1 - transmitted, transmitted)

        # --- Bob measures ---
        # When Bob's basis matches Alice's, he gets the transmitted bit.
        # When it doesn't, his result is random and the qubit is discarded
        # in sifting anyway, so it never affects QBER.
        bases_match = (bob_basis == alice_basis)
        bob_bit = np.where(bases_match, transmitted,
                           self.rng.integers(0, 2, size=n))

        # --- Sift: keep only matched-basis qubits, compute QBER on them ---
        sifted = bases_match
        n_sifted = int(sifted.sum())
        if n_sifted > 0:
            errors = int(np.sum(alice_bit[sifted] != bob_bit[sifted]))
            qber = errors / n_sifted
        else:
            qber = 0.0

        # --- Eve's information gain ---
        eve_attacks = int(attacks.sum())
        eve_correct_bits = int(eve_right_basis.sum())

        return BlockResult(
            block_index=block_index,
            qber=qber,
            n_sifted=n_sifted,
            eve_attacks=eve_attacks,
            eve_correct_bits=eve_correct_bits,
        )


# ---------------------------------------------------------------------------
# Sanity checks -- the Paper 1 physics gates.
# ---------------------------------------------------------------------------

def sanity_check():
    print("=" * 60)
    print("BB84 Simulator Sanity Checks")
    print("=" * 60)
    N = 5000
    interval = 100

    def mean_qber(mu, attack_rate, n_eps=N):
        sim = BB84Simulator(channel_error_rate=mu, rng_seed=0)
        qbers = []
        for _ in range(n_eps):
            flags = (sim.rng.random(interval) < attack_rate).astype(int)
            b = sim.process_block(0, flags)
            qbers.append(b.qber)
        return np.mean(qbers)

    q1 = mean_qber(0.0, 0.0)
    print(f"\nCheck 1 - No Eve, no noise")
    print(f"  Expected QBER: ~0.00%   Observed: {q1*100:.3f}%")

    q2 = mean_qber(0.0, 1.0)
    print(f"\nCheck 2 - Full attack, no noise")
    print(f"  Expected QBER: ~25.00%  Observed: {q2*100:.2f}%")

    q3 = mean_qber(0.05, 0.0)
    print(f"\nCheck 3 - No Eve, 5% noise")
    print(f"  Expected QBER: ~5.00%   Observed: {q3*100:.2f}%")

    # Check 4: full attack + 3% noise. Eve contributes 25%, noise 3%, minus
    # the overlap where both would flip (a double flip cancels):
    #   E[QBER] = 0.25 + mu - 2*0.25*mu   (independent error sources)
    mu = 0.03
    q4 = mean_qber(mu, 1.0)
    expected4 = 0.25 + mu - 2 * 0.25 * mu
    print(f"\nCheck 4 - Full attack, 3% noise")
    print(f"  Expected QBER: ~{expected4*100:.2f}%  Observed: {q4*100:.2f}%")

    # Check 5: partial-attack linearity, QBER ~ r*0.25 (no noise).
    print(f"\nCheck 5 - Partial-attack linearity (no noise)")
    for r in [0.25, 0.5, 0.75]:
        qr = mean_qber(0.0, r)
        print(f"  r={r:.2f}  Expected ~{r*0.25*100:.2f}%  Observed: {qr*100:.2f}%")

    # Check 6: variable horizon range.
    print(f"\nCheck 6 - Variable blocks (min=5, max=15)")
    sim = BB84Simulator(min_blocks=5, max_blocks=15, rng_seed=1)
    seen = [sim.sample_n_blocks() for _ in range(N)]
    print(f"  Mean ~10.0  Observed: {np.mean(seen):.2f} "
          f"| min={min(seen)} max={max(seen)}")

    print("\n" + "=" * 60)
    print("If Checks 1-5 match expectations, the physics is correct.")
    print("=" * 60)


if __name__ == "__main__":
    sanity_check()
