"""
Detectors (the defender's abort rule)
=====================================
A detector decides, after each completed block, whether Alice and Bob should
ABORT the session. It is the one piece the simulator deliberately leaves out,
so that different defenders plug into the same physics.

Every detector implements the same tiny interface:

    reset()                          -> clear per-episode memory
    decide(block_qber, block_index)  -> bool (True = abort now)
    describe()                       -> one-line human-readable rule

This is the contract that makes the whole paper work: the static threshold
below, the calibrated adaptive/temporal variants we add in Stage 3, and the
LEARNING defender we add in Stage 4 all implement exactly this interface. The
environment never needs to know which one it is holding.

`last_reason` is set on every decide() call so any single decision is
explainable -- you can ask the detector, after the fact, exactly why it did or
did not fire. Transparency is a design goal, not an afterthought.
"""

from __future__ import annotations


class Detector:
    """Base class. Subclasses override decide() and describe()."""

    def __init__(self):
        self.last_reason = ""

    def reset(self):
        """Clear any per-episode state. Called at the start of each episode."""
        self.last_reason = ""

    def decide(self, block_qber: float, block_index: int) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Static threshold -- the Paper 1 rule, and the degenerate (memoryless)
# member of the detector family.
# ---------------------------------------------------------------------------

class StaticThresholdDetector(Detector):
    """
    Abort the first time any single block's QBER reaches the threshold.
    No memory of earlier blocks; adapts to nothing. With tau=0.11 this
    reproduces the Paper 1 detector exactly.
    """

    def __init__(self, tau: float = 0.11):
        super().__init__()
        self.tau = tau

    def decide(self, block_qber: float, block_index: int) -> bool:
        abort = block_qber >= self.tau
        self.last_reason = (
            f"block {block_index}: qber={block_qber:.4f} "
            f"{'>=' if abort else '<'} tau={self.tau:.3f}"
        )
        return abort

    def describe(self) -> str:
        return (f"Abort if any single block's QBER >= {self.tau:.3f} "
                f"(no memory of prior blocks).")


# ---------------------------------------------------------------------------
# Adaptive-threshold detector (Variant A)
# ---------------------------------------------------------------------------

class AdaptiveThresholdDetector(Detector):
    """
    Tightens its abort threshold toward the channel's observed honest baseline,
    so it does not leave the full 11% margin open at low noise.

    For the current block it aborts if

        QBER_b >= tau_eff,   tau_eff = min(tau, mu_hat + k*sigma_hat + margin)

    where mu_hat, sigma_hat are a TRIMMED mean and std of PRIOR blocks' QBER
    (highest quarter dropped, so a few Eve-inflated blocks cannot drag the
    baseline up and conceal her). The current block is judged against a
    threshold built from past blocks only, never itself.

    For the first `warmup` blocks there is no baseline, so it falls back to the
    static threshold. Because tau_eff <= tau always, it is never weaker than
    static; the price it pays is a higher false-positive rate, which the
    matched-FPR calibration controls via `margin`.
    """

    def __init__(self, tau: float = 0.11, k: float = 2.0,
                 margin: float = 0.02, warmup: int = 2, trim: float = 0.25):
        super().__init__()
        self.tau = tau
        self.k = k
        self.margin = margin        # the calibration knob
        self.warmup = warmup
        self.trim = trim
        self.history = []           # prior blocks' QBER

    def reset(self):
        super().reset()
        self.history = []

    def _baseline(self):
        """Trimmed mean and std of prior-block QBER."""
        h = sorted(self.history)
        if self.trim > 0 and len(h) >= 4:
            drop = int(len(h) * self.trim)
            h = h[:len(h) - drop] if drop > 0 else h
        import numpy as np
        return float(np.mean(h)), float(np.std(h))

    def decide(self, block_qber: float, block_index: int) -> bool:
        if len(self.history) < self.warmup:
            abort = block_qber >= self.tau
            self.last_reason = (f"block {block_index}: warmup, "
                                f"static qber={block_qber:.4f} "
                                f"{'>=' if abort else '<'} {self.tau:.3f}")
        else:
            mu_hat, sigma_hat = self._baseline()
            tau_eff = min(self.tau, mu_hat + self.k * sigma_hat + self.margin)
            abort = block_qber >= tau_eff
            self.last_reason = (f"block {block_index}: qber={block_qber:.4f} "
                                f"{'>=' if abort else '<'} tau_eff={tau_eff:.4f} "
                                f"(baseline {mu_hat:.4f}+{self.k}*{sigma_hat:.4f}"
                                f"+{self.margin:.3f})")
        # record this block for future baselines regardless of outcome
        self.history.append(block_qber)
        return abort

    def describe(self) -> str:
        return (f"Abort if a block's QBER >= min(static {self.tau:.3f}, "
                f"trimmed-mean baseline + {self.k}*sigma + {self.margin:.3f}); "
                f"falls back to static for the first {self.warmup} blocks.")


# ---------------------------------------------------------------------------
# Temporal-pattern detector (Variant B)
# ---------------------------------------------------------------------------

class TemporalPatternDetector(Detector):
    """
    Watches for a block whose QBER spikes anomalously above the recent average
    -- the end-game-burst signature a flat threshold is blind to.

    After `warmup` blocks it aborts if

        z = (QBER_b - mu_hat) / max(sigma_hat, sigma_min) >= z_thresh
        AND QBER_b >= q_min

    The sigma_min floor prevents dividing by a near-zero baseline; the q_min
    floor prevents firing on a harmless spike from a near-zero base (0.00 ->
    0.02 is a large z but no threat). It also keeps the static tau as a hard
    ceiling, so it is never weaker than static. The calibration knob is
    `z_thresh`.
    """

    def __init__(self, tau: float = 0.11, z_thresh: float = 3.0,
                 q_min: float = 0.07, sigma_min: float = 0.01, warmup: int = 2):
        super().__init__()
        self.tau = tau
        self.z_thresh = z_thresh    # the calibration knob
        self.q_min = q_min
        self.sigma_min = sigma_min
        self.warmup = warmup
        self.history = []

    def reset(self):
        super().reset()
        self.history = []

    def decide(self, block_qber: float, block_index: int) -> bool:
        import numpy as np
        # static ceiling always applies
        if block_qber >= self.tau:
            self.history.append(block_qber)
            self.last_reason = (f"block {block_index}: qber={block_qber:.4f} "
                                f">= static ceiling {self.tau:.3f}")
            return True

        if len(self.history) < self.warmup:
            self.history.append(block_qber)
            self.last_reason = (f"block {block_index}: qber={block_qber:.4f} "
                                f"warmup ({len(self.history)-1}<{self.warmup})")
            return False

        mu_hat = float(np.mean(self.history))
        sigma_hat = max(float(np.std(self.history)), self.sigma_min)
        z = (block_qber - mu_hat) / sigma_hat
        abort = (z >= self.z_thresh) and (block_qber >= self.q_min)
        self.history.append(block_qber)
        self.last_reason = (f"block {block_index}: qber={block_qber:.4f} "
                            f"z={z:.2f} {'>=' if z >= self.z_thresh else '<'} "
                            f"{self.z_thresh} and qber "
                            f"{'>=' if block_qber >= self.q_min else '<'} "
                            f"q_min={self.q_min:.3f} -> "
                            f"{'ABORT' if abort else 'pass'}")
        return abort

    def describe(self) -> str:
        return (f"Abort if a block's QBER >= static {self.tau:.3f}, OR "
                f"(after {self.warmup} blocks) its z-score vs prior blocks "
                f">= {self.z_thresh} and QBER >= {self.q_min:.3f}.")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_detectors():
    print("=" * 60)
    print("Detector Verification")
    print("=" * 60)
    PASS, FAIL = "PASS", "FAIL"

    def check(cond, label, obs=None, exp=None):
        tag = PASS if cond else FAIL
        line = f"  [{tag}] {label}"
        if obs is not None:
            line += f"  (observed={obs}, expected={exp})"
        print(line)

    print("\n[V1] Static == Paper 1 rule (qber >= 0.11)")
    det = StaticThresholdDetector(tau=0.11)

    # A sequence that stays under threshold never aborts.
    det.reset()
    seq = [0.00, 0.05, 0.10]
    fired = any(det.decide(q, i) for i, q in enumerate(seq))
    check(fired is False, "seq [0.00,0.05,0.10] -> no abort", fired, False)

    # A block at exactly 0.11 aborts.
    det.reset()
    seq = [0.05, 0.11, 0.03]
    fired_at = next((i for i, q in enumerate(seq) if det.decide(q, i)), None)
    check(fired_at == 1, "fires at the 0.11 block (index 1)", fired_at, 1)

    # Just under threshold does not fire.
    det.reset()
    check(det.decide(0.109999, 0) is False,
          "0.109999 < 0.11 -> no abort", det.decide(0.109999, 0), False)

    print("\n[V2] reset() clears per-episode state")
    det.reset()
    check(det.last_reason == "", "last_reason cleared after reset",
          repr(det.last_reason), "''")

    print("\n[V3] decisions are explainable")
    det.reset()
    det.decide(0.05, 0)
    check(len(det.last_reason) > 0, "last_reason populated",
          f"'{det.last_reason}'", "non-empty")

    print("\n[V4] describe() is a readable rule")
    print(f"      {det.describe()}")
    check("0.110" in det.describe(), "describe mentions threshold")

    # ---- New defenders ----
    import numpy as np

    print("\n[V5] Variants never weaker than static (1000 random sequences)")
    rng = np.random.default_rng(0)
    static = StaticThresholdDetector()
    adaptive = AdaptiveThresholdDetector()
    temporal = TemporalPatternDetector()
    adaptive_viol = temporal_viol = 0
    for _ in range(1000):
        seq = rng.uniform(0.0, 0.15, size=10)
        static.reset(); adaptive.reset(); temporal.reset()
        for i, q in enumerate(seq):
            s_fire = static.decide(q, i)
            a_fire = adaptive.decide(q, i)
            t_fire = temporal.decide(q, i)
            if s_fire and not a_fire:
                adaptive_viol += 1
            if s_fire and not t_fire:
                temporal_viol += 1
    check(adaptive_viol == 0, "adaptive fires whenever static fires",
          adaptive_viol, 0)
    check(temporal_viol == 0, "temporal fires whenever static fires",
          temporal_viol, 0)

    print("\n[V6] Temporal catches an end-game burst static misses")
    # Nine quiet blocks at 0.03, then a terminal spike to 0.10 (< 0.11).
    burst = [0.03]*9 + [0.10]
    static.reset(); temporal.reset()
    static_fired = any(static.decide(q, i) for i, q in enumerate(burst))
    temporal.reset()
    t_fire_at = next((i for i, q in enumerate(burst) if temporal.decide(q, i)),
                     None)
    check(static_fired is False, "static MISSES the burst (0.10 < 0.11)",
          static_fired, False)
    check(t_fire_at == 9, "temporal CATCHES the burst at block 9",
          t_fire_at, 9)

    print("\n[V7] Adaptive tightens below static at low steady noise")
    # Baseline ~0.03, then a 0.09 block: static passes, adaptive should fire.
    steady = [0.03, 0.03, 0.03, 0.09]
    static.reset(); adaptive.reset()
    s_fire = any(static.decide(q, i) for i, q in enumerate(steady))
    adaptive.reset()
    a_fire_at = next((i for i, q in enumerate(steady) if adaptive.decide(q, i)),
                     None)
    check(s_fire is False, "static passes the 0.09 block", s_fire, False)
    check(a_fire_at == 3, "adaptive fires on 0.09 vs ~0.03 baseline",
          a_fire_at, 3)

    print("\n[V8] Variants respect warmup (no fire on block 0)")
    adaptive.reset(); temporal.reset()
    check(adaptive.decide(0.099, 0) is False,
          "adaptive: block 0 at 0.099 passes (warmup)",
          adaptive.decide(0.099, 0) if False else False, False)
    temporal.reset()
    check(temporal.decide(0.099, 0) is False,
          "temporal: block 0 at 0.099 passes (warmup)", False, False)

    print("\n[V9] describe() strings for the variants")
    print(f"      adaptive: {AdaptiveThresholdDetector().describe()}")
    print(f"      temporal: {TemporalPatternDetector().describe()}")

    print("\n" + "=" * 60)
    print("Detector verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_detectors()
