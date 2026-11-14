"""
Distillable secure key fraction
===============================
Turns a QBER into the fraction of key that survives privacy amplification.
This is Paper 1's real threat metric: raw intercepted bits overstate a loud
attacker and understate a quiet one, because Alice and Bob amplify away more
key when they see more error. The key fraction corrects for that.

For BB84 against intercept-resend, the asymptotic Shor-Preskill bound is

    r(e) = 1 - 2 h(e),       h(e) = -e log2(e) - (1-e) log2(1-e)

where e is the QBER. The factor 2 is the two sacrifices Alice and Bob make:
one h(e) for error correction, one h(e) for privacy amplification against Eve.
Below e = 0 the fraction is 1 (perfect key); above e ~ 0.11 the bound goes to
0 (no secure key can be distilled) -- which is exactly why 11% is the abort
threshold. We clamp the fraction to [0, 1].

The headline consequence (Paper 1):
    fixed-rate Eve  holds QBER ~10.4%  -> r ~ 0.04  (key almost all amplified away)
    RL Eve          holds QBER ~ 1%    -> r ~ 0.84  (key survives nearly intact)
So the RL agent, though it collects FEWER raw bits, compromises far MORE
actual key, because it stays quiet enough to avoid heavy amplification.
"""

import numpy as np


def binary_entropy(e: float) -> float:
    """
    Binary Shannon entropy h(e) in bits. h(0)=h(1)=0, h(0.5)=1.
    Guards the log at the endpoints so h(0) and h(1) return 0 cleanly.
    """
    e = float(e)
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return -e * np.log2(e) - (1.0 - e) * np.log2(1.0 - e)


def key_fraction(qber: float) -> float:
    """
    Distillable secure key fraction r(e) = 1 - 2 h(e), clamped to [0, 1].
    r = 1 at e = 0 (perfect key); r = 0 at e ~ 0.11 (no secure key).
    """
    r = 1.0 - 2.0 * binary_entropy(qber)
    return float(min(1.0, max(0.0, r)))


def key_fraction_array(qbers) -> np.ndarray:
    """Vectorized key_fraction over an array of QBER values."""
    return np.array([key_fraction(q) for q in np.asarray(qbers, dtype=float)])


# ---------------------------------------------------------------------------
# Verification against Paper 1's reported numbers
# ---------------------------------------------------------------------------

def verify_keyfraction():
    print("=" * 60)
    print("Key-Fraction Verification")
    print("=" * 60)
    PASS, FAIL = "PASS", "FAIL"

    def check(cond, label, obs=None, exp=None):
        tag = PASS if cond else FAIL
        line = f"  [{tag}] {label}"
        if obs is not None:
            line += f"  (observed={obs}, expected={exp})"
        print(line)

    print("\n[V1] Binary entropy endpoints and midpoint")
    check(binary_entropy(0.0) == 0.0, "h(0) = 0", binary_entropy(0.0), 0.0)
    check(binary_entropy(1.0) == 0.0, "h(1) = 0", binary_entropy(1.0), 0.0)
    check(abs(binary_entropy(0.5) - 1.0) < 1e-9, "h(0.5) = 1",
          round(binary_entropy(0.5), 6), 1.0)

    print("\n[V2] Key fraction limits")
    check(key_fraction(0.0) == 1.0, "r(0) = 1 (perfect key)",
          key_fraction(0.0), 1.0)
    check(key_fraction(0.11) < 0.01, "r(0.11) ~ 0 (threshold)",
          round(key_fraction(0.11), 4), "~0")
    check(key_fraction(0.25) == 0.0, "r(0.25) = 0 (clamped)",
          key_fraction(0.25), 0.0)

    print("\n[V3] Paper 1 headline numbers")
    # RL Eve keeps QBER ~ channel baseline 1%
    r_rl = key_fraction(0.01)
    check(abs(r_rl - 0.84) < 0.02, "RL Eve (QBER~1%): r ~ 0.84",
          round(r_rl, 3), 0.84)
    # Fixed-rate Eve holds QBER ~ 10.4% (just under threshold)
    r_fixed = key_fraction(0.104)
    check(abs(r_fixed - 0.04) < 0.02, "Fixed-rate Eve (QBER~10.4%): r ~ 0.04",
          round(r_fixed, 3), 0.04)

    print(f"\n  -> RL agent compromises ~{r_rl*100:.0f}% of key material;")
    print(f"     fixed-rate agent only ~{r_fixed*100:.0f}%, despite the fixed")
    print(f"     agent collecting far MORE raw bits. This is the real threat.")

    print("\n[V4] Monotonic decrease in QBER")
    qs = [0.0, 0.01, 0.03, 0.05, 0.08, 0.11]
    rs = key_fraction_array(qs)
    mono = all(rs[i] >= rs[i+1] for i in range(len(rs)-1))
    check(mono, "r decreases as QBER rises",
          [round(x, 3) for x in rs], "monotone down")

    print("\n" + "=" * 60)
    print("Key-fraction verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    verify_keyfraction()
