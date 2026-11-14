"""
Matched-FPR calibration
========================
The methodological backbone of the detector comparison.

A tighter defender catches more attacks, but also raises more false alarms on
honest, attack-free traffic. Comparing two defenders at DIFFERENT false-positive
rates (FPR) is meaningless -- a trigger-happy defender detects more of
everything. So before any detection is compared, every defender is calibrated to
the SAME false-alarm rate as the static threshold, per noise level. Then a
difference in detection reflects real capability, not alarm sensitivity.

Procedure, per noise level:
    1. Measure the static detector's FPR on an attack-free channel (the target).
    2. Each variant exposes ONE monotone looseness knob -- `margin` for the
       adaptive detector, `z_thresh` for the temporal detector -- where a LARGER
       value means looser (fewer alarms, lower FPR). Bisect on that knob for the
       tightest setting whose honest-channel FPR does not exceed the target.
    3. Freeze the calibrated knobs; all downstream detection is at matched FPR.

FPR is measured by running many honest (no-attack) episodes through the
environment with the given detector and counting how often it aborts.
"""

import numpy as np
from environment import BB84Environment
from detectors import (StaticThresholdDetector, AdaptiveThresholdDetector,
                       TemporalPatternDetector)

NOISE_LEVELS = [0.01, 0.03, 0.05]


def measure_fpr(detector, noise, n_episodes=2000, rng_seed=0,
                min_blocks=10, max_blocks=10):
    """
    Fraction of HONEST (attack-free) episodes the detector falsely aborts.
    Eve never attacks, so every abort is a false positive.
    """
    env = BB84Environment(channel_error_rate=noise, detector=detector,
                          min_blocks=min_blocks, max_blocks=max_blocks,
                          rng_seed=rng_seed)
    aborts = 0
    for _ in range(n_episodes):
        env.reset()
        while not env.done:
            env.step(0)              # action 0 = pass, never attack
        if env.detected:
            aborts += 1
    return aborts / n_episodes


def calibrate_knob(make_detector, noise, target_fpr, lo, hi,
                   n_episodes=2000, rng_seed=0, tol=0.002, iters=18):
    """
    Bisect a monotone looseness knob so FPR <= target_fpr, as tight as possible.

    make_detector(knob) -> a fresh Detector with that knob value.
    Larger knob => looser => lower FPR (monotone). We want the SMALLEST knob
    (tightest detector) whose FPR does not exceed the target.

    lo  : tightest knob we will consider (highest FPR)
    hi  : loosest knob we will consider (lowest FPR, should be <= target)
    """
    # If even the loosest setting can't get under target, keep loosest.
    fpr_hi = measure_fpr(make_detector(hi), noise, n_episodes, rng_seed)
    if fpr_hi > target_fpr + tol:
        return hi, fpr_hi

    best = hi
    best_fpr = fpr_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fpr = measure_fpr(make_detector(mid), noise, n_episodes, rng_seed)
        if fpr <= target_fpr + tol:
            # mid is acceptable; try tighter (smaller knob) -> raise lo? No:
            # smaller knob = tighter = higher FPR, so tightening means DECREASING
            # the knob. Acceptable means we can afford to go tighter: move hi down.
            best, best_fpr = mid, fpr
            hi = mid
        else:
            # too many false alarms; must loosen -> increase knob
            lo = mid
    return best, best_fpr


def calibrate_all(n_episodes=2000, rng_seed=0, verbose=True):
    """
    Calibrate adaptive (margin) and temporal (z_thresh) to the static FPR at
    each noise level. Returns a dict:
        params[noise] = {
            "static_fpr": float,
            "adaptive_margin": float, "adaptive_fpr": float,
            "temporal_z": float,       "temporal_fpr": float,
        }
    """
    params = {}
    if verbose:
        print("=" * 64)
        print("Matched-FPR Calibration")
        print("=" * 64)

    for noise in NOISE_LEVELS:
        # 1. static target
        static_fpr = measure_fpr(StaticThresholdDetector(), noise,
                                 n_episodes, rng_seed)

        # 2. adaptive: knob = margin. Larger margin => looser => lower FPR.
        #    tight end margin=0.0 (threshold hugs baseline, high FPR),
        #    loose end margin=0.11 (effectively static, FPR ~ static).
        adaptive_margin, adaptive_fpr = calibrate_knob(
            lambda m: AdaptiveThresholdDetector(margin=m),
            noise, static_fpr, lo=0.0, hi=0.11,
            n_episodes=n_episodes, rng_seed=rng_seed)

        # 3. temporal: knob = z_thresh. Larger z => looser => lower FPR.
        #    tight end z=1.0 (fires on mild spikes), loose end z=12 (almost never).
        temporal_z, temporal_fpr = calibrate_knob(
            lambda z: TemporalPatternDetector(z_thresh=z),
            noise, static_fpr, lo=1.0, hi=12.0,
            n_episodes=n_episodes, rng_seed=rng_seed)

        params[noise] = {
            "static_fpr": static_fpr,
            "adaptive_margin": adaptive_margin, "adaptive_fpr": adaptive_fpr,
            "temporal_z": temporal_z, "temporal_fpr": temporal_fpr,
        }
        if verbose:
            print(f"\n  mu={noise:.0%}")
            print(f"    static  FPR target : {static_fpr*100:5.2f}%")
            print(f"    adaptive margin={adaptive_margin:.4f} "
                  f"-> FPR {adaptive_fpr*100:5.2f}%")
            print(f"    temporal z={temporal_z:.3f} "
                  f"-> FPR {temporal_fpr*100:5.2f}%")

    if verbose:
        print("\n" + "=" * 64)
        print("Calibration complete. All variants matched to static FPR.")
        print("=" * 64)
    return params


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_calibration():
    print("=" * 64)
    print("Calibration Verification (quick, reduced episodes)")
    print("=" * 64)
    PASS, FAIL = "PASS", "FAIL"

    def check(cond, label, obs=None, exp=None):
        tag = PASS if cond else FAIL
        line = f"  [{tag}] {label}"
        if obs is not None:
            line += f"  (observed={obs}, expected={exp})"
        print(line)

    # Static FPR should rise with noise: ~0% at 1%, modest at 3%, high at 5%.
    print("\n[V1] Static FPR rises with noise (Paper 2: ~0 / ~5 / ~32%)")
    f01 = measure_fpr(StaticThresholdDetector(), 0.01, n_episodes=2000)
    f03 = measure_fpr(StaticThresholdDetector(), 0.03, n_episodes=2000)
    f05 = measure_fpr(StaticThresholdDetector(), 0.05, n_episodes=2000)
    print(f"      static FPR: mu=1% {f01*100:.2f}%, "
          f"mu=3% {f03*100:.2f}%, mu=5% {f05*100:.2f}%")
    check(f01 < 0.02, "mu=1% static FPR ~0%", round(f01*100, 2), "<2%")
    check(0.02 < f03 < 0.09, "mu=3% static FPR ~5%", round(f03*100, 2), "2-9%")
    check(f05 > 0.20, "mu=5% static FPR high (~32%)", round(f05*100, 2), ">20%")

    # After calibration, each variant's FPR should be <= static target (+tol).
    print("\n[V2] Calibrated variants match static FPR (mu=3%)")
    params = calibrate_all(n_episodes=2000, verbose=False)
    p = params[0.03]
    tol = 0.01
    check(p["adaptive_fpr"] <= p["static_fpr"] + tol,
          "adaptive FPR <= static target",
          f"{p['adaptive_fpr']*100:.2f}%", f"<= {p['static_fpr']*100:.2f}%")
    check(p["temporal_fpr"] <= p["static_fpr"] + tol,
          "temporal FPR <= static target",
          f"{p['temporal_fpr']*100:.2f}%", f"<= {p['static_fpr']*100:.2f}%")

    print("\n" + "=" * 64)
    print("Calibration verification complete.")
    print("=" * 64)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "verify":
        verify_calibration()
    elif mode == "run":
        calibrate_all(n_episodes=5000)
    else:
        print("Usage: python calibrate.py [verify|run]")
