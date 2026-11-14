"""
Paper 2 figures — learning defender / non-converging best-response game
=======================================================================
Reads the logs you ALREADY HAVE. No new compute.

    results/sweep/*.json          45 K=5 runs  (3 agents x 3 noise x 5 seeds)
    results/convergence/*.json    10 K=20 runs (the keystone)

Each log dict (produced by game.run_game):
    agent, noise, K   (NOTE: game.py does not store `seed` in the log; figures
                       group by (agent, noise) and overlay all seeds, so this
                       is fine — we never key on seed.)
    rounds: [ {round, fresh_attacker_det, fresh_attacker_keyfrac,
               fresh_attacker_correct, cotrained_attacker_det}, ... ]
    exploitability: {det, keyfrac, correct}         # fresh atk vs FINAL defender
    convergence:    {settled, tail_std, fresh_attacker_det_trajectory}
Detection/keyfrac are stored as fractions in [0,1]; we convert det to %.

Figures produced (saved to figures/):
  fig_b_cycling.png     (BUILD FIRST) per-round fresh-attacker detection vs round,
                         one panel per noise regime, overlaying agents/seeds.
                         The visceral "it never settles" figure — the thesis.
  fig_a_gradient.png     exploitability (det + keyfrac) vs noise, mean +/- seed
                         spread, per agent. The three-regime phase structure.
  fig_c_keyfrac.png      distillable key fraction vs detection %, final-defender
                         exploitability points, colored by regime. The trade-off.
  fig_d_amplitude.png    (from convergence/ only) 1st-half vs 2nd-half cycle
                         amplitude bars — the K=20 "amplitude does not decay"
                         evidence in one glance. Needs K=20 logs; skipped if none.

Usage:
    python figures.py                 # build all, reading default dirs
    python figures.py b               # build only figure (b)
    python figures.py a c             # build figures (a) and (c)
    python figures.py --sweep DIR --conv DIR   # override log directories
"""

import os
import sys
import json
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
SWEEP_DIR = "results/sweep"
CONV_DIR  = "results/convergence"
OUT_DIR   = "figures"

AGENTS = ["qlearning", "sarsa", "doubleq"]
NOISES = [0.01, 0.03, 0.05]

# Palette matched to Paper 1 (muted agent scheme), sampled from its figures.
# Agent fills are the muted series colors; AGENT_TITLE are the darker bold
# tones Paper 1 uses for its colored panel titles.
AGENT_LABEL = {"qlearning": "Q-Learning",
               "sarsa":     "SARSA",
               "doubleq":   "Double Q-Learning"}
AGENT_COLOR = {"qlearning": "#5080B0",   # muted steel blue
               "sarsa":     "#C89068",   # muted ochre
               "doubleq":   "#709880"}   # muted sage green
AGENT_TITLE = {"qlearning": "#3A6EA5",   # bold blue  (Paper 1 titles)
               "sarsa":     "#C08552",   # bold ochre
               "doubleq":   "#5A8A6B"}   # bold green
# Paper 1's two non-RL baselines (kept for continuity if we ever add them here)
BASELINE_COLOR = {"fixed": "#9880A0", "always": "#A87070"}
# Paper 1 facets by noise rather than coloring by it; pick three tones in the
# same muted family so fig_c stays visually consistent with Paper 1.
NOISE_COLOR = {0.01: "#5080B0", 0.03: "#C89068", 0.05: "#8A8A8A"}
NOISE_LABEL = {0.01: r"$\mu=1\%$", 0.03: r"$\mu=3\%$", 0.05: r"$\mu=5\%$"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})


# ----------------------------------------------------------------------------
# loading helpers (defensive: tolerate missing keys / either detector storage)
# ----------------------------------------------------------------------------
def _as_pct(x):
    """A detection value stored as fraction (0-1) -> percent. Pass through %>1."""
    if x is None:
        return np.nan
    return x * 100.0 if x <= 1.0 else x


def load_logs(directory):
    """Return list of log dicts from a directory, skipping unreadable files."""
    if not os.path.isdir(directory):
        return []
    logs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path) as fh:
                logs.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {os.path.basename(path)} ({e})")
    return logs


def det_trajectory(log):
    """Per-round fresh-attacker detection trajectory in percent.

    Primary source is log["rounds"][*]["fresh_attacker_det"]. Falls back to
    convergence["fresh_attacker_det_trajectory"] (the same numbers game.py also
    stores there) if rounds is empty/missing — belt and suspenders."""
    rounds = log.get("rounds", [])
    traj = [_as_pct(r.get("fresh_attacker_det")) for r in rounds]
    if traj and not all(np.isnan(traj)):
        return traj
    fallback = log.get("convergence", {}).get("fresh_attacker_det_trajectory")
    if fallback:
        return [_as_pct(x) for x in fallback]
    return traj


def keyfrac_trajectory(log):
    rounds = log.get("rounds", [])
    return [r.get("fresh_attacker_keyfrac", np.nan) for r in rounds]


def group_by_cell(logs):
    """{(agent, noise): [log, ...]}"""
    cells = {}
    for log in logs:
        key = (log.get("agent"), float(log.get("noise")))
        cells.setdefault(key, []).append(log)
    return cells


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# FIGURE B  — per-round cycling trajectories (BUILD FIRST: carries the thesis)
# ----------------------------------------------------------------------------
def fig_b_cycling(sweep_logs, conv_logs):
    """
    One panel per noise regime. x = best-response round, y = fresh-attacker
    detection %. Prefer the long K=20 convergence logs where available (they
    show the cycle persists); fall back to the K=5 sweep logs otherwise.
    Thin lines = individual seeds, colored by agent; this makes the cycling
    (vs the mu=5% lock, vs mu=1% agent-dependence) immediately visible.
    """
    # choose the richest source per cell: K=20 if present, else K=5
    conv_cells  = group_by_cell(conv_logs)
    sweep_cells = group_by_cell(sweep_logs)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, noise in zip(axes, NOISES):
        used_agents = set()
        for agent in AGENTS:
            key = (agent, noise)
            logs = conv_cells.get(key) or sweep_cells.get(key) or []
            for log in logs:
                traj = det_trajectory(log)
                if not traj or all(np.isnan(traj)):
                    continue
                rounds = np.arange(1, len(traj) + 1)
                ax.plot(rounds, traj, color=AGENT_COLOR[agent],
                        alpha=0.65, linewidth=1.4, marker="o", markersize=3)
                used_agents.add(agent)
        ax.axhline(50, color="k", linewidth=0.7, linestyle=":", alpha=0.5)
        # annotate each panel with the longest K actually plotted, so the
        # shorter mu=5% panel (K=5) isn't read as a compressed version of the
        # K=20 panels.
        kmax = 0
        for agent in AGENTS:
            key = (agent, noise)
            for log in (conv_cells.get(key) or sweep_cells.get(key) or []):
                kmax = max(kmax, len(det_trajectory(log)))
        ax.set_title(f"{NOISE_LABEL[noise]}  (K={kmax})")
        ax.set_ylim(-3, 103)
        ax.margins(x=0.02)
    axes[0].set_ylabel("fresh-attacker detection (%)")
    # single shared x-label centered under all three panels (not one per panel)
    try:
        fig.supxlabel("best-response round")
    except AttributeError:
        # matplotlib < 3.4: place one centered label manually
        fig.text(0.5, 0.02, "best-response round", ha="center")

    handles = [Line2D([0], [0], color=AGENT_COLOR[a], lw=2,
                      label=AGENT_LABEL[a]) for a in AGENTS]
    # push the legend clear of the x-axis labels
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.12))
    # no suptitle: the LaTeX caption carries the title
    fig.subplots_adjust(bottom=0.20)
    out = os.path.join(OUT_DIR, "fig_b_cycling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------
# FIGURE A  — exploitability gradient vs noise (phase structure)
# ----------------------------------------------------------------------------
def fig_a_gradient(sweep_logs, conv_logs):
    """
    Exploitability of the FINAL defender vs noise, mean +/- std over seeds,
    one series per agent. Two stacked panels: detection % (top) and
    distillable key fraction (bottom). Shows locked (mu=5%) / cycling (mu=3%)
    / attacker-dominant (mu=1%). Uses the sweep (5 seeds) for spread; falls
    back to whatever logs exist.
    """
    logs = sweep_logs if sweep_logs else conv_logs
    cells = group_by_cell(logs)

    fig, (ax_d, ax_k) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    jitter = {"qlearning": -0.0008, "sarsa": 0.0, "doubleq": 0.0008}

    for agent in AGENTS:
        xs, dm, ds, km, ks = [], [], [], [], []
        for noise in NOISES:
            vals = cells.get((agent, noise), [])
            dets = [_as_pct(v.get("exploitability", {}).get("det")) for v in vals]
            kfs  = [v.get("exploitability", {}).get("keyfrac") for v in vals]
            dets = [x for x in dets if x is not None and not np.isnan(x)]
            kfs  = [x for x in kfs if x is not None and not np.isnan(x)]
            if not dets:
                continue
            xs.append(noise + jitter[agent])
            dm.append(np.mean(dets)); ds.append(np.std(dets))
            km.append(np.mean(kfs));  ks.append(np.std(kfs))
        if not xs:
            continue
        c = AGENT_COLOR[agent]
        ax_d.errorbar(xs, dm, yerr=ds, color=c, marker="o", capsize=3,
                      linewidth=1.6, label=AGENT_LABEL[agent])
        ax_k.errorbar(xs, km, yerr=ks, color=c, marker="o", capsize=3,
                      linewidth=1.6, label=AGENT_LABEL[agent])

    ax_d.set_ylabel("exploitability:\ndetection (%)")
    ax_d.set_ylim(-3, 103)
    ax_k.set_ylabel("exploitability:\ndistillable key fraction")
    ax_k.set_ylim(-0.03, 1.0)
    ax_k.set_xlabel(r"channel noise $\mu$")
    ax_k.set_xticks(NOISES)
    ax_k.set_xticklabels([f"{int(n*100)}%" for n in NOISES])
    ax_k.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax_d.legend(frameon=False)
    out = os.path.join(OUT_DIR, "fig_a_gradient.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------
# FIGURE C  — key-fraction / detection trade-off
# ----------------------------------------------------------------------------
def fig_c_keyfrac(sweep_logs, conv_logs):
    """
    Scatter of final-defender exploitability points: x = detection %,
    y = distillable key fraction, colored by noise regime. Upper-left =
    window open (low det, high key). Shows the regimes sit in different
    corners, and the mu=1% cloud is spread out (= non-convergence).
    """
    logs = sweep_logs if sweep_logs else conv_logs
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for noise in NOISES:
        xs, ys = [], []
        for log in logs:
            if abs(float(log.get("noise")) - noise) > 1e-9:
                continue
            e = log.get("exploitability", {})
            d = _as_pct(e.get("det")); k = e.get("keyfrac")
            if d is None or k is None or np.isnan(d):
                continue
            xs.append(d); ys.append(k)
        if xs:
            ax.scatter(xs, ys, color=NOISE_COLOR[noise], s=55, alpha=0.75,
                       edgecolor="white", linewidth=0.6, label=NOISE_LABEL[noise])

    ax.set_xlabel("fresh-attacker detection (%)")
    ax.set_ylabel("distillable key fraction  $r = 1 - 2h(e)$")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.03, 1.0)
    ax.annotate("window OPEN\n(stealthy, usable key)", xy=(8, 0.9),
                fontsize=9, color="#333333", ha="left")
    ax.annotate("contained", xy=(78, 0.08), fontsize=9,
                color="#333333", ha="left")
    ax.legend(frameon=False, title="channel noise")
    out = os.path.join(OUT_DIR, "fig_c_keyfrac.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------
# FIGURE D  — half-amplitude bars (K=20 convergence logs only)
# ----------------------------------------------------------------------------
def _half_amplitudes(traj):
    traj = [x for x in traj if not np.isnan(x)]
    n = len(traj)
    if n < 4:
        return np.nan, np.nan
    mid = n // 2
    return float(np.std(traj[:mid])), float(np.std(traj[mid:]))


def fig_d_amplitude(sweep_logs, conv_logs):
    """
    First-half vs second-half std of the fresh-attacker detection trajectory,
    per (agent, noise) cell, from the K=20 convergence logs. If the second
    bar is not shorter than the first, the cycle does NOT decay -> no
    convergence. Skipped with a note if there are no K=20 logs.
    """
    if not conv_logs:
        print("  SKIP fig_d: no convergence logs found (need K=20 runs in "
              f"{CONV_DIR}).")
        return
    cells = group_by_cell(conv_logs)
    labels, first, second = [], [], []
    for agent in AGENTS:
        for noise in NOISES:
            logs = cells.get((agent, noise), [])
            if not logs:
                continue
            a1s, a2s = [], []
            for log in logs:
                a1, a2 = _half_amplitudes(det_trajectory(log))
                if not np.isnan(a1):
                    a1s.append(a1); a2s.append(a2)
            if not a1s:
                continue
            labels.append(f"{AGENT_LABEL[agent]}\n{int(noise*100)}%")
            first.append(np.mean(a1s)); second.append(np.mean(a2s))

    if not labels:
        print("  SKIP fig_d: convergence logs had no usable trajectories.")
        return

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 1.7 * len(labels)), 4.8))
    ax.bar(x - w / 2, first,  w, label="first half of rounds",
           color="#5080B0", edgecolor="#3A6EA5")
    ax.bar(x + w / 2, second, w, label="second half of rounds",
           color="#C89068", edgecolor="#C08552")
    # labels use " @ " on one line and rotate slightly so adjacent two-line
    # labels can't collide (the Double-Q cells sat right next to each other).
    labels = [lab.replace("\n", " @ ") for lab in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=18, ha="right")
    ax.set_ylabel("swing amplitude\n(std of detection, pp)")
    ax.legend(frameon=False)
    out = os.path.join(OUT_DIR, "fig_d_amplitude.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------
# FIGURE E  — last-mover mechanism: cotrained vs fresh detection per round
# ----------------------------------------------------------------------------
def _cotrained_trajectory(log):
    """Per-round cotrained-attacker detection (old attacker vs NEW defender)."""
    rounds = log.get("rounds", [])
    return [_as_pct(r.get("cotrained_attacker_det")) for r in rounds]


def fig_e_lastmover(sweep_logs, conv_logs):
    """
    The last-mover mechanism, shown as the per-round GAP rather than as raw
    overlaid trajectories (which overplot into a hairball with 5 seeds x 2
    curves x 3 agents).

    game.py logs, per round, both:
      cotrained_attacker_det -- the FROZEN old attacker measured against the
                                NEWLY trained defender (defender moved last).
      fresh_attacker_det     -- a FRESH attacker trained against that same
                                defender (attacker moved last).
    The last-mover advantage is the difference
        gap(round) = cotrained_attacker_det - fresh_attacker_det,
    which is POSITIVE whenever the player who moved last (here the defender,
    against the stale attacker) beats the player who moved first. We plot the
    seed-mean gap per round, one line per agent, with a shaded band spanning
    +/- one std across seeds, and a reference line at zero. A gap that stays
    above zero across rounds is the mechanism: whoever moves last wins, every
    round, and the advantage does not close.

    One panel per noise regime. Prefers K=20 convergence logs; falls back to
    the K=5 sweep.
    """
    conv_cells  = group_by_cell(conv_logs)
    sweep_cells = group_by_cell(sweep_logs)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, noise in zip(axes, NOISES):
        kmax = 0
        for agent in AGENTS:
            key = (agent, noise)
            logs = conv_cells.get(key) or sweep_cells.get(key) or []
            # collect per-seed gap trajectories, truncated to the common length
            gaps = []
            for log in logs:
                cot = _cotrained_trajectory(log)
                fr  = det_trajectory(log)
                if not fr or all(np.isnan(fr)) or not cot or all(np.isnan(cot)):
                    continue
                n = min(len(cot), len(fr))
                g = np.array(cot[:n]) - np.array(fr[:n])
                gaps.append(g)
                kmax = max(kmax, n)
            if not gaps:
                continue
            n = min(len(g) for g in gaps)
            M = np.vstack([g[:n] for g in gaps])
            rounds = np.arange(1, n + 1)
            mean = M.mean(axis=0)
            sd   = M.std(axis=0)
            ax.fill_between(rounds, mean - sd, mean + sd,
                            color=AGENT_COLOR[agent], alpha=0.15, linewidth=0)
            ax.plot(rounds, mean, color=AGENT_COLOR[agent],
                    linewidth=1.8, marker="o", markersize=3,
                    label=AGENT_LABEL[agent])
        ax.axhline(0, color="k", linewidth=0.9, linestyle="-", alpha=0.6)
        ax.set_title(f"{NOISE_LABEL[noise]}  (K={kmax})")
        ax.margins(x=0.02)
    axes[0].set_ylabel("last-mover advantage\n(cotrained $-$ fresh detection, pp)")
    try:
        fig.supxlabel("best-response round")
    except AttributeError:
        fig.text(0.5, 0.02, "best-response round", ha="center")

    agent_handles = [Line2D([0], [0], color=AGENT_COLOR[a], lw=2,
                            label=AGENT_LABEL[a]) for a in AGENTS]
    fig.legend(handles=agent_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.20)
    out = os.path.join(OUT_DIR, "fig_e_lastmover.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
BUILDERS = {
    "b": fig_b_cycling,
    "a": fig_a_gradient,
    "c": fig_c_keyfrac,
    "d": fig_d_amplitude,
    "e": fig_e_lastmover,
}


def main(argv):
    global SWEEP_DIR, CONV_DIR
    which = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--sweep":
            SWEEP_DIR = argv[i + 1]; i += 2; continue
        if a == "--conv":
            CONV_DIR = argv[i + 1]; i += 2; continue
        if a in BUILDERS:
            which.append(a)
        else:
            print(f"  (ignoring unknown arg '{a}'; valid: a b c d e)")
        i += 1
    if not which:
        which = ["b", "a", "c", "d", "e"]   # b first on purpose

    ensure_out()
    print("Loading logs...")
    sweep_logs = load_logs(SWEEP_DIR)
    conv_logs  = load_logs(CONV_DIR)
    print(f"  {len(sweep_logs)} sweep logs ({SWEEP_DIR}), "
          f"{len(conv_logs)} convergence logs ({CONV_DIR})")
    if not sweep_logs and not conv_logs:
        print("No logs found. Run the sweep / convergence runs first, or point "
              "at the right dirs with --sweep / --conv.")
        return

    print("Building figures:", ", ".join(which))
    for key in which:
        print(f"[{key}]")
        BUILDERS[key](sweep_logs, conv_logs)
    print(f"\nDone. PNGs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main(sys.argv[1:])