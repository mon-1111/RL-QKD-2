# No Convergence in Learned Eavesdropping–Detection Dynamics for BB84 QKD

Simulation and analysis code for the paper:

> **No Convergence in Learned Eavesdropping–Detection Dynamics for BB84 Quantum Key Distribution**
> Ramon Jose C. Bagunu
> arXiv:2611.16372

This is the follow-on to **Paper 1**, *Adversarial Reinforcement Learning for
Adaptive Eavesdropping in BB84 Quantum Key Distribution* (*Quantum Inf. Process.*
**25**, 369 (2026), [doi:10.1007/s11128-026-05390-z](https://doi.org/10.1007/s11128-026-05390-z)),
whose learned attacker and simulator are reused here unchanged.

---

## What this code shows

Paper 1 showed that a reinforcement-learning eavesdropper can hold the BB84
quantum bit error rate (QBER) below the abort threshold while keeping a usable
distillable key, leaving a stealth window that a fixed detector cannot close.
This repository asks the next question: **can an equally adaptive, learning
detector close that window?**

The attacker and detector are trained against each other as an **alternating
best-response game**, and exploitability is measured across rounds. The central
result is that **the contest does not settle**. Out to twenty best-response
rounds the exploitability does not approach a fixed point, and two control
experiments (one per player) show that the residual round-to-round motion is at
the scale of the best-response procedure's own variance, so neither side
converges to a fixed response. The behavior is noise-dependent, and at
intermediate noise the exploitability trajectory is significantly
anti-persistent, consistent with a last-mover information asymmetry.

---

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`

```bash
pip install -r requirements.txt
```

No GPU, no quantum hardware, and no external data are required. Everything is a
CPU simulation, and a full reproduction is a few hours on a laptop. Every long run is
**resumable** (finished results are skipped on restart) and writes JSON logs to
`results/`.

---

## Repository layout

### Core simulation

| File | Role |
|------|------|
| `bb84_simulator.py` | BB84 protocol with intercept-resend attack and channel noise |
| `environment.py` | The attacker's MDP (state encoding, step, reward) over the simulator |
| `agents.py` | Tabular RL agents: Q-Learning, SARSA, Double Q-Learning |
| `train.py` | Training loop and greedy evaluation for an attacker vs a fixed detector |
| `detectors.py` | Fixed detectors, including the static 11% abort-threshold baseline |
| `learning_defender.py` | The learning defender: a tabular agent that adapts the abort threshold |
| `keyfraction.py` | Distillable key fraction r(e) = 1 − 2h(e) |
| `calibrate.py` | False-positive-rate calibration helper (not used by the learning defender; see note) |

### Game and experiments

| File | Role | Paper output |
|------|------|--------------|
| `game.py` | Alternating best-response game + exploitability metric | — |
| `run_sweep.py` | Full factorial sweep, 3 agents × 3 noise × 5 seeds at K=5 | Table 2, Figs. 1–3 |
| `run_convergence.py` | Extended K=20 runs on targeted cells, 5 seeds each | Table 3, Figs. 1, 4 |
| `run_attacker_variance.py` | Attacker-variance control (freeze defender, re-seed attacker) | Table 4 |
| `run_defender_policy_check.py` | Defender decision dynamics + defender-variance control | Table 5 |

### Analysis and figures

| File | Role |
|------|------|
| `analyze_convergence.py` | First/second-half swing amplitude and reversal rate |
| `analyze_cycle_statistics.py` | Lag-one autocorrelation ρ₁ and reversal-vs-null test (Table 3 ρ₁ column) |
| `figures.py` | Builds all five figures from the saved logs |

---

## Reproducing the paper

Run from the repository root. Each command is resumable.

```bash
# 1. K=5 sweep (Table 2, phase structure)
python3 run_sweep.py run
python3 run_sweep.py analyze

# 2. K=20 convergence runs (Table 3), the longest step
python3 run_convergence.py run
python3 analyze_convergence.py results/convergence
python3 analyze_cycle_statistics.py           # ρ₁ and reversal-vs-null

# 3. Attacker-variance control (Table 4)
python3 run_attacker_variance.py run
python3 run_attacker_variance.py analyze

# 4. Defender decision dynamics + control (Table 5)
python3 run_defender_policy_check.py run
python3 run_defender_policy_check.py analyze
python3 run_defender_policy_check.py run_defvar
python3 run_defender_policy_check.py analyze_defvar

# 5. Figures (fig_a … fig_e into figures/)
python3 figures.py
```

Individual figures: `python3 figures.py d` builds only Fig. 4; valid keys are
`a b c d e`.

### Fast smoke test (minutes, not hours)

To confirm the pipeline runs before committing to the full compute:

```bash
python3 run_convergence.py run_lean
python3 run_attacker_variance.py run_lean
python3 run_defender_policy_check.py run_lean
```

These use the same code paths at reduced scope. The numbers will not match the
paper (too few seeds), but a clean finish confirms the environment is set up
correctly.

---

## Paper ↔ output map

| Paper element | Produced by |
|---------------|-------------|
| Table 2 (exploitability) | `run_sweep.py analyze` |
| Table 3 (swing amplitude, reversal, ρ₁) | `analyze_convergence.py`, `analyze_cycle_statistics.py` |
| Table 4 (attacker-variance control) | `run_attacker_variance.py analyze` |
| Table 5 (defender decision dynamics) | `run_defender_policy_check.py analyze` + `analyze_defvar` |
| Fig. 1 (per-round trajectories) | `figures.py b` |
| Fig. 2 (exploitability vs noise) | `figures.py a` |
| Fig. 3 (detection vs key fraction) | `figures.py c` |
| Fig. 4 (first/second-half amplitude) | `figures.py d` |
| Fig. 5 (last-mover advantage) | `figures.py e` |

---

## Notes on design and reproducibility

- **Determinism.** All runs are seeded. The control scripts reconstruct a given
  round's frozen opponent by replaying the game's deterministic seed schedule, so
  no per-round policies need to be serialized.
- **`calibrate.py` is not used by the learning defender.** The defender's
  false-alarm rate is disciplined uniformly by `honest_frac = 0.5`, not by a
  per-noise FPR calibration. The paper states this explicitly; the file is kept
  for completeness.
- **Defender decision metric (Table 5).** The defender's threshold *label*
  changes in most states per round, but many changes are inert (a QBER bin below
  the lowest threshold never aborts, above the highest always aborts). Table 5
  therefore reports the change in the actual abort/continue **decision** over
  decision-relevant states only.
- **If you change the metric or the code, delete the affected `results/`
  subfolder before re-running**, since runs skip existing logs by filename.

---

## Citation

If you use this code, please cite the paper and the Paper 1 baseline:

```bibtex
@misc{bagunu2026noconvergence,
  author = {Bagunu, R. J. C.},
  title  = {No Convergence in Learned Eavesdropping--Detection Dynamics for BB84 Quantum Key Distribution},
  note   = {arXiv:2611.16372},
  year   = {2026}
}

@article{bagunu2026adversarial,
  author  = {Bagunu, R. J. C.},
  title   = {Adversarial reinforcement learning for adaptive eavesdropping in BB84 quantum key distribution},
  journal = {Quantum Inf. Process.},
  volume  = {25},
  pages   = {369},
  year    = {2026},
  doi     = {10.1007/s11128-026-05390-z}
}
```

---

## License

Released under the MIT License. See `LICENSE`.
