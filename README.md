# GADDA-RL
## Generalizable and Uncertainty-Aware Dynamic Difficulty Adjustment using Reinforcement Learning

<p align="center">
  <strong>Can reinforcement learning learn to adapt game difficulty to changing and previously unseen player behaviour?</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Research-Reinforcement%20Learning-blue" alt="Research">
  <img src="https://img.shields.io/badge/Method-PPO-orange" alt="Method">
  <img src="https://img.shields.io/badge/Domain-Dynamic%20Difficulty%20Adjustment-green" alt="DDA">
  <img src="https://img.shields.io/badge/Status-Ongoing-yellow" alt="Status">
</p>

---

## 📌 Project Status

> **Current milestone: Generalisation evaluation completed.**

| Research Stage | Status |
|---|---|
| Player Simulation | ✅ Complete |
| Skill Estimation | ✅ Complete |
| RL Environment | ✅ Complete |
| Baseline Methods | ✅ Complete |
| PPO Training | ✅ Complete |
| Standard Evaluation | ✅ Complete |
| Generalisation Evaluation | ✅ Complete |
| Uncertainty Ablation | 🔜 Planned |
| Robustness Testing | 🔜 Planned |
| Final Statistical Analysis | 🔜 Planned |
| Manuscript | 🔜 Planned |
| Interactive Research Dashboard | 🔜 Planned |

GADDA-RL is an ongoing research project investigating whether reinforcement learning can perform adaptive Dynamic Difficulty Adjustment under changing, uncertain, and previously unseen simulated player behaviour.

The current results are **simulation-based**. Human-player validation has not yet been conducted.

---

# 🧠 Research Question

> **Can a reinforcement-learning-based Dynamic Difficulty Adjustment system learn a generalizable difficulty adaptation policy that maintains player engagement under uncertainty and changing player skill, including player profiles not encountered during training?**

The project studies whether an RL controller can continuously adapt game difficulty using observed player behaviour instead of relying only on fixed difficulty levels or manually designed rules.

---

# 🎯 Motivation

Dynamic Difficulty Adjustment (DDA) aims to adapt game difficulty to the player's current experience.

A simple system may use:

```text
Player Performance
       ↓
Rule / Threshold
       ↓
Difficulty Change
