# GADDA-RL 🎮
### Generalizable & Uncertainty-Aware Dynamic Difficulty Adjustment using Reinforcement Learning

> A research project exploring whether Reinforcement Learning can adapt game difficulty in real time to maintain a target player experience under changing and uncertain player behaviour.

## 🔬 Research Question

**Can an RL-based Dynamic Difficulty Adjustment (DDA) policy generalize to player conditions not encountered during training?**

## 🧠 Approach

GADDA-RL models simulated players with different behaviours:

- Stable
- Learning
- Fatigue
- Noisy

The PPO agent observes:

`Estimated Skill · Uncertainty · Recent Win Rate · Error Rate · Episode Progress`

and continuously selects a game difficulty between **0 and 1**.

The reward encourages:
- Target win-rate tracking
- Smooth difficulty adaptation
- Player engagement
- Lower uncertainty

## 📊 Evaluation

The trained PPO agent is compared against:

- Static Easy
- Static Medium
- Static Hard
- Rule-Based DDA

### Generalization Experiment

The frozen PPO policy was evaluated on **5 held-out simulated player conditions** using **10 random seeds per condition**.

| Method | Mean Win Rate Error |
|---|---:|
| **PPO DDA** | **0.0800** |
| Rule-Based DDA | 0.1708 |
| Static Easy | 0.2414 |
| Static Medium | 0.3261 |
| Static Hard | 0.4088 |

Across the 50 matched generalization runs, PPO showed **53.16% lower mean win-rate error than the Rule-Based DDA** (paired test: *p* = 2.45 × 10⁻⁷, Cohen's *d* = 0.85).

Performance varied across unseen profiles, providing evidence of promising but profile-dependent generalization rather than universal generalization.

## 📁 Project Structure

```text
GADDA-RL/
├── notebooks/
├── src/
├── results/
├── figures/
├── tables/
├── models/
├── README.md
└── requirements.txt
```

<summary>🔎 View current research milestone</summary>

### Current Milestone — Generalization

The trained PPO policy has been evaluated on held-out simulated player conditions outside the training parameter ranges.

**Current focus:** determining how uncertainty-aware RL behaves under different unseen player dynamics.

</details>

## 🛠️ Technologies

`Python` · `NumPy` · `Pandas` · `SciPy` · `Scikit-learn` · `Gymnasium` · `Stable-Baselines3` · `PyTorch` · `Matplotlib` · `Jupyter / Google Colab`

## 🚧 Research Status

**Completed:** Player simulation → Skill estimation → Environment → Baselines → PPO training → Standard evaluation → Generalization

**Next:** Uncertainty ablation → Robustness testing → Final statistical analysis → Manuscript preparation

## ⚠️ Limitations

This study uses simulated players and a simulated environment. Generalization has been evaluated on a limited set of held-out parameter conditions, and no human-player validation has yet been conducted.

## 👤 Author

**Aasmeet Kaur**

B.Tech. Computer Science & Engineering | Research Project

**GitHub:** `aasmeetkaur03`  
**LinkedIn:** `aasmeetkaur1703`

---

> 🎮 *GADDA-RL is an ongoing research project focused on generalizable and uncertainty-aware adaptive difficulty using reinforcement learning.*
