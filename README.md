# GADDA: Generalizable and Uncertainty-Aware Dynamic Difficulty Adjustment using Reinforcement Learning

**GADDA** is an undergraduate research project investigating whether an uncertainty-aware Reinforcement Learning (RL) agent can dynamically adjust game difficulty toward a target player success rate under changing and uncertain player behaviour.

The study focuses on **generalization, uncertainty, and adaptive decision-making** rather than simply training an RL agent to control difficulty.

---

## 🔬 Research Question

> **Can an RL-based Dynamic Difficulty Adjustment (DDA) policy generalize to player conditions not encountered during training, and does uncertainty in the player-state representation contribute to target-tracking performance?**

The study evaluates this through standard baseline comparison, held-out generalization, uncertainty ablation, and behavioural-noise robustness experiments.

---

## 🧠 GADDA Approach

GADDA models player behaviour in a controlled simulation and trains a **Proximal Policy Optimization (PPO)** agent to continuously select game difficulty.

The simulated training population includes:

- **Stable players**
- **Learning players**
- **Fatigue-affected players**
- **Noisy players**

The PPO agent receives a five-dimensional player state:

| State Variable | Description |
|---|---|
| Estimated Skill | Estimated current player ability |
| Uncertainty | Uncertainty associated with the skill estimate |
| Recent Win Rate | Recent player success rate |
| Recent Error Rate | Recent player error behaviour |
| Episode Progress | Normalized progress through the episode |

The agent outputs a continuous difficulty value between **0 and 1**.

### Reward Design

The reward combines four components:

- **Target-tracking term** — encourages the player's win rate to remain close to the target of **0.70**
- **Difficulty-change penalty** — discourages abrupt difficulty changes
- **Error-related term** — incorporates observed player error behaviour
- **Uncertainty penalty** — accounts for uncertainty in the estimated player state

The objective is therefore not simply to maximize reward, but to learn a difficulty-control policy that can maintain target-based adaptation while responding to changing player behaviour.

---

# 📊 Experimental Evaluation

GADDA was evaluated against four baselines:

- **Static Easy** — difficulty = 0.30
- **Static Medium** — difficulty = 0.50
- **Static Hard** — difficulty = 0.70
- **Rule-Based DDA** — deterministic target-tracking controller

The primary metric is **mean target win-rate error**, where lower values indicate better tracking of the target success rate.

---

## 1. Standard Evaluation

Across **40 matched evaluation runs**, the PPO controller achieved:

| Method | Mean Target Win-Rate Error |
|---|---:|
| **GADDA PPO** | **0.0855 ± 0.0103** |
| Rule-Based DDA | 0.1340 ± 0.0176 |
| Static Easy | 0.2303 |
| Static Medium | 0.2045 |
| Static Hard | 0.5183 |

Compared with the Rule-Based DDA:

- **36.19% lower target win-rate error**
- **Paired Cohen's d = 2.98**

These results indicate substantially better target tracking under the standard evaluation conditions.

---

# 2. Held-Out Generalization

To evaluate generalization, the trained PPO policy was **frozen and evaluated without retraining** on five simulated player configurations whose parameter values were outside the training ranges.

The evaluation used **10 random seeds per configuration**, giving **50 matched generalization runs**.

### Overall Generalization Results

| Method | Mean Target Win-Rate Error |
|---|---:|
| **GADDA PPO** | **0.0800** |
| Rule-Based DDA | 0.1708 |
| Static Easy | 0.2414 |
| Static Medium | 0.3261 |
| Static Hard | 0.4088 |

GADDA PPO achieved:

**53.16% lower mean target win-rate error than Rule-Based DDA**

with:

- **p = 2.45 × 10⁻⁷**
- **Paired Cohen's d = 0.85**

### Held-Out Profiles

| Held-Out Player Condition | GADDA PPO | Rule-Based DDA |
|---|---:|---:|
| Unseen Low Skill | 0.1216 | 0.0207 |
| Unseen High Skill | 0.1283 | 0.2964 |
| Unseen Fast Learning | 0.0707 | 0.2375 |
| Unseen Strong Fatigue | 0.0269 | 0.1794 |
| Unseen High Noise | 0.0525 | 0.1199 |

Performance varied across unseen profiles. In particular, the Rule-Based DDA performed better than PPO on the **Unseen Low Skill** profile.

Therefore, the results support **promising but profile-dependent generalization**, rather than universal generalization.

---

# 3. Uncertainty Ablation

A matched ablation experiment was conducted to investigate whether uncertainty in the player-state representation contributes to performance.

Two PPO models were compared:

- **Full GADDA:** estimated skill + uncertainty + other state variables
- **No-Uncertainty GADDA:** uncertainty removed from the observation

### Results

| Model | Mean Target Win-Rate Error |
|---|---:|
| **Full GADDA** | **0.0800** |
| No-Uncertainty GADDA | 0.1949 |

Removing uncertainty increased mean target-tracking error by approximately **143.66% relative to the full model**.

The paired analysis showed:

- Mean error increase: **0.1149**
- 95% CI: **[0.0940, 0.1358]**
- **p = 6.41 × 10⁻¹⁵**
- **Paired Cohen's d = 1.56**

Equivalently, retaining uncertainty reduced error by approximately **58.96% relative to the no-uncertainty model**.

These results provide evidence that uncertainty is an important component of the player-state representation **within the tested simulation conditions**.

---

# 4. Behavioural-Noise Robustness

The frozen PPO policy was evaluated under increasing behavioural noise levels:

**0.05 → 0.10 → 0.15 → 0.20 → 0.30**

Mean target win-rate error remained low across the tested range:

| Noise Level | Mean Target Win-Rate Error |
|---:|---:|
| 0.05 | 0.0256 |
| 0.10 | 0.0216 |
| 0.15 | 0.0214 |
| 0.20 | 0.0178 |
| 0.30 | 0.0107 |

The target-tracking metric did not degrade as behavioural noise increased in this experiment.

However, **total reward decreased at the highest noise level**, indicating that stable target tracking did not imply that all aspects of the reward objective were unaffected by noise.

The robustness experiment was conducted on a single base player configuration, so these results should not be interpreted as universal robustness to all forms of behavioural variability.

---

# 🔎 Key Findings

The completed experiments provide four main findings:

### 1. Better target tracking under standard evaluation

GADDA PPO achieved **36.19% lower target win-rate error** than the Rule-Based DDA baseline.

### 2. Promising but profile-dependent generalization

On held-out player configurations outside the training ranges, PPO achieved **53.16% lower aggregate error** than the Rule-Based DDA.

However, performance varied by player profile, including one held-out condition where the rule-based controller performed better.

### 3. Uncertainty substantially contributes to performance

Removing uncertainty from the player-state representation increased error from **0.0800 to 0.1949**, with a large paired effect size (**d = 1.56**).

### 4. Target tracking remained stable under tested behavioural noise

Across noise levels from **0.05 to 0.30**, target-tracking error remained low, although total reward decreased at the highest noise level.

---

# 🧪 What the Research Demonstrates

The results provide evidence that an **uncertainty-aware PPO formulation can support target-based adaptive difficulty control under varied simulated player behaviour**.

The experiments also show that:

- adaptive policies can outperform simple rule-based control under the tested conditions;
- evaluating only on training-like conditions can miss important generalization behaviour;
- uncertainty can materially affect adaptive-control performance;
- generalization should be evaluated across multiple unseen player configurations rather than assumed from standard evaluation results.

Importantly, the results do **not** establish universal generalization or demonstrate improved real-player enjoyment or flow.

---

# ⚠️ Limitations

This study is simulation-based.

The current evaluation:

- uses **simulated rather than human players**;
- does not directly measure subjective player experience, enjoyment, or flow;
- evaluates generalization over a limited set of held-out player configurations;
- uses a fixed target win rate of **0.70**;
- evaluates PPO as the RL algorithm;
- tests behavioural-noise robustness on a single base player configuration.

Therefore, the findings should be interpreted as evidence from a **controlled simulation study**, rather than as validation of a production-ready human-player DDA system.

---

# 📁 Project Structure

```text
GADDA-DDA-RL/
├── notebooks/
├── src/
├── results/
├── figures/
├── models/
├── .gitignore
└── READ.md
# 🛠️ Technologies

| Category | Technologies |
|---|---|
| **Programming** | Python |
| **Reinforcement Learning** | PPO · Gymnasium · Stable-Baselines3 |
| **Machine Learning & Analysis** | NumPy · Pandas · SciPy · Scikit-learn |
| **Deep Learning** | PyTorch |
| **Visualization** | Matplotlib |
| **Development** | Jupyter · Google Colab |

---

# 📄 Research Status

> **Experimental Research Completed · Manuscript in Preparation**

The experimental study has been completed, including the implementation, evaluation, ablation analysis, robustness testing, statistical analysis, and final figures.

<details>
<summary><strong>🔬 Completed Research Components</strong></summary>

<br>

- ✅ Player behaviour simulation
- ✅ Skill estimation
- ✅ Uncertainty modelling
- ✅ Custom Gymnasium environment
- ✅ Static and rule-based baselines
- ✅ PPO training
- ✅ Standard evaluation
- ✅ Held-out generalization
- ✅ Uncertainty ablation
- ✅ Behavioural-noise robustness
- ✅ Statistical analysis
- ✅ Results and figures

</details>

### 📑 Current Stage

**Manuscript Preparation**

The research is being prepared as an academic manuscript titled:

> ### **GADDA: Generalizable and Uncertainty-Aware Dynamic Difficulty Adjustment using Reinforcement Learning**

---

# 📌 Research Contribution

The contribution of GADDA is **not a new reinforcement learning algorithm**.

Instead, the research investigates a specific **uncertainty-aware adaptive difficulty formulation** and evaluates it through a controlled experimental design combining:

| # | Component |
|---:|---|
| 1 | 🎮 RL-based continuous difficulty control |
| 2 | 🧠 Uncertainty-aware player-state representation |
| 3 | 📊 Standard baseline comparison |
| 4 | 🔄 Held-out generalization testing |
| 5 | 🧪 Matched uncertainty ablation |
| 6 | 📈 Behavioural-noise robustness analysis |

Together, these experiments provide a systematic evaluation of how an adaptive RL controller behaves when simulated player conditions change beyond those encountered during training.

<details>
<summary><strong>🔍 What the experiments investigate</strong></summary>

<br>

**Standard Evaluation**

Compares GADDA PPO against static and rule-based DDA strategies under standard evaluation conditions.

**Held-Out Generalization**

Tests the frozen PPO policy on simulated player configurations whose parameter values fall outside the training ranges.

**Uncertainty Ablation**

Removes uncertainty from the player-state representation while keeping the evaluation design matched, allowing its contribution to target-tracking performance to be examined.

**Behavioural-Noise Robustness**

Evaluates target-tracking behaviour across increasing levels of simulated behavioural noise.

</details>

---

# 👤 Author

### **Aasmeet Kaur**

**B.Tech. Computer Science & Engineering**  
*Undergraduate Researcher*

<p align="left">
  <a href="https://github.com/aasmeetkaur03">
    <img src="https://img.shields.io/badge/GitHub-aasmeetkaur03-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/aasmeetkaur1703">
    <img src="https://img.shields.io/badge/LinkedIn-aasmeetkaur1703-0A66C2?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
</p>

---

<div align="center">

# 🎮 GADDA

### **Generalizable and Uncertainty-Aware Dynamic Difficulty Adjustment using Reinforcement Learning**

*A simulation-based research study on adaptive decision-making, uncertainty, and generalization in reinforcement-learning-based dynamic difficulty adjustment.*

<br>

[![Research](https://img.shields.io/badge/Research-Completed-success?style=flat-square)](#-research-status)
[![RL](https://img.shields.io/badge/Reinforcement-Learning-blue?style=flat-square)](#-technologies)
[![PPO](https://img.shields.io/badge/Algorithm-PPO-orange?style=flat-square)](#-technologies)
[![Python](https://img.shields.io/badge/Language-Python-yellow?style=flat-square&logo=python)](#-technologies)

</div>
