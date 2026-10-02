import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

try:
    from src.dda_env import GADDAEnv
    from src.baselines import RuleBasedDDA, StaticPolicy
except Exception:
    GADDAEnv = None
    RuleBasedDDA = None
    StaticPolicy = None

st.set_page_config(
    page_title="GADDA • Adaptive Difficulty Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(98,230,255,.12), transparent 28%),
    radial-gradient(circle at 88% 10%, rgba(167,139,250,.14), transparent 30%),
    radial-gradient(circle at 55% 95%, rgba(255,120,200,.10), transparent 30%),
    #070816;
  color: #f4f6ff;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.hero {
  padding: 1.2rem 1.4rem 1.3rem;
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(18,25,55,.82), rgba(13,16,37,.66));
  box-shadow: 0 18px 60px rgba(0,0,0,.28);
}
.hero h1 {
  margin: 0;
  font-size: clamp(2rem, 5vw, 3.7rem);
  letter-spacing: -0.045em;
  line-height: 1;
}
.hero .eyebrow {
  color: #62e6ff;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  font-size: .76rem;
}
.hero p { color: #aeb7d5; max-width: 820px; margin-bottom: 0; }
.badge {
  display:inline-block; padding:.32rem .65rem; border-radius:999px;
  border:1px solid rgba(98,230,255,.25); background:rgba(98,230,255,.08);
  color:#bdf5ff; font-size:.78rem; margin:.25rem .3rem 0 0;
}
.metric-card {
  padding: .85rem 1rem; border-radius: 16px; border: 1px solid rgba(255,255,255,.10);
  background: rgba(15,19,40,.66);
}
.metric-label { color: #aeb7d5; font-size: .76rem; }
.metric-value { color: #f4f6ff; font-size: 1.45rem; font-weight: 800; }
.section-title { font-size: 1.25rem; font-weight: 800; margin: 1rem 0 .4rem; }
.small-note { color: #aeb7d5; font-size: .82rem; }
div[data-testid="stSidebar"] {
  background: rgba(8,10,25,.92);
  border-right: 1px solid rgba(255,255,255,.10);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">GADDA • Research Demonstration</div>
  <h1>Adaptive difficulty, in motion.</h1>
  <p>
    Explore how the learned policy responds to a simulated player's changing
    skill, uncertainty, performance and behavioural noise. The 3D view exposes
    the adaptation trajectory rather than hiding it behind a single score.
  </p>
  <span class="badge">PPO</span>
  <span class="badge">Uncertainty-aware</span>
  <span class="badge">Held-out profiles</span>
  <span class="badge">Interactive simulation</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ✦ Experiment controls")
method = st.sidebar.selectbox(
    "Policy",
    ["GADDA • PPO", "Rule-Based DDA", "Static Easy", "Static Medium", "Static Hard"],
)
profile = st.sidebar.selectbox(
    "Player profile",
    [
        "Balanced", "Learning", "Fatigue", "Noisy",
        "Unseen Low Skill", "Unseen High Skill",
        "Unseen Fast Learning", "Unseen Strong Fatigue", "Unseen High Noise",
    ],
)
target = st.sidebar.slider("Target win rate", 0.50, 0.90, 0.70, 0.01)
steps = st.sidebar.slider("Simulation steps", 50, 300, 200, 10)
noise = st.sidebar.slider("Behavioural noise", 0.01, 0.30, 0.05, 0.01)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

PROFILE_CONFIGS = {
    "Balanced": {"initial_skill": 0.50, "learning_rate": 0.0, "fatigue_rate": 0.0, "noise_std": noise},
    "Learning": {"initial_skill": 0.35, "learning_rate": 0.003, "fatigue_rate": 0.0, "noise_std": noise},
    "Fatigue": {"initial_skill": 0.75, "learning_rate": 0.0, "fatigue_rate": 0.002, "noise_std": noise},
    "Noisy": {"initial_skill": 0.50, "learning_rate": 0.0, "fatigue_rate": 0.0, "noise_std": max(noise, 0.12)},
    "Unseen Low Skill": {"initial_skill": 0.15, "learning_rate": 0.0, "fatigue_rate": 0.0, "noise_std": noise},
    "Unseen High Skill": {"initial_skill": 0.90, "learning_rate": 0.0, "fatigue_rate": 0.0, "noise_std": noise},
    "Unseen Fast Learning": {"initial_skill": 0.20, "learning_rate": 0.006, "fatigue_rate": 0.0, "noise_std": noise},
    "Unseen Strong Fatigue": {"initial_skill": 0.90, "learning_rate": 0.0, "fatigue_rate": 0.005, "noise_std": noise},
    "Unseen High Noise": {"initial_skill": 0.30, "learning_rate": 0.0, "fatigue_rate": 0.0, "noise_std": 0.20},
}

def make_env(config, seed_value):
    if GADDAEnv is None:
        raise RuntimeError("GADDAEnv could not be imported. Run this app from the project root.")
    try:
        return GADDAEnv(
            target_win_rate=target,
            max_steps=steps,
            player_config=config,
            seed=int(seed_value),
        )
    except TypeError:
        return GADDAEnv(
            target_win_rate=target,
            max_steps=steps,
            player_config=config,
        )

@st.cache_resource
def load_model():
    if PPO is None:
        return None
    model_path = Path("models/ppo_dda.zip")
    if not model_path.exists():
        return None
    return PPO.load(str(model_path), device="cpu")

def run_simulation():
    config = PROFILE_CONFIGS[profile]
    env = make_env(config, seed)
    try:
        obs, _ = env.reset(seed=int(seed))
    except TypeError:
        obs, _ = env.reset()

    if method == "GADDA • PPO":
        policy = load_model()
        if policy is None:
            raise RuntimeError("models/ppo_dda.zip was not found. Add the trained model to the repository.")
    elif method == "Rule-Based DDA":
        policy = RuleBasedDDA(target_win_rate=target)
        policy.reset()
    else:
        fixed = {"Static Easy": 0.30, "Static Medium": 0.50, "Static Hard": 0.70}[method]
        policy = StaticPolicy(difficulty=fixed)
        policy.reset()

    rows = []
    total_reward = 0.0

    for step in range(steps):
        if method == "GADDA • PPO":
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        rows.append({
            "Step": step + 1,
            "Difficulty": float(info.get("difficulty", np.asarray(action).reshape(-1)[0])),
            "Estimated Skill": float(info.get("estimated_skill", obs[0])),
            "Uncertainty": float(info.get("uncertainty", obs[1])),
            "Win Rate": float(info.get("recent_win_rate", obs[2])),
            "Error Rate": float(info.get("error_rate", 0.0)),
            "True Skill": float(info.get("true_skill", np.nan)),
            "Reward": float(reward),
            "Success": int(info.get("success", 0)),
        })

        if terminated or truncated:
            break

    return pd.DataFrame(rows), total_reward

def metric(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

if st.button("▶ Run adaptive simulation", type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating player–agent interaction..."):
            data, total_reward = run_simulation()
        st.session_state["gadda_data"] = data
        st.session_state["gadda_reward"] = total_reward
    except Exception as exc:
        st.error(str(exc))

if "gadda_data" not in st.session_state:
    st.info("Choose a player profile and policy, then run the simulation.")
    st.stop()

data = st.session_state["gadda_data"]
total_reward = st.session_state["gadda_reward"]
error = float(abs(data["Win Rate"].mean() - target))
final_wr = float(data["Win Rate"].iloc[-1])
mean_diff = float(data["Difficulty"].mean())
mean_unc = float(data["Uncertainty"].mean())

cols = st.columns(5)
for col, label, value in zip(
    cols,
    ["Target error", "Final rolling WR", "Mean difficulty", "Mean uncertainty", "Episode reward"],
    [f"{error:.3f}", f"{final_wr:.2f}", f"{mean_diff:.2f}", f"{mean_unc:.3f}", f"{total_reward:.1f}"],
):
    with col:
        metric(label, value)

st.markdown('<div class="section-title">✦ 3D adaptation trajectory</div>', unsafe_allow_html=True)
st.caption(
    "X = estimated skill · Y = uncertainty · Z = difficulty · marker colour = rolling win rate. "
    "Rotate, zoom and hover to inspect the trajectory."
)

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=data["Estimated Skill"], y=data["Uncertainty"], z=data["Difficulty"],
    mode="lines",
    line=dict(width=7, color="#62e6ff"),
    name="Adaptation path",
    hovertemplate=(
        "Step %{customdata[0]}<br>Estimated skill %{x:.3f}<br>"
        "Uncertainty %{y:.3f}<br>Difficulty %{z:.3f}<br>"
        "Win rate %{customdata[1]:.2f}<extra></extra>"
    ),
    customdata=np.column_stack([data["Step"], data["Win Rate"]]),
))
fig3d.add_trace(go.Scatter3d(
    x=data["Estimated Skill"], y=data["Uncertainty"], z=data["Difficulty"],
    mode="markers",
    marker=dict(
        size=4.5,
        color=data["Win Rate"],
        colorscale=[
            [0.0, "#ff4d8d"], [0.45, "#ffd166"],
            [0.70, "#62e6ff"], [1.0, "#a78bfa"],
        ],
        cmin=0, cmax=1, opacity=.85,
        colorbar=dict(title="Win rate"),
    ),
    name="Player state",
    customdata=np.column_stack([data["Step"], data["Win Rate"], data["Error Rate"]]),
    hovertemplate=(
        "Step %{customdata[0]}<br>Win rate %{customdata[1]:.2f}<br>"
        "Error rate %{customdata[2]:.2f}<br>Skill %{x:.3f}<br>"
        "Uncertainty %{y:.3f}<br>Difficulty %{z:.3f}<extra></extra>"
    ),
))
fig3d.update_layout(
    height=650,
    margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#eef2ff"),
    scene=dict(
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Estimated Skill", gridcolor="rgba(255,255,255,.10)"),
        yaxis=dict(title="Uncertainty", gridcolor="rgba(255,255,255,.10)"),
        zaxis=dict(title="Difficulty", range=[0, 1], gridcolor="rgba(255,255,255,.10)"),
        camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
    ),
    showlegend=False,
)
st.plotly_chart(fig3d, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = go.Figure(go.Scatter(
        x=data["Step"], y=data["Win Rate"], mode="lines",
        line=dict(width=3, color="#62e6ff"), name="Rolling win rate"
    ))
    fig.add_hline(y=target, line_dash="dash", line_color="#ffd166",
                  annotation_text=f"Target {target:.2f}")
    fig.update_layout(
        title="Target tracking", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eef2ff"), margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Step", yaxis_title="Win rate", yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = go.Figure(go.Scatter(
        x=data["Step"], y=data["Difficulty"], mode="lines",
        line=dict(width=3, color="#ff78c8"), name="Difficulty"
    ))
    fig.update_layout(
        title="Difficulty adaptation", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eef2ff"), margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Step", yaxis_title="Difficulty", yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">Research telemetry</div>', unsafe_allow_html=True)
st.dataframe(
    data[["Step", "Estimated Skill", "Uncertainty", "Win Rate", "Error Rate", "Difficulty", "Reward"]].tail(20),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    '<div class="small-note">Research note: this is a simulation-based demonstration '
    'of the GADDA environment and frozen PPO policy. It does not represent a human-player '
    'study or a claim of universal player generalization.</div>',
    unsafe_allow_html=True,
)
