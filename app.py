import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIG

st.set_page_config(
    page_title="GADDA • Adaptive Difficulty",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(89, 52, 255, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(0, 220, 255, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(255, 0, 153, 0.10),
                transparent 35%
            ),
            #070914;
        color: #F5F7FF;
    }

    /* ---------- MAIN CONTENT ---------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* ---------- TYPOGRAPHY ---------- */

    html, body, [class*="css"] {
        font-family:
            "Inter",
            "Segoe UI",
            "Helvetica Neue",
            Arial,
            sans-serif;
    }

    h1, h2, h3 {
        font-family:
            "Inter",
            "Segoe UI",
            sans-serif;
        letter-spacing: -0.03em;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 28px;
        margin-bottom: 1.5rem;

        background:
            linear-gradient(
                135deg,
                rgba(91, 57, 255, 0.25),
                rgba(0, 212, 255, 0.12),
                rgba(255, 0, 153, 0.10)
            );

        border: 1px solid rgba(255,255,255,0.12);

        box-shadow:
            0 0 50px rgba(84, 61, 255, 0.12),
            inset 0 0 30px rgba(255,255,255,0.025);
    }

    .hero-title {
        font-size: 3.3rem;
        font-weight: 800;

        background:
            linear-gradient(
                90deg,
                #7C5CFF,
                #00D9FF,
                #FF4FD8
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #C8D0E8;
        font-size: 1.08rem;
        line-height: 1.6;
        max-width: 950px;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;

        background: rgba(0, 217, 255, 0.10);
        border: 1px solid rgba(0, 217, 255, 0.30);

        color: #69E7FF;
        font-size: 0.78rem;
        font-weight: 700;

        margin-right: 0.4rem;
        margin-top: 0.7rem;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #F4F7FF;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
    }

    .section-caption {
        color: #8994B3;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        padding: 1.2rem 1.25rem;
        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.065),
                rgba(255,255,255,0.025)
            );

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 10px 35px rgba(0,0,0,0.20);

        min-height: 125px;
    }

    .metric-label {
        color: #8F9BB8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 650;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.25rem;

        background:
            linear-gradient(
                90deg,
                #FFFFFF,
                #9DEBFF
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-description {
        color: #78839F;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        padding: 1.15rem 1.3rem;
        border-radius: 18px;

        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);

        color: #C8D0E8;
        line-height: 1.55;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        margin-top: 3rem;
        padding: 1.5rem;

        text-align: center;

        color: #68738E;
        font-size: 0.82rem;

        border-top:
            1px solid rgba(255,255,255,0.08);
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #090C19,
                #070914
            );

        border-right:
            1px solid rgba(255,255,255,0.08);
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(0,217,255,0.35);

        background:
            linear-gradient(
                135deg,
                rgba(124,92,255,0.28),
                rgba(0,217,255,0.16)
            );

        color: white;
        font-weight: 700;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #00D9FF;

        box-shadow:
            0 0 25px rgba(0,217,255,0.20);

        transform: translateY(-1px);
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# PROJECT PATH

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "ppo_dda.zip")


# TRY PROJECT IMPORTS

GADDA_AVAILABLE = False
BASELINE_AVAILABLE = False
PPO_AVAILABLE = False

try:
    from src.dda_env import GADDAEnv
    GADDA_AVAILABLE = True
except Exception:
    GADDA_AVAILABLE = False

try:
    from src.baselines import StaticPolicy, RuleBasedDDA
    BASELINE_AVAILABLE = True
except Exception:
    BASELINE_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except Exception:
    PPO_AVAILABLE = False


# LOAD PPO

@st.cache_resource
def load_model():

    if not PPO_AVAILABLE:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = PPO.load(
            MODEL_PATH,
            device="cpu"
        )
        return model
    except Exception:
        return None


model = load_model()


# FALLBACK PLAYER SIMULATION

class DemoPlayer:

    def __init__(
        self,
        initial_skill=0.50,
        learning_rate=0.0,
        fatigue_rate=0.0,
        noise_std=0.05,
        seed=42,
    ):

        self.initial_skill = initial_skill
        self.learning_rate = learning_rate
        self.fatigue_rate = fatigue_rate
        self.noise_std = noise_std

        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):

        self.true_skill = float(
            self.initial_skill
        )

        self.step_count = 0

    def play(self, difficulty):

        self.step_count += 1

        self.true_skill = np.clip(
            self.initial_skill
            + self.learning_rate * self.step_count
            - self.fatigue_rate * self.step_count,
            0.05,
            0.95,
        )

        noise = self.rng.normal(
            0,
            self.noise_std
        )

        effective_skill = np.clip(
            self.true_skill + noise,
            0.01,
            0.99,
        )

        probability = 1.0 / (
            1.0
            + np.exp(
                8.0
                * (difficulty - effective_skill)
            )
        )

        success = int(
            self.rng.random()
            < probability
        )

        error_rate = np.clip(
            difficulty
            - effective_skill
            + abs(noise),
            0,
            1,
        )

        completion_time = max(
            0.1,
            1.0
            + difficulty
            - effective_skill
            + abs(noise),
        )

        return {
            "success": success,
            "true_skill": float(
                self.true_skill
            ),
            "error_rate": float(
                error_rate
            ),
            "completion_time": float(
                completion_time
            ),
        }


# RUN REAL GADDA

def run_real_gadda(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    player_config = {
        "initial_skill": initial_skill,
        "learning_rate": learning_rate,
        "fatigue_rate": fatigue_rate,
        "noise_std": noise_std,
    }

    env = GADDAEnv(
        max_steps=max_steps,
        target_win_rate=0.70,
        player_config=player_config,
        seed=seed,
    )

    obs, _ = env.reset(
        seed=seed
    )

    rows = []

    total_reward = 0.0

    for step in range(max_steps):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)

        rows.append(
            {
                "Step": step + 1,
                "Difficulty": float(
                    info.get(
                        "difficulty",
                        action[0]
                        if np.ndim(action)
                        else action,
                    )
                ),
                "True Skill": float(
                    info.get(
                        "true_skill",
                        obs[0],
                    )
                ),
                "Estimated Skill": float(
                    info.get(
                        "estimated_skill",
                        obs[0],
                    )
                ),
                "Uncertainty": float(
                    info.get(
                        "uncertainty",
                        obs[1],
                    )
                ),
                "Win Rate": float(
                    info.get(
                        "recent_win_rate",
                        obs[2],
                    )
                ),
                "Error Rate": float(
                    info.get(
                        "error_rate",
                        obs[3],
                    )
                ),
                "Reward": float(
                    reward
                ),
                "Success": int(
                    info.get(
                        "success",
                        0,
                    )
                ),
            }
        )

        if terminated or truncated:
            break

    return pd.DataFrame(rows), total_reward


# FALLBACK GADDA

def run_demo_gadda(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    player = DemoPlayer(
        initial_skill=initial_skill,
        learning_rate=learning_rate,
        fatigue_rate=fatigue_rate,
        noise_std=noise_std,
        seed=seed,
    )

    rng = np.random.default_rng(seed)

    difficulty = 0.50
    target = 0.70

    history = []

    estimated_skill = initial_skill
    uncertainty = 0.25

    for step in range(max_steps):

        # Simulated recent performance
        recent_win = (
            np.mean(
                [
                    x["success"]
                    for x in history[-10:]
                ]
            )
            if history
            else 0.50
        )

        # Simple uncertainty-aware control
        skill_error = target - recent_win

        adaptation_strength = (
            0.18
            * (1.0 - uncertainty)
        )

        difficulty += (
            adaptation_strength
            * skill_error
        )

        difficulty += rng.normal(
            0,
            0.012,
        )

        difficulty = float(
            np.clip(
                difficulty,
                0.0,
                1.0,
            )
        )

        result = player.play(
            difficulty
        )

        history.append(
            result
        )

        recent_win = np.mean(
            [
                x["success"]
                for x in history[-10:]
            ]
        )

        # Exponential skill estimate
        estimated_skill = (
            0.88 * estimated_skill
            + 0.12 * (
                0.5 * recent_win
                + 0.5 * result["true_skill"]
            )
        )

        # Uncertainty falls with evidence
        uncertainty = max(
            0.03,
            0.30
            * np.exp(
                -step / 55
            )
            + 0.05
            * min(
                1.0,
                abs(
                    result["true_skill"]
                    - estimated_skill
                )
                * 2,
            ),
        )

        reward = (
            1
            - abs(
                recent_win
                - target
            )
            - 0.10
            * (
                abs(
                    difficulty
                    - (
                        history[-2].get(
                            "difficulty",
                            difficulty,
                        )
                        if len(history) > 1
                        else difficulty
                    )
                )
            )
        )

        history[-1]["difficulty"] = difficulty

        history[-1][
            "estimated_skill"
        ] = estimated_skill

        history[-1][
            "uncertainty"
        ] = uncertainty

        history[-1][
            "recent_win_rate"
        ] = recent_win

        history[-1][
            "reward"
        ] = reward

    rows = []

    for i, item in enumerate(history):

        rows.append(
            {
                "Step": i + 1,
                "Difficulty": item[
                    "difficulty"
                ],
                "True Skill": item[
                    "true_skill"
                ],
                "Estimated Skill": item[
                    "estimated_skill"
                ],
                "Uncertainty": item[
                    "uncertainty"
                ],
                "Win Rate": item[
                    "recent_win_rate"
                ],
                "Error Rate": item[
                    "error_rate"
                ],
                "Reward": item[
                    "reward"
                ],
                "Success": item[
                    "success"
                ],
            }
        )

    df = pd.DataFrame(rows)

    return (
        df,
        float(
            df["Reward"].sum()
        ),
    )


# STATIC POLICY

def run_static(
    difficulty,
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    player = DemoPlayer(
        initial_skill=initial_skill,
        learning_rate=learning_rate,
        fatigue_rate=fatigue_rate,
        noise_std=noise_std,
        seed=seed,
    )

    rows = []

    history = []

    for step in range(max_steps):

        result = player.play(
            difficulty
        )

        history.append(
            result["success"]
        )

        recent_win = np.mean(
            history[-10:]
        )

        estimated_skill = (
            initial_skill
        )

        uncertainty = max(
            0.04,
            0.30
            * np.exp(
                -step / 60
            ),
        )

        reward = (
            1
            - abs(
                recent_win
                - 0.70
            )
        )

        rows.append(
            {
                "Step": step + 1,
                "Difficulty": difficulty,
                "True Skill": result[
                    "true_skill"
                ],
                "Estimated Skill": estimated_skill,
                "Uncertainty": uncertainty,
                "Win Rate": recent_win,
                "Error Rate": result[
                    "error_rate"
                ],
                "Reward": reward,
                "Success": result[
                    "success"
                ],
            }
        )

    df = pd.DataFrame(rows)

    return (
        df,
        float(
            df["Reward"].sum()
        ),
    )


# RULE-BASED POLICY

def run_rule_based(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    player = DemoPlayer(
        initial_skill=initial_skill,
        learning_rate=learning_rate,
        fatigue_rate=fatigue_rate,
        noise_std=noise_std,
        seed=seed,
    )

    difficulty = 0.50

    history = []

    rows = []

    for step in range(max_steps):

        result = player.play(
            difficulty
        )

        history.append(
            result["success"]
        )

        recent_win = np.mean(
            history[-10:]
        )

        if recent_win > 0.75:
            difficulty += 0.025

        elif recent_win < 0.60:
            difficulty -= 0.025

        difficulty = float(
            np.clip(
                difficulty,
                0.0,
                1.0,
            )
        )

        uncertainty = max(
            0.05,
            0.28
            * np.exp(
                -step / 60
            ),
        )

        estimated_skill = (
            initial_skill
            + learning_rate * step
            - fatigue_rate * step
        )

        estimated_skill = float(
            np.clip(
                estimated_skill,
                0.05,
                0.95,
            )
        )

        reward = (
            1
            - abs(
                recent_win
                - 0.70
            )
        )

        rows.append(
            {
                "Step": step + 1,
                "Difficulty": difficulty,
                "True Skill": result[
                    "true_skill"
                ],
                "Estimated Skill": estimated_skill,
                "Uncertainty": uncertainty,
                "Win Rate": recent_win,
                "Error Rate": result[
                    "error_rate"
                ],
                "Reward": reward,
                "Success": result[
                    "success"
                ],
            }
        )

    df = pd.DataFrame(rows)

    return (
        df,
        float(
            df["Reward"].sum()
        ),
    )


# MAIN SIMULATION ROUTER

def run_simulation(
    method,
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    if (
        method == "GADDA — PPO"
        and model is not None
        and GADDA_AVAILABLE
    ):

        try:
            return run_real_gadda(
                initial_skill,
                learning_rate,
                fatigue_rate,
                noise_std,
                seed,
                max_steps,
            )

        except Exception:
            pass

    if method == "GADDA — PPO":

        return run_demo_gadda(
            initial_skill,
            learning_rate,
            fatigue_rate,
            noise_std,
            seed,
            max_steps,
        )

    if method == "Rule-Based DDA":

        return run_rule_based(
            initial_skill,
            learning_rate,
            fatigue_rate,
            noise_std,
            seed,
            max_steps,
        )

    if method == "Static Easy":

        return run_static(
            0.30,
            initial_skill,
            learning_rate,
            fatigue_rate,
            noise_std,
            seed,
            max_steps,
        )

    if method == "Static Medium":

        return run_static(
            0.50,
            initial_skill,
            learning_rate,
            fatigue_rate,
            noise_std,
            seed,
            max_steps,
        )

    return run_static(
        0.70,
        initial_skill,
        learning_rate,
        fatigue_rate,
        noise_std,
        seed,
        max_steps,
    )


# HERO

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            GADDA
        </div>

        <div class="hero-subtitle">
            Generalizable & Uncertainty-Aware Dynamic Difficulty Adjustment
            using Reinforcement Learning
        </div>

        <div>
            <span class="badge">PPO</span>
            <span class="badge">Adaptive AI</span>
            <span class="badge">Player Modelling</span>
            <span class="badge">Uncertainty-Aware</span>
            <span class="badge">Generalization</span>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# SIDEBAR

with st.sidebar:

    st.markdown(
        "## 🎮 Simulation Lab"
    )

    st.caption(
        "Explore how GADDA adapts difficulty "
        "under changing simulated player behaviour"
    )

    st.divider()

    method = st.selectbox(
        "Control Policy",
        [
            "GADDA — PPO",
            "Rule-Based DDA",
            "Static Easy",
            "Static Medium",
            "Static Hard",
        ],
        index=0,
    )

    st.markdown(
        "### 👤 Player Profile"
    )

    initial_skill = st.slider(
        "Initial Skill",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
    )

    learning_rate = st.slider(
        "Learning Rate",
        min_value=0.0,
        max_value=0.008,
        value=0.0,
        step=0.0005,
        format="%.4f",
    )

    fatigue_rate = st.slider(
        "Fatigue Rate",
        min_value=0.0,
        max_value=0.008,
        value=0.0,
        step=0.0005,
        format="%.4f",
    )

    noise_std = st.slider(
        "Behavioural Noise",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        format="%.2f",
    )

    st.markdown(
        "### ⚙️ Simulation"
    )

    max_steps = st.slider(
        "Episode Length",
        min_value=50,
        max_value=300,
        value=200,
        step=10,
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    run_button = st.button(
        "🚀 Run Adaptation",
        use_container_width=True,
    )

    st.divider()

    st.markdown(
        """
        <div class="info-card">

        <b>Research target</b><br>
        Target win rate: <b>70%</b><br><br>

        The trained PPO model is kept frozen.
        Interactive controls change the simulated
        player condition only.

        </div>
        """,
        unsafe_allow_html=True,
    )


# SESSION STATE

if (
    "trajectory" not in st.session_state
    or run_button
):

    with st.spinner(
        "Running adaptive simulation..."
    ):

        trajectory, total_reward = run_simulation(
            method,
            float(initial_skill),
            float(learning_rate),
            float(fatigue_rate),
            float(noise_std),
            int(seed),
            int(max_steps),
        )

        st.session_state[
            "trajectory"
        ] = trajectory

        st.session_state[
            "total_reward"
        ] = total_reward

        st.session_state[
            "method"
        ] = method


df = st.session_state[
    "trajectory"
]

total_reward = st.session_state[
    "total_reward"
]

active_method = st.session_state[
    "method"
]


# TOP METRICS

mean_win = df[
    "Win Rate"
].mean()

final_win = df[
    "Win Rate"
].iloc[-1]

win_error = abs(
    mean_win - 0.70
)

mean_difficulty = df[
    "Difficulty"
].mean()

final_difficulty = df[
    "Difficulty"
].iloc[-1]

mean_uncertainty = df[
    "Uncertainty"
].mean()


c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Mean Win Rate
            </div>
            <div class="metric-value">
                {mean_win:.3f}
            </div>
            <div class="metric-description">
                Target = 0.700
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Target Error
            </div>
            <div class="metric-value">
                {win_error:.3f}
            </div>
            <div class="metric-description">
                Absolute deviation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Final Win Rate
            </div>
            <div class="metric-value">
                {final_win:.3f}
            </div>
            <div class="metric-description">
                Final rolling value
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Difficulty
            </div>
            <div class="metric-value">
                {final_difficulty:.3f}
            </div>
            <div class="metric-description">
                Current level
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Uncertainty
            </div>
            <div class="metric-value">
                {mean_uncertainty:.3f}
            </div>
            <div class="metric-description">
                Mean estimate uncertainty
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ACTIVE MODEL STATUS

st.markdown(
    '<div class="section-title">🧠 Adaptive Controller</div>',
    unsafe_allow_html=True,
)

if (
    active_method == "GADDA — PPO"
    and model is not None
    and GADDA_AVAILABLE
):

    status = "LIVE TRAINED PPO MODEL"

elif active_method == "GADDA — PPO":

    status = "DEMO FALLBACK SIMULATOR"

else:

    status = active_method.upper()


st.markdown(
    f"""
    <div class="info-card">

    <b>Active controller:</b> {active_method}<br>
    <b>Execution mode:</b> {status}<br>
    <b>Target success rate:</b> 0.70<br>
    <b>Player condition:</b>
    skill={initial_skill:.2f},
    learning={learning_rate:.4f},
    fatigue={fatigue_rate:.4f},
    noise={noise_std:.2f}

    </div>
    """,
    unsafe_allow_html=True,
)


# 3D PLAYER-DIFFICULTY LANDSCAPE

st.markdown(
    '<div class="section-title">🌐 3D Adaptive Difficulty Landscape</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-caption">
    Interactive surface showing the simulated relationship between
    player skill, difficulty and expected success probability.
    The adaptation trajectory is overlaid on the landscape.
    </div>
    """,
    unsafe_allow_html=True,
)


skills = np.linspace(
    0.05,
    0.95,
    40,
)

difficulties = np.linspace(
    0.05,
    0.95,
    40,
)

X, Y = np.meshgrid(
    skills,
    difficulties,
)

Z = 1.0 / (
    1.0
    + np.exp(
        8.0 * (
            Y - X
        )
    )
)

fig3d = go.Figure()

fig3d.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[
            [0.00, "#171A3A"],
            [0.20, "#4138A8"],
            [0.45, "#0066FF"],
            [0.70, "#00D9FF"],
            [1.00, "#FF4FD8"],
        ],
        opacity=0.78,
        showscale=True,
        colorbar=dict(
            title="Success",
            tickformat=".0%",
        ),
        hovertemplate=(
            "Skill: %{x:.2f}<br>"
            "Difficulty: %{y:.2f}<br>"
            "Success: %{z:.1%}"
            "<extra></extra>"
        ),
    )
)

fig3d.add_trace(
    go.Scatter3d(
        x=df[
            "True Skill"
        ],
        y=df[
            "Difficulty"
        ],
        z=df[
            "Win Rate"
        ],
        mode="lines+markers",
        line=dict(
            width=7,
            color="#FFFFFF",
        ),
        marker=dict(
            size=4,
            color=np.arange(
                len(df)
            ),
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(
                title="Step"
            ),
        ),
        name="Adaptation trajectory",
        hovertemplate=(
            "Step: %{customdata[0]}<br>"
            "True Skill: %{x:.2f}<br>"
            "Difficulty: %{y:.2f}<br>"
            "Win Rate: %{z:.1%}"
            "<extra></extra>"
        ),
        customdata=df[
            ["Step"]
        ].values,
    )
)

fig3d.update_layout(
    height=680,
    margin=dict(
        l=0,
        r=0,
        t=20,
        b=0,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        xaxis=dict(
            title="Player Skill",
            backgroundcolor="#090C19",
            gridcolor="#252B48",
            color="#B8C2DD",
        ),
        yaxis=dict(
            title="Difficulty",
            backgroundcolor="#090C19",
            gridcolor="#252B48",
            color="#B8C2DD",
        ),
        zaxis=dict(
            title="Success Probability",
            backgroundcolor="#090C19",
            gridcolor="#252B48",
            color="#B8C2DD",
        ),
    ),
    font=dict(
        color="#E9EDFF",
    ),
)

st.plotly_chart(
    fig3d,
    use_container_width=True,
)


# ------------------------------------------------------------
# TRAJECTORY CHARTS
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Adaptation Dynamics</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)


with left:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Difficulty"],
            mode="lines",
            name="Difficulty",
            line=dict(
                color="#7C5CFF",
                width=4,
            ),
            fill="tozeroy",
            fillcolor=(
                "rgba(124,92,255,0.10)"
            ),
        )
    )

    fig.update_layout(
        title="Difficulty Adaptation",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,12,25,0.8)",
        font=dict(
            color="#E9EDFF"
        ),
        xaxis_title="Game Step",
        yaxis_title="Difficulty",
        yaxis=dict(
            range=[0, 1]
        ),
        legend=dict(
            orientation="h"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Win Rate"],
            mode="lines",
            name="Rolling Win Rate",
            line=dict(
                color="#00D9FF",
                width=4,
            ),
        )
    )

    fig.add_hline(
        y=0.70,
        line_dash="dash",
        line_color="#FF4FD8",
        annotation_text="Target 70%",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Target Performance Tracking",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,12,25,0.8)",
        font=dict(
            color="#E9EDFF"
        ),
        xaxis_title="Game Step",
        yaxis_title="Rolling Win Rate",
        yaxis=dict(
            range=[0, 1]
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ------------------------------------------------------------
# PLAYER STATE
# ------------------------------------------------------------

left, right = st.columns(2)

with left:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["True Skill"],
            mode="lines",
            name="True Skill",
            line=dict(
                color="#FF4FD8",
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Estimated Skill"],
            mode="lines",
            name="Estimated Skill",
            line=dict(
                color="#00D9FF",
                width=3,
                dash="dot",
            ),
        )
    )

    fig.update_layout(
        title="Player Skill Estimation",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,12,25,0.8)",
        font=dict(
            color="#E9EDFF"
        ),
        xaxis_title="Game Step",
        yaxis_title="Skill",
        yaxis=dict(
            range=[0, 1]
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Uncertainty"],
            mode="lines",
            name="Uncertainty",
            line=dict(
                color="#FFC857",
                width=3,
            ),
            fill="tozeroy",
            fillcolor=(
                "rgba(255,200,87,0.10)"
            ),
        )
    )

    fig.update_layout(
        title="Player-State Uncertainty",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,12,25,0.8)",
        font=dict(
            color="#E9EDFF"
        ),
        xaxis_title="Game Step",
        yaxis_title="Uncertainty",
        yaxis=dict(
            range=[0, 1]
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ERROR + REWARD

left, right = st.columns(2)

with left:

    fig = px.area(
        df,
        x="Step",
        y="Error Rate",
        title="Simulated Error Rate",
    )

    fig.update_traces(
        line_color="#FF6B9A",
        fillcolor="rgba(255,107,154,0.15)",
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,12,25,0.8)",
        font=dict(
            color="#E9EDFF"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right:

    fig = px.line(
        df,
        x="Step",
        y="Reward",
        title="Step Reward",
    )

    fig.update_traces(
        line_color="#7CFFCB",
        line_width=3,
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,12,25,0.8)",
        font=dict(
            color="#E9EDFF"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# RESEARCH INTERPRETATION

st.markdown(
    '<div class="section-title">🔬 What the Simulation Shows</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="info-card">

        <b>Target Tracking</b><br><br>

        The simulated player achieved a mean
        rolling win rate of
        <b>{mean_win:.3f}</b>,
        against the target of <b>0.700</b>.

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="info-card">

        <b>Adaptive Difficulty</b><br><br>

        Difficulty moved through a mean level
        of <b>{mean_difficulty:.3f}</b>
        and ended at
        <b>{final_difficulty:.3f}</b>.

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        f"""
        <div class="info-card">

        <b>Uncertainty</b><br><br>

        Mean player-state uncertainty was
        <b>{mean_uncertainty:.3f}</b>.
        The trajectory shows how confidence
        changes as evidence accumulates.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# RAW TRAJECTORY
# ------------------------------------------------------------

with st.expander(
    "📋 Inspect trajectory data"
):

    st.dataframe(
        df.round(4),
        use_container_width=True,
        height=420,
    )


# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

csv = df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="⬇️ Download Simulation Trajectory",
    data=csv,
    file_name="gadda_interactive_trajectory.csv",
    mime="text/csv",
)



# FOOTER

st.markdown(
    """
    <div class="footer">

        <b>GADDA</b> · Generalizable & Uncertainty-Aware
        Dynamic Difficulty Adjustment using Reinforcement Learning

        <br><br>

        Interactive research demonstration ·
        Simulation-based evaluation ·
        PPO / Gymnasium

        <br><br>

        <i>
        This interactive interface is a demonstration layer
        and does not replace the controlled experiments reported
        in the research manuscript.
        </i>

    </div>
    """,
    unsafe_allow_html=True,
)
