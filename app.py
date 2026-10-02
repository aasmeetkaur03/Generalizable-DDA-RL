# ============================================================
# GADDA
# Generalizable & Uncertainty-Aware Dynamic Difficulty Adjustment
# Interactive Research Demonstration
# ============================================================

import os
import sys
import inspect
import importlib
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GADDA | Adaptive Difficulty",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    dedent(
        """
        <style>

        /* ==================================================
           BASE
           ================================================== */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 0%,
                    rgba(166, 180, 255, 0.12),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 95% 20%,
                    rgba(180, 225, 216, 0.10),
                    transparent 28%
                ),
                #0B0F19;

            color: #F5F7FA;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        html, body, [class*="css"] {
            font-family:
                Inter,
                "Segoe UI",
                Helvetica,
                Arial,
                sans-serif;
        }

        /* ==================================================
           SIDEBAR
           ================================================== */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #0D1220 0%,
                    #090D17 100%
                );

            border-right:
                1px solid rgba(255,255,255,0.07);
        }

        /* ==================================================
           HERO
           ================================================== */

        .hero {
            padding: 2.6rem 2.8rem;
            border-radius: 26px;
            margin-bottom: 1.6rem;

            background:
                linear-gradient(
                    135deg,
                    rgba(154, 165, 255, 0.12),
                    rgba(178, 222, 213, 0.07)
                );

            border:
                1px solid rgba(255,255,255,0.10);

            box-shadow:
                0 24px 70px rgba(0,0,0,0.24);
        }

        .hero-kicker {
            color: #AAB5FF;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            color: #F8FAFC;
            font-size: 3.5rem;
            line-height: 1.0;
            font-weight: 800;
            letter-spacing: -0.055em;
            margin-bottom: 0.75rem;
        }

        .hero-subtitle {
            color: #C9D0DC;
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 920px;
        }

        .hero-tags {
            margin-top: 1.2rem;
        }

        .tag {
            display: inline-block;

            padding:
                0.38rem 0.72rem;

            margin:
                0.18rem 0.25rem 0.18rem 0;

            border-radius: 999px;

            color: #DDE3EE;

            background:
                rgba(255,255,255,0.045);

            border:
                1px solid rgba(255,255,255,0.10);

            font-size: 0.76rem;
            font-weight: 600;
        }

        /* ==================================================
           SECTION HEADINGS
           ================================================== */

        .section-title {
            color: #F5F7FA;
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -0.025em;
            margin-top: 1.8rem;
            margin-bottom: 0.25rem;
        }

        .section-subtitle {
            color: #8993A5;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }

        /* ==================================================
           CARDS
           ================================================== */

        .glass-card {
            padding: 1.25rem 1.35rem;
            border-radius: 20px;

            background:
                rgba(255,255,255,0.035);

            border:
                1px solid rgba(255,255,255,0.075);

            box-shadow:
                0 14px 40px rgba(0,0,0,0.18);
        }

        .metric-card {
            padding: 1.15rem 1.2rem;
            min-height: 120px;

            border-radius: 18px;

            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.055),
                    rgba(255,255,255,0.018)
                );

            border:
                1px solid rgba(255,255,255,0.075);
        }

        .metric-label {
            color: #8993A5;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        .metric-value {
            color: #F5F7FA;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-top: 0.25rem;
        }

        .metric-note {
            color: #6F7A8D;
            font-size: 0.72rem;
            margin-top: 0.15rem;
        }

        /* ==================================================
           STATUS
           ================================================== */

        .status {
            display: flex;
            align-items: center;
            gap: 0.55rem;

            color: #DCE5E2;
            font-size: 0.82rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #A9D8CB;
            box-shadow:
                0 0 12px rgba(169,216,203,0.45);
        }

        /* ==================================================
           BUTTON
           ================================================== */

        .stButton > button {
            width: 100%;

            border-radius: 12px;

            background:
                #DCE4FF;

            color:
                #111827;

            border:
                none;

            font-weight: 750;

            padding:
                0.65rem 1rem;

            transition:
                all 0.18s ease;
        }

        .stButton > button:hover {
            background:
                #EEF1FF;

            transform:
                translateY(-1px);

            box-shadow:
                0 10px 30px rgba(220,228,255,0.16);
        }

        /* ==================================================
           FOOTER
           ================================================== */

        .footer {
            margin-top: 3.5rem;
            padding-top: 1.4rem;

            border-top:
                1px solid rgba(255,255,255,0.07);

            color: #657084;
            text-align: center;
            font-size: 0.76rem;
            line-height: 1.7;
        }

        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "ppo_dda.zip",
)


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from stable_baselines3 import PPO

    SB3_AVAILABLE = True

except Exception:
    PPO = None
    SB3_AVAILABLE = False


try:
    import gymnasium as gym

    GYM_AVAILABLE = True

except Exception:
    gym = None
    GYM_AVAILABLE = False


# ============================================================
# LOAD PPO MODEL
# ============================================================

@st.cache_resource
def load_ppo_model():

    if not SB3_AVAILABLE:
        return None

    if not os.path.exists(
        MODEL_PATH
    ):
        return None

    try:

        return PPO.load(
            MODEL_PATH,
            device="cpu",
        )

    except Exception:

        return None


ppo_model = load_ppo_model()


# ============================================================
# DISCOVER ENVIRONMENT CLASS
# ============================================================

@st.cache_resource
def load_environment_class():

    try:

        module = importlib.import_module(
            "src.environment"
        )

    except Exception:
        return None

    candidates = []

    for name in dir(module):

        obj = getattr(
            module,
            name,
        )

        if not inspect.isclass(obj):
            continue

        try:

            if (
                GYM_AVAILABLE
                and issubclass(
                    obj,
                    gym.Env,
                )
            ):
                candidates.append(obj)

        except Exception:
            continue

    # Prefer names containing GADDA.
    for cls in candidates:

        if "gadda" in cls.__name__.lower():

            return cls

    # Otherwise prefer environment-like class names.
    for cls in candidates:

        name = cls.__name__.lower()

        if (
            "environment" in name
            or "difficulty" in name
            or "dda" in name
        ):

            return cls

    if candidates:
        return candidates[0]

    return None


ENV_CLASS = load_environment_class()


# ============================================================
# REAL ENVIRONMENT CONSTRUCTOR
# ============================================================

def build_real_environment(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    if ENV_CLASS is None:
        return None

    try:

        signature = inspect.signature(
            ENV_CLASS.__init__
        )

        parameters = signature.parameters

        kwargs = {}

        if (
            "target_win_rate"
            in parameters
        ):
            kwargs[
                "target_win_rate"
            ] = 0.70

        if (
            "max_steps"
            in parameters
        ):
            kwargs[
                "max_steps"
            ] = int(max_steps)

        if (
            "seed"
            in parameters
        ):
            kwargs[
                "seed"
            ] = int(seed)

        player_config = {
            "initial_skill":
                float(initial_skill),

            "learning_rate":
                float(learning_rate),

            "fatigue_rate":
                float(fatigue_rate),

            "noise_std":
                float(noise_std),
        }

        if (
            "player_config"
            in parameters
        ):
            kwargs[
                "player_config"
            ] = player_config

        env = ENV_CLASS(
            **kwargs
        )

        return env

    except Exception:

        return None


# ============================================================
# REAL PPO RUNNER
# ============================================================

def run_real_model(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    env = build_real_environment(
        initial_skill,
        learning_rate,
        fatigue_rate,
        noise_std,
        seed,
        max_steps,
    )

    if (
        env is None
        or ppo_model is None
    ):
        return None

    try:

        obs, info = env.reset(
            seed=int(seed)
        )

        records = []

        total_reward = 0.0

        previous_difficulty = 0.50

        for step in range(
            int(max_steps)
        ):

            action, _ = (
                ppo_model.predict(
                    obs,
                    deterministic=True,
                )
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(
                reward
            )

            # -------------------------
            # Robust information lookup
            # -------------------------

            difficulty = info.get(
                "difficulty",
                float(
                    np.asarray(
                        action
                    ).reshape(-1)[0]
                ),
            )

            true_skill = info.get(
                "true_skill",
                np.nan,
            )

            estimated_skill = info.get(
                "estimated_skill",
                float(
                    obs[0]
                )
                if len(obs) > 0
                else np.nan,
            )

            uncertainty = info.get(
                "uncertainty",
                float(
                    obs[1]
                )
                if len(obs) > 1
                else np.nan,
            )

            win_rate = info.get(
                "recent_win_rate",
                float(
                    obs[2]
                )
                if len(obs) > 2
                else np.nan,
            )

            error_rate = info.get(
                "error_rate",
                float(
                    obs[3]
                )
                if len(obs) > 3
                else np.nan,
            )

            success = info.get(
                "success",
                np.nan,
            )

            records.append(
                {
                    "Step":
                        step + 1,

                    "Difficulty":
                        float(
                            difficulty
                        ),

                    "True Skill":
                        float(
                            true_skill
                        )
                        if not pd.isna(
                            true_skill
                        )
                        else np.nan,

                    "Estimated Skill":
                        float(
                            estimated_skill
                        ),

                    "Uncertainty":
                        float(
                            uncertainty
                        ),

                    "Win Rate":
                        float(
                            win_rate
                        ),

                    "Error Rate":
                        float(
                            error_rate
                        ),

                    "Reward":
                        float(
                            reward
                        ),

                    "Success":
                        float(
                            success
                        )
                        if not pd.isna(
                            success
                        )
                        else np.nan,

                    "Difficulty Change":
                        abs(
                            float(
                                difficulty
                            )
                            - previous_difficulty
                        ),
                }
            )

            previous_difficulty = float(
                difficulty
            )

            if (
                terminated
                or truncated
            ):
                break

        return (
            pd.DataFrame(
                records
            ),
            total_reward,
            True,
        )

    except Exception:

        return None


# ============================================================
# VISUAL FALLBACK
#
# IMPORTANT:
# This is only used if the local GADDA environment/model
# cannot be loaded. It is clearly labelled in the UI.
# ============================================================

def run_visual_demo(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    rng = np.random.default_rng(
        int(seed)
    )

    target = 0.70

    true_skill = float(
        initial_skill
    )

    difficulty = 0.50

    estimated_skill = true_skill

    uncertainty = 0.28

    history = []

    rows = []

    for step in range(
        int(max_steps)
    ):

        # -------------------------
        # Simulated skill dynamics
        # -------------------------

        true_skill = np.clip(
            true_skill
            + learning_rate
            - fatigue_rate
            + rng.normal(
                0,
                noise_std * 0.04,
            ),
            0.03,
            0.97,
        )

        # -------------------------
        # Success probability
        # -------------------------

        probability = (
            1.0
            /
            (
                1.0
                +
                np.exp(
                    8.0
                    *
                    (
                        difficulty
                        - true_skill
                    )
                )
            )
        )

        success = int(
            rng.random()
            < probability
        )

        history.append(
            success
        )

        recent_win = np.mean(
            history[-10:]
        )

        # -------------------------
        # Estimate skill
        # -------------------------

        estimated_skill = (
            0.88
            * estimated_skill
            +
            0.12
            * recent_win
        )

        # -------------------------
        # Uncertainty
        # -------------------------

        uncertainty = np.clip(
            0.28
            * np.exp(
                -step / 65
            )
            +
            abs(
                true_skill
                - estimated_skill
            )
            * 0.35
            +
            noise_std
            * 0.20,
            0.025,
            0.60,
        )

        # -------------------------
        # Adaptive controller
        # -------------------------

        direction = (
            recent_win
            - target
        )

        adaptation = (
            0.045
            * direction
            *
            (
                1.0
                - uncertainty
            )
        )

        difficulty += adaptation

        difficulty += rng.normal(
            0,
            0.008,
        )

        difficulty = float(
            np.clip(
                difficulty,
                0.02,
                0.98,
            )
        )

        error_rate = np.clip(
            difficulty
            - true_skill
            + abs(
                rng.normal(
                    0,
                    noise_std * 0.20,
                )
            ),
            0,
            1,
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
                    adaptation
                )
            )
            + 0.20
            * (
                1
                - error_rate
            )
            - 0.10
            * uncertainty
        )

        rows.append(
            {
                "Step":
                    step + 1,

                "Difficulty":
                    difficulty,

                "True Skill":
                    true_skill,

                "Estimated Skill":
                    estimated_skill,

                "Uncertainty":
                    uncertainty,

                "Win Rate":
                    recent_win,

                "Error Rate":
                    error_rate,

                "Reward":
                    reward,

                "Success":
                    success,

                "Difficulty Change":
                    abs(
                        adaptation
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    return (
        df,
        float(
            df["Reward"].sum()
        ),
        False,
    )


# ============================================================
# RUN SIMULATION
# ============================================================

def run_gadda(
    initial_skill,
    learning_rate,
    fatigue_rate,
    noise_std,
    seed,
    max_steps,
):

    result = run_real_model(
        initial_skill,
        learning_rate,
        fatigue_rate,
        noise_std,
        seed,
        max_steps,
    )

    if result is not None:
        return result

    return run_visual_demo(
        initial_skill,
        learning_rate,
        fatigue_rate,
        noise_std,
        seed,
        max_steps,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    dedent(
        """
        <div class="hero">

            <div class="hero-kicker">
                Adaptive AI Research Demonstration
            </div>

            <div class="hero-title">
                GADDA
            </div>

            <div class="hero-subtitle">
                Generalizable &amp; Uncertainty-Aware Dynamic Difficulty
                Adjustment using Reinforcement Learning
            </div>

            <div class="hero-tags">

                <span class="tag">
                    PPO
                </span>

                <span class="tag">
                    Player Modelling
                </span>

                <span class="tag">
                    Uncertainty
                </span>

                <span class="tag">
                    Adaptive Decision-Making
                </span>

                <span class="tag">
                    Generalization
                </span>

            </div>

        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Simulation Controls"
    )

    st.caption(
        "Explore how the difficulty policy responds "
        "to changing simulated player behaviour."
    )

    st.divider()

    st.markdown(
        "### Player state"
    )

    initial_skill = st.slider(
        "Initial skill",
        0.05,
        0.95,
        0.50,
        0.01,
    )

    learning_rate = st.slider(
        "Learning rate",
        0.0000,
        0.0080,
        0.0000,
        0.0005,
        format="%.4f",
    )

    fatigue_rate = st.slider(
        "Fatigue rate",
        0.0000,
        0.0080,
        0.0000,
        0.0005,
        format="%.4f",
    )

    noise_std = st.slider(
        "Behavioural noise",
        0.01,
        0.30,
        0.05,
        0.01,
        format="%.2f",
    )

    st.markdown(
        "### Run settings"
    )

    max_steps = st.slider(
        "Episode length",
        50,
        300,
        200,
        10,
    )

    seed = st.number_input(
        "Seed",
        0,
        9999,
        42,
        1,
    )

    run_button = st.button(
        "Run adaptation",
        use_container_width=True,
    )

    st.divider()

    model_state = (
        "Loaded"
        if ppo_model is not None
        else "Not loaded"
    )

    env_state = (
        ENV_CLASS.__name__
        if ENV_CLASS is not None
        else "Fallback visual mode"
    )

    st.markdown(
        dedent(
            f"""
            <div class="glass-card">

                <div class="status">
                    <span class="status-dot"></span>
                    Controller status
                </div>

                <br>

                <b>Policy</b><br>
                PPO · frozen model

                <br><br>

                <b>Model</b><br>
                {model_state}

                <br><br>

                <b>Environment</b><br>
                {env_state}

                <br><br>

                <b>Target</b><br>
                70% success rate

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# INITIAL / UPDATED RUN
# ============================================================

if (
    "gadda_df"
    not in st.session_state
    or run_button
):

    with st.spinner(
        "Following the adaptation trajectory..."
    ):

        df, total_reward, using_real_model = (
            run_gadda(
                float(initial_skill),
                float(learning_rate),
                float(fatigue_rate),
                float(noise_std),
                int(seed),
                int(max_steps),
            )
        )

        st.session_state[
            "gadda_df"
        ] = df

        st.session_state[
            "gadda_reward"
        ] = total_reward

        st.session_state[
            "using_real_model"
        ] = using_real_model


df = st.session_state[
    "gadda_df"
]

total_reward = st.session_state[
    "gadda_reward"
]

using_real_model = st.session_state[
    "using_real_model"
]


# ============================================================
# MODE NOTICE
# ============================================================

if using_real_model:

    st.success(
        "Live research model: the frozen PPO policy is driving this trajectory.",
        icon="✓",
    )

else:

    st.info(
        "Visual demonstration mode: the local PPO/environment pair could not be loaded, so this run is illustrative and is not a publication result.",
        icon="i",
    )


# ============================================================
# KEY METRICS
# ============================================================

mean_win = float(
    df["Win Rate"].mean()
)

final_win = float(
    df["Win Rate"].iloc[-1]
)

target_error = abs(
    mean_win - 0.70
)

final_difficulty = float(
    df["Difficulty"].iloc[-1]
)

mean_uncertainty = float(
    df["Uncertainty"].mean()
)

mean_skill = float(
    df["Estimated Skill"].mean()
)


st.markdown(
    '<div class="section-title">The adaptation at a glance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        A compact view of how the controller responds to the simulated
        player state while tracking the target performance region.
    </div>
    """,
    unsafe_allow_html=True,
)


m1, m2, m3, m4, m5 = st.columns(5)


def metric_card(
    label,
    value,
    note,
):

    return dedent(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-note">
                {note}
            </div>

        </div>
        """
    )


with m1:

    st.markdown(
        metric_card(
            "Mean win rate",
            f"{mean_win:.3f}",
            "Target = 0.700",
        ),
        unsafe_allow_html=True,
    )

with m2:

    st.markdown(
        metric_card(
            "Target error",
            f"{target_error:.3f}",
            "Absolute deviation",
        ),
        unsafe_allow_html=True,
    )

with m3:

    st.markdown(
        metric_card(
            "Current difficulty",
            f"{final_difficulty:.3f}",
            "Latest policy action",
        ),
        unsafe_allow_html=True,
    )

with m4:

    st.markdown(
        metric_card(
            "Estimated skill",
            f"{mean_skill:.3f}",
            "Mean trajectory estimate",
        ),
        unsafe_allow_html=True,
    )

with m5:

    st.markdown(
        metric_card(
            "Uncertainty",
            f"{mean_uncertainty:.3f}",
            "Mean state uncertainty",
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# "WHERE DIFFICULTY LEARNS"
# ============================================================

st.markdown(
    '<div class="section-title">Where difficulty learns</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        The trajectory connects player capability, selected difficulty,
        and observed success. Rotate, zoom and hover to inspect the policy.
    </div>
    """,
    unsafe_allow_html=True,
)


skills = np.linspace(
    0.05,
    0.95,
    45,
)

difficulty_axis = np.linspace(
    0.05,
    0.95,
    45,
)

X, Y = np.meshgrid(
    skills,
    difficulty_axis,
)

Z = (
    1
    /
    (
        1
        +
        np.exp(
            8 * (Y - X)
        )
    )
)


fig = go.Figure()


# Soft pastel surface
fig.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[
            [0.00, "#20263A"],
            [0.25, "#59627D"],
            [0.50, "#9EA9C8"],
            [0.75, "#B7D8D0"],
            [1.00, "#E2D5EA"],
        ],
        opacity=0.72,
        showscale=True,
        colorbar=dict(
            title="Success",
            tickformat=".0%",
            tickfont=dict(
                color="#CBD2DE"
            ),
            titlefont=dict(
                color="#CBD2DE"
            ),
        ),
        hovertemplate=(
            "Skill %{x:.2f}"
            "<br>"
            "Difficulty %{y:.2f}"
            "<br>"
            "Success %{z:.1%}"
            "<extra></extra>"
        ),
        name="Performance field",
    )
)


# Adaptation path
fig.add_trace(
    go.Scatter3d(
        x=df["True Skill"],
        y=df["Difficulty"],
        z=df["Win Rate"],
        mode="lines+markers",
        line=dict(
            color="#F2F4F8",
            width=7,
        ),
        marker=dict(
            size=4,
            color=df["Step"],
            colorscale=[
                [0.00, "#AAB5FF"],
                [0.50, "#C5DAD4"],
                [1.00, "#E6CFE5"],
            ],
            showscale=False,
        ),
        name="Policy trajectory",
        customdata=df[
            [
                "Step",
                "Estimated Skill",
                "Uncertainty",
            ]
        ].values,
        hovertemplate=(
            "Step %{customdata[0]}"
            "<br>"
            "True skill %{x:.3f}"
            "<br>"
            "Difficulty %{y:.3f}"
            "<br>"
            "Win rate %{z:.1%}"
            "<br>"
            "Estimated skill %{customdata[1]:.3f}"
            "<br>"
            "Uncertainty %{customdata[2]:.3f}"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    height=690,
    margin=dict(
        l=0,
        r=0,
        t=10,
        b=0,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#DCE2EA"
    ),
    scene=dict(
        bgcolor="#0B0F19",

        xaxis=dict(
            title="Player capability",
            color="#AEB7C7",
            gridcolor="#252D3D",
            zerolinecolor="#252D3D",
        ),

        yaxis=dict(
            title="Difficulty",
            color="#AEB7C7",
            gridcolor="#252D3D",
            zerolinecolor="#252D3D",
        ),

        zaxis=dict(
            title="Observed success",
            color="#AEB7C7",
            gridcolor="#252D3D",
            zerolinecolor="#252D3D",
        ),
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# ADAPTATION DYNAMICS
# ============================================================

st.markdown(
    '<div class="section-title">Inside the adaptation loop</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        How the selected difficulty and target performance evolve over the episode.
    </div>
    """,
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
            line=dict(
                color="#AAB5FF",
                width=3.5,
            ),
            name="Difficulty",
        )
    )

    fig.update_layout(
        title="Difficulty trajectory",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#DCE2EA"
        ),
        xaxis=dict(
            title="Step",
            gridcolor="#252D3D",
        ),
        yaxis=dict(
            title="Difficulty",
            range=[0, 1],
            gridcolor="#252D3D",
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
            line=dict(
                color="#B7D8D0",
                width=3.5,
            ),
            name="Rolling win rate",
        )
    )

    fig.add_hline(
        y=0.70,
        line_dash="dot",
        line_color="#E4D4E7",
        annotation_text="Target · 70%",
        annotation_font_color="#E4D4E7",
    )

    fig.update_layout(
        title="Target performance tracking",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#DCE2EA"
        ),
        xaxis=dict(
            title="Step",
            gridcolor="#252D3D",
        ),
        yaxis=dict(
            title="Rolling win rate",
            range=[0, 1],
            gridcolor="#252D3D",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# PLAYER STATE
# ============================================================

st.markdown(
    '<div class="section-title">The player state behind the policy</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        The controller does not observe a static player. Skill estimates,
        uncertainty and recent outcomes evolve throughout the episode.
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns(2)


with left:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["True Skill"],
            mode="lines",
            name="True skill",
            line=dict(
                color="#E6CFE5",
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Estimated Skill"],
            mode="lines",
            name="Estimated skill",
            line=dict(
                color="#AAB5FF",
                width=3,
                dash="dot",
            ),
        )
    )

    fig.update_layout(
        title="Skill estimation",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#DCE2EA"
        ),
        xaxis=dict(
            title="Step",
            gridcolor="#252D3D",
        ),
        yaxis=dict(
            title="Skill",
            range=[0, 1],
            gridcolor="#252D3D",
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
                color="#D8C8A8",
                width=3,
            ),
            fill="tozeroy",
            fillcolor="rgba(216,200,168,0.07)",
        )
    )

    fig.update_layout(
        title="Uncertainty over time",
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#DCE2EA"
        ),
        xaxis=dict(
            title="Step",
            gridcolor="#252D3D",
        ),
        yaxis=dict(
            title="Uncertainty",
            range=[0, 1],
            gridcolor="#252D3D",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# RESPONSE SIGNALS
# ============================================================

st.markdown(
    '<div class="section-title">Response signals</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)


with left:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Error Rate"],
            mode="lines",
            line=dict(
                color="#D7B8C9",
                width=3,
            ),
            name="Error rate",
        )
    )

    fig.update_layout(
        title="Observed error rate",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#DCE2EA"
        ),
        xaxis=dict(
            title="Step",
            gridcolor="#252D3D",
        ),
        yaxis=dict(
            title="Error rate",
            range=[0, 1],
            gridcolor="#252D3D",
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
            y=df["Reward"],
            mode="lines",
            line=dict(
                color="#B7D8D0",
                width=3,
            ),
            name="Reward",
        )
    )

    fig.update_layout(
        title="Policy reward",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#DCE2EA"
        ),
        xaxis=dict(
            title="Step",
            gridcolor="#252D3D",
        ),
        yaxis=dict(
            title="Reward",
            gridcolor="#252D3D",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">What the trajectory reveals</div>',
    unsafe_allow_html=True,
)

a, b, c = st.columns(3)


with a:

    st.markdown(
        dedent(
            f"""
            <div class="glass-card">

                <b>Target tracking</b>

                <br><br>

                Mean rolling win rate:
                <b>{mean_win:.3f}</b>

                <br>

                Target:
                <b>0.700</b>

                <br>

                Absolute error:
                <b>{target_error:.3f}</b>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


with b:

    st.markdown(
        dedent(
            f"""
            <div class="glass-card">

                <b>Adaptive response</b>

                <br><br>

                Final difficulty:
                <b>{final_difficulty:.3f}</b>

                <br>

                Mean difficulty:
                <b>{df["Difficulty"].mean():.3f}</b>

                <br>

                Mean change:
                <b>{df["Difficulty Change"].mean():.4f}</b>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


with c:

    st.markdown(
        dedent(
            f"""
            <div class="glass-card">

                <b>State confidence</b>

                <br><br>

                Mean uncertainty:
                <b>{mean_uncertainty:.3f}</b>

                <br>

                Mean estimated skill:
                <b>{mean_skill:.3f}</b>

                <br>

                Episode reward:
                <b>{total_reward:.2f}</b>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# TRAJECTORY DATA
# ============================================================

with st.expander(
    "Inspect the trajectory"
):

    st.dataframe(
        df.round(4),
        use_container_width=True,
        height=420,
    )


csv_data = df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    "Download trajectory data",
    data=csv_data,
    file_name="gadda_demo_trajectory.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    dedent(
        """
        <div class="footer">

            <b>GADDA</b> · Generalizable &amp; Uncertainty-Aware
            Dynamic Difficulty Adjustment using Reinforcement Learning

            <br>

            Interactive research demonstration · Simulation environment · PPO

            <br><br>

            This interface is a demonstration layer.
            Controlled experimental results reported in the manuscript
            are generated separately under the defined evaluation protocol.

        </div>
        """
    ),
    unsafe_allow_html=True,
)
