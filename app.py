import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GADDA | Adaptive Difficulty Research",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(99,102,241,0.12), transparent 30%),
            radial-gradient(circle at 85% 20%, rgba(168,85,247,0.10), transparent 30%),
            #080b14;
        color: #f5f7ff;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Hero */
    .hero {
        padding: 2.5rem 2.5rem 2.2rem 2.5rem;
        border-radius: 28px;
        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.20),
                rgba(139,92,246,0.10),
                rgba(8,11,20,0.95)
            );
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.8rem;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #b8bfd3;
        line-height: 1.6;
        max-width: 900px;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(129,140,248,0.30);
        color: #c7d2fe;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    /* KPI cards */
    .kpi {
        padding: 1.25rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 135px;
    }

    .kpi-label {
        color: #929bb3;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.35rem;
        color: #ffffff;
    }

    .kpi-note {
        color: #8f98ae;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }

    /* Section */
    .section-title {
        font-size: 1.65rem;
        font-weight: 750;
        margin-top: 2rem;
        margin-bottom: 0.3rem;
    }

    .section-subtitle {
        color: #8f98ae;
        margin-bottom: 1rem;
    }

    /* Research cards */
    .research-card {
        padding: 1.3rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        height: 100%;
    }

    .research-card h4 {
        margin-top: 0;
        color: #f5f7ff;
    }

    .research-card p {
        color: #aab2c5;
        line-height: 1.55;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #080b14;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    /* Slider labels */
    .stSlider label {
        color: #dce1ef !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.07);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

standard_results = pd.DataFrame(
    {
        "Method": [
            "GADDA PPO",
            "Rule-Based DDA",
            "Static Easy",
            "Static Medium",
            "Static Hard",
        ],
        "Mean Target Win-Rate Error": [
            0.0855,
            0.1340,
            0.2303,
            0.2045,
            0.5183,
        ],
    }
)

heldout_results = pd.DataFrame(
    {
        "Player Configuration": [
            "Unseen Low Skill",
            "Unseen High Skill",
            "Fast Learning",
            "Strong Fatigue",
            "High Observation Noise",
        ],
        "GADDA PPO": [
            0.1216,
            0.1283,
            0.0707,
            0.0269,
            0.0525,
        ],
        "Rule-Based DDA": [
            0.0207,
            0.2964,
            0.2375,
            0.1794,
            0.1199,
        ],
    }
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎮 GADDA")

    st.caption("Research Dashboard")

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Results",
            "🧊 3D State Explorer",
            "🕸️ 5D Player Profile",
            "🎮 DDA Simulator",
        ],
    )

    st.divider()

    st.markdown("### Research Status")

    st.success("Simulation Study Complete")

    st.caption(
        "GADDA: Generalizable and Uncertainty-Aware "
        "Dynamic Difficulty Adjustment using Reinforcement Learning"
    )

    st.divider()

    st.caption("Built with Streamlit + Plotly")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            REINFORCEMENT LEARNING • ADAPTIVE AI • GAME INTELLIGENCE
        </div>

        <div class="hero-title">
            GADDA
        </div>

        <div style="
            font-size:1.35rem;
            font-weight:600;
            color:#d8dcf0;
            margin-bottom:1rem;
        ">
            Generalizable and Uncertainty-Aware Dynamic Difficulty Adjustment
        </div>

        <div class="hero-subtitle">
            An adaptive reinforcement-learning framework that dynamically
            adjusts game difficulty using estimated player skill,
            uncertainty, learning dynamics, fatigue, and noisy observations.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Research at a glance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Interactive overview of the GADDA simulation study."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="kpi">
                <div class="kpi-label">Primary Method</div>
                <div class="kpi-value">PPO</div>
                <div class="kpi-note">Reinforcement learning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="kpi">
                <div class="kpi-label">Target Win Rate</div>
                <div class="kpi-value">50–60%</div>
                <div class="kpi-note">Desired player zone</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="kpi">
                <div class="kpi-label">Standard Error</div>
                <div class="kpi-value">0.0855</div>
                <div class="kpi-note">GADDA PPO</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="kpi">
                <div class="kpi-label">Held-Out Profiles</div>
                <div class="kpi-value">5</div>
                <div class="kpi-note">Generalization evaluation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Why GADDA?</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="research-card">
                <h4>🧠 Adaptive</h4>
                <p>
                Difficulty is treated as a sequential decision problem.
                The agent continuously observes player behavior and selects
                difficulty adjustments rather than following fixed rules.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="research-card">
                <h4>🎯 Target-Aware</h4>
                <p>
                The reward structure encourages the agent to maintain
                player performance near a target win-rate zone instead
                of simply maximizing or minimizing difficulty.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="research-card">
                <h4>🌎 Generalizable</h4>
                <p>
                The framework is evaluated on held-out simulated player
                configurations involving skill differences, learning speed,
                fatigue, and observation noise.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Research pipeline</div>',
        unsafe_allow_html=True,
    )

    pipeline = pd.DataFrame(
        {
            "Stage": [
                "Player Modeling",
                "State Estimation",
                "RL Environment",
                "PPO Training",
                "Baseline Comparison",
                "Held-Out Evaluation",
            ],
            "Progress": [100, 100, 100, 100, 100, 100],
        }
    )

    fig = px.bar(
        pipeline,
        x="Progress",
        y="Stage",
        orientation="h",
        range_x=[0, 100],
        text="Progress",
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="inside",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="Completion",
        yaxis_title="",
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RESULTS
# ============================================================

elif page == "📊 Results":

    st.markdown(
        '<div class="section-title">Experimental Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Lower Mean Target Win-Rate Error indicates better control of the target zone."
        "</div>",
        unsafe_allow_html=True,
    )

    # Standard evaluation
    st.markdown("### Standard Evaluation")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=standard_results["Method"],
            y=standard_results["Mean Target Win-Rate Error"],
            text=[
                f"{x:.4f}"
                for x in standard_results["Mean Target Win-Rate Error"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean Target Win-Rate Error: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=470,
        title="Mean Target Win-Rate Error",
        yaxis_title="Error",
        xaxis_title="Method",
        margin=dict(l=20, r=20, t=60, b=30),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        standard_results.style.format(
            {"Mean Target Win-Rate Error": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Generalization
    st.markdown("### Held-Out Generalization")

    melted = heldout_results.melt(
        id_vars="Player Configuration",
        var_name="Method",
        value_name="Mean Target Win-Rate Error",
    )

    fig2 = px.bar(
        melted,
        x="Player Configuration",
        y="Mean Target Win-Rate Error",
        color="Method",
        barmode="group",
        text="Mean Target Win-Rate Error",
    )

    fig2.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    fig2.update_layout(
        template="plotly_dark",
        height=520,
        yaxis_title="Mean Target Win-Rate Error",
        xaxis_title="Held-Out Player Configuration",
        margin=dict(l=20, r=20, t=40, b=100),
        legend_title="Method",
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Improvement calculation
    improvement = (
        (heldout_results["Rule-Based DDA"]
         - heldout_results["GADDA PPO"])
        / heldout_results["Rule-Based DDA"]
        * 100
    )

    improvement_df = pd.DataFrame(
        {
            "Configuration": heldout_results["Player Configuration"],
            "GADDA Improvement (%)": improvement,
        }
    )

    st.markdown("### GADDA improvement over Rule-Based DDA")

    fig3 = px.bar(
        improvement_df,
        x="Configuration",
        y="GADDA Improvement (%)",
        text="GADDA Improvement (%)",
    )

    fig3.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig3.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="Improvement (%)",
        xaxis_title="",
        margin=dict(l=20, r=20, t=30, b=90),
    )

    st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# 3D STATE EXPLORER
# ============================================================

elif page == "🧊 3D State Explorer":

    st.markdown(
        '<div class="section-title">3D Player-State Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Explore how player state can be represented across skill, fatigue, and uncertainty."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        skill = st.slider(
            "Estimated Skill",
            0.0,
            1.0,
            0.55,
            0.01,
        )

    with col2:
        fatigue = st.slider(
            "Fatigue",
            0.0,
            1.0,
            0.25,
            0.01,
        )

    with col3:
        uncertainty = st.slider(
            "Skill Uncertainty",
            0.0,
            1.0,
            0.20,
            0.01,
        )

    # Generate conceptual state cloud.
    # This is an interactive visualization, not an experimental result.
    rng = np.random.default_rng(42)

    n = 180

    skills = np.clip(
        rng.normal(skill, 0.16, n),
        0,
        1,
    )

    fatigues = np.clip(
        rng.normal(fatigue, 0.16, n),
        0,
        1,
    )

    uncertainties = np.clip(
        rng.normal(uncertainty, 0.13, n),
        0,
        1,
    )

    difficulty = np.clip(
        0.55 * skills
        + 0.25 * (1 - fatigues)
        + 0.20 * (1 - uncertainties),
        0,
        1,
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=skills,
                y=fatigues,
                z=uncertainties,
                mode="markers",
                marker=dict(
                    size=5,
                    color=difficulty,
                    colorscale="Viridis",
                    opacity=0.78,
                    colorbar=dict(
                        title="Difficulty"
                    ),
                ),
                text=[
                    f"Difficulty: {d:.2f}"
                    for d in difficulty
                ],
                hovertemplate=(
                    "Skill: %{x:.2f}<br>"
                    "Fatigue: %{y:.2f}<br>"
                    "Uncertainty: %{z:.2f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        ]
    )

    # Current player state
    fig.add_trace(
        go.Scatter3d(
            x=[skill],
            y=[fatigue],
            z=[uncertainty],
            mode="markers",
            marker=dict(
                size=13,
                color="red",
                symbol="diamond",
                line=dict(
                    width=3,
                    color="white",
                ),
            ),
            name="Current Player State",
            hovertemplate=(
                "<b>Current Player</b><br>"
                "Skill: %{x:.2f}<br>"
                "Fatigue: %{y:.2f}<br>"
                "Uncertainty: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        scene=dict(
            xaxis_title="Estimated Skill",
            yaxis_title="Fatigue",
            zaxis_title="Skill Uncertainty",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Interactive 3D Player State Space",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This 3D view is an interactive conceptual visualization of the "
        "state variables used by the adaptive framework. It is not presented "
        "as an additional experimental result."
    )


# ============================================================
# 5D PLAYER PROFILE
# ============================================================

elif page == "🕸️ 5D Player Profile":

    st.markdown(
        '<div class="section-title">5D Player Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Explore five dimensions that can influence adaptive difficulty decisions."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        skill = st.slider("Skill", 0, 100, 60)
        learning = st.slider("Learning Rate", 0, 100, 55)
        fatigue = st.slider("Fatigue Resistance", 0, 100, 70)

    with c2:
        consistency = st.slider("Performance Consistency", 0, 100, 65)
        confidence = st.slider("Estimation Confidence", 0, 100, 75)

    categories = [
        "Skill",
        "Learning",
        "Fatigue Resistance",
        "Consistency",
        "Confidence",
    ]

    values = [
        skill,
        learning,
        fatigue,
        consistency,
        confidence,
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Player Profile",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=570,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            )
        ),
        showlegend=False,
        title="Five-Dimensional Player Profile",
        margin=dict(l=50, r=50, t=80, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Adaptive recommendation
    estimated_difficulty = (
        0.35 * skill
        + 0.25 * learning
        + 0.20 * fatigue
        + 0.10 * consistency
        + 0.10 * confidence
    )

    st.markdown("### Adaptive Difficulty Signal")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Estimated Difficulty",
            f"{estimated_difficulty:.1f}/100",
        )

    with m2:
        if estimated_difficulty < 40:
            zone = "Easy"
        elif estimated_difficulty < 70:
            zone = "Balanced"
        else:
            zone = "Challenging"

        st.metric(
            "Suggested Zone",
            zone,
        )

    with m3:
        st.metric(
            "Target Win Rate",
            "50–60%",
        )

    st.info(
        "This profile visualization is an interactive research-demo "
        "interface and should not be interpreted as a validated clinical, "
        "psychological, or player-assessment instrument."
    )


# ============================================================
# DDA SIMULATOR
# ============================================================

elif page == "🎮 DDA Simulator":

    st.markdown(
        '<div class="section-title">Interactive DDA Simulator</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Experiment with player conditions and observe a conceptual adaptive difficulty response."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        player_skill = st.slider(
            "Player Skill",
            0.0,
            1.0,
            0.55,
            0.01,
        )

        player_fatigue = st.slider(
            "Player Fatigue",
            0.0,
            1.0,
            0.20,
            0.01,
        )

    with c2:

        player_uncertainty = st.slider(
            "Skill Uncertainty",
            0.0,
            1.0,
            0.20,
            0.01,
        )

        observation_noise = st.slider(
            "Observation Noise",
            0.0,
            1.0,
            0.10,
            0.01,
        )

    # Conceptual adaptive policy
    base = (
        0.55 * player_skill
        + 0.25 * (1 - player_fatigue)
        + 0.20 * (1 - player_uncertainty)
    )

    noise_penalty = 0.15 * observation_noise

    recommended = np.clip(
        base - noise_penalty,
        0,
        1,
    )

    difficulty = recommended * 100

    if difficulty < 35:
        level = "Easy"
    elif difficulty < 65:
        level = "Balanced"
    elif difficulty < 82:
        level = "Hard"
    else:
        level = "Very Hard"

    st.markdown("### Current Recommendation")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Difficulty",
            f"{difficulty:.1f}/100",
        )

    with r2:
        st.metric(
            "Difficulty Level",
            level,
        )

    with r3:
        st.metric(
            "Target Zone",
            "50–60%",
        )

    # Gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=difficulty,
            title={"text": "Adaptive Difficulty"},
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "threshold": {
                    "line": {
                        "width": 5
                    },
                    "thickness": 0.8,
                    "value": 60,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=30, r=30, t=70, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Important: this simulator is a conceptual interactive demonstration "
        "of the dashboard interface. It does not execute the trained PPO model."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#70788f;
        font-size:0.78rem;
        padding:1rem;
    ">
        GADDA • Generalizable and Uncertainty-Aware Dynamic Difficulty Adjustment
        <br>
        Reinforcement Learning Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
