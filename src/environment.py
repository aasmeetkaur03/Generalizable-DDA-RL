
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from src.player import SimulatedPlayer


class GADDAEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=200,
        target_win_rate=0.70,
        player_config=None,
        seed=None
    ):

        super().__init__()

        if player_config is None:
            player_config = {
                "initial_skill": 0.5,
                "learning_rate": 0.0,
                "fatigue_rate": 0.0,
                "noise_std": 0.05
            }

        self.max_steps = max_steps
        self.target_win_rate = target_win_rate
        self.player_config = player_config
        self.seed_value = seed

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.player = None

        self.step_count = 0
        self.difficulty = 0.5

        self.outcome_history = []
        self.difficulty_history = []
        self.error_history = []

        self.estimated_skill = 0.5
        self.uncertainty = 0.5

        self.reset(seed=seed)


    def _create_player(self, seed=None):

        self.player = SimulatedPlayer(
            initial_skill=self.player_config["initial_skill"],
            learning_rate=self.player_config["learning_rate"],
            fatigue_rate=self.player_config["fatigue_rate"],
            noise_std=self.player_config["noise_std"],
            seed=seed
        )


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed

        self._create_player(seed=self.seed_value)

        self.step_count = 0
        self.difficulty = 0.5

        self.outcome_history = []
        self.difficulty_history = [self.difficulty]
        self.error_history = []

        self.estimated_skill = float(
            self.player_config["initial_skill"]
        )

        self.uncertainty = 0.5

        observation = self._get_observation()

        return observation, {}


    def _get_recent_win_rate(self):

        if len(self.outcome_history) == 0:
            return self.target_win_rate

        window = min(
            10,
            len(self.outcome_history)
        )

        return float(
            np.mean(
                self.outcome_history[-window:]
            )
        )


    def _update_estimate(
        self,
        telemetry
    ):

        observed_skill = float(
            np.clip(
                self.difficulty
                + (
                    0.15
                    if telemetry["success"] == 1
                    else -0.15
                )
                - 0.10 * telemetry["error_rate"],
                0.0,
                1.0
            )
        )

        prediction_error = abs(
            observed_skill
            - self.estimated_skill
        )

        learning_rate = 0.15

        self.estimated_skill = float(
            np.clip(
                (
                    (1.0 - learning_rate)
                    * self.estimated_skill
                )
                +
                (
                    learning_rate
                    * observed_skill
                ),
                0.0,
                1.0
            )
        )

        if len(self.outcome_history) < 5:

            self.uncertainty = 0.5

        else:

            recent_outcomes = np.array(
                self.outcome_history[-10:]
            )

            outcome_variability = np.std(
                recent_outcomes
            )

            self.uncertainty = float(
                np.clip(
                    0.5 * prediction_error
                    +
                    0.5 * outcome_variability,
                    0.0,
                    1.0
                )
            )

        return {
            "estimated_skill":
                self.estimated_skill,

            "uncertainty":
                self.uncertainty
        }


    def _get_streak(self):

        if len(self.outcome_history) == 0:
            return 0.0

        recent = self.outcome_history

        last_outcome = recent[-1]

        streak = 0

        for outcome in reversed(recent):

            if outcome == last_outcome:
                streak += 1
            else:
                break

        signed_streak = (
            streak
            if last_outcome == 1
            else -streak
        )

        return float(
            np.clip(
                signed_streak / 10.0,
                -1.0,
                1.0
            )
        )


    def _get_observation(self):

        recent_win_rate = (
            self._get_recent_win_rate()
        )

        streak = self._get_streak()

        normalized_streak = (
            (streak + 1.0)
            / 2.0
        )

        observation = np.array(
            [
                self.estimated_skill,
                self.uncertainty,
                self.difficulty,
                recent_win_rate,
                normalized_streak
            ],
            dtype=np.float32
        )

        return np.clip(
            observation,
            0.0,
            1.0
        )


    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        target_difficulty = float(
            np.clip(
                action[0],
                0.0,
                1.0
            )
        )

        previous_difficulty = (
            self.difficulty
        )

        smoothing = 0.20

        self.difficulty = float(
            (
                (1.0 - smoothing)
                * self.difficulty
            )
            +
            (
                smoothing
                * target_difficulty
            )
        )

        self.difficulty = float(
            np.clip(
                self.difficulty,
                0.0,
                1.0
            )
        )

        self.difficulty_history.append(
            self.difficulty
        )

        telemetry = self.player.play(
            self.difficulty
        )

        self.outcome_history.append(
            telemetry["success"]
        )

        self.error_history.append(
            telemetry["error_rate"]
        )

        estimate = self._update_estimate(
            telemetry
        )

        recent_win_rate = (
            self._get_recent_win_rate()
        )

        win_rate_error = (
            recent_win_rate
            - self.target_win_rate
        )

        flow_penalty = (
            8.0
            * (
                win_rate_error ** 2
            )
        )

        flow_reward = (
            1.0
            - flow_penalty
        )

        if abs(win_rate_error) <= 0.05:

            flow_zone_bonus = 0.50

        else:

            flow_zone_bonus = 0.0

        difficulty_change = abs(
            self.difficulty
            - previous_difficulty
        )

        smoothness_penalty = (
            0.20
            * difficulty_change
        )

        error_penalty = (
            0.10
            * telemetry["error_rate"]
        )

        uncertainty_penalty = (
            0.05
            * estimate["uncertainty"]
        )

        reward = float(
            flow_reward
            +
            flow_zone_bonus
            -
            smoothness_penalty
            -
            error_penalty
            -
            uncertainty_penalty
        )

        self.step_count += 1

        terminated = (
            self.step_count
            >= self.max_steps
        )

        truncated = False

        observation = (
            self._get_observation()
        )

        info = {
            "difficulty":
                float(self.difficulty),

            "target_difficulty":
                float(target_difficulty),

            "true_skill":
                float(
                    telemetry["true_skill"]
                ),

            "estimated_skill":
                float(
                    estimate[
                        "estimated_skill"
                    ]
                ),

            "uncertainty":
                float(
                    estimate[
                        "uncertainty"
                    ]
                ),

            "success":
                int(
                    telemetry["success"]
                ),

            "recent_win_rate":
                float(
                    recent_win_rate
                ),

            "error_rate":
                float(
                    telemetry["error_rate"]
                ),

            "completion_time":
                float(
                    telemetry["completion_time"]
                ),

            "success_probability":
                float(
                    telemetry[
                        "success_probability"
                    ]
                )
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
