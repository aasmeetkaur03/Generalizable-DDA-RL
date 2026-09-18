
import numpy as np


class PlayerSkillEstimator:
    """
    Estimates hidden player skill from observable gameplay telemetry.

    The estimator does NOT access true player skill.

    Inputs:
    - success
    - completion_time
    - error_rate

    Outputs:
    - estimated_skill
    - uncertainty

    Uncertainty is a heuristic confidence proxy based on:
    - recent estimation variability
    - number of observations
    """

    def __init__(
        self,
        window_size=20,
        smoothing=0.2
    ):

        self.window_size = window_size
        self.smoothing = smoothing

        self.reset()


    def reset(self):

        self.estimated_skill = 0.5

        self.observation_history = []

        self.estimate_history = []


    def _telemetry_to_skill(
        self,
        success,
        completion_time,
        error_rate
    ):

        success_signal = float(success)

        time_signal = 1.0 / (
            1.0 + completion_time
        )

        error_signal = (
            1.0 - error_rate
        )

        skill_signal = (
            0.50 * success_signal
            + 0.25 * time_signal
            + 0.25 * error_signal
        )

        return float(
            np.clip(
                skill_signal,
                0.0,
                1.0
            )
        )


    def update(
        self,
        success,
        completion_time,
        error_rate
    ):

        observation_skill = (
            self._telemetry_to_skill(
                success,
                completion_time,
                error_rate
            )
        )

        self.observation_history.append(
            observation_skill
        )

        if len(self.observation_history) > self.window_size:

            self.observation_history.pop(0)


        self.estimated_skill = (
            (1 - self.smoothing)
            * self.estimated_skill
            +
            self.smoothing
            * observation_skill
        )

        self.estimated_skill = float(
            np.clip(
                self.estimated_skill,
                0.0,
                1.0
            )
        )


        # Store smoothed estimates
        self.estimate_history.append(
            self.estimated_skill
        )

        if len(self.estimate_history) > self.window_size:

            self.estimate_history.pop(0)


        # -----------------------------
        # Improved uncertainty estimate
        # -----------------------------

        n = len(
            self.estimate_history
        )

        if n < 2:

            uncertainty = 0.20

        else:

            estimate_std = np.std(
                self.estimate_history
            )

            # Standard error decreases
            # as more observations arrive.
            uncertainty = (
                estimate_std
                /
                np.sqrt(n)
            )

            # Keep uncertainty within
            # a useful normalized range.
            uncertainty = float(
                np.clip(
                    uncertainty,
                    0.02,
                    0.25
                )
            )


        return {

            "estimated_skill":
                float(
                    self.estimated_skill
                ),

            "uncertainty":
                uncertainty,

            "observation_skill":
                float(
                    observation_skill
                )
        }
