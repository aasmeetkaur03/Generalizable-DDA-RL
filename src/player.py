
import numpy as np


class SimulatedPlayer:
    """
    Simulated game player.

    The RL agent cannot directly observe true_skill.
    It only receives gameplay telemetry.

    Supports:
    - stable skill
    - learning
    - fatigue
    - noisy performance
    """

    def __init__(
        self,
        initial_skill=0.5,
        learning_rate=0.0,
        fatigue_rate=0.0,
        noise_std=0.05,
        seed=None
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

        self.history = []


    def update_skill(self):

        self.step_count += 1

        learning = (
            self.learning_rate
            * self.step_count
        )

        fatigue = (
            self.fatigue_rate
            * self.step_count
        )

        self.true_skill = (
            self.initial_skill
            + learning
            - fatigue
        )

        self.true_skill = float(
            np.clip(
                self.true_skill,
                0.05,
                0.95
            )
        )


    def play(
        self,
        difficulty
    ):

        self.update_skill()

        performance_noise = (
            self.rng.normal(
                0,
                self.noise_std
            )
        )

        effective_skill = np.clip(
            self.true_skill
            + performance_noise,
            0.01,
            0.99
        )

        success_probability = 1 / (
            1
            + np.exp(
                8
                * (
                    difficulty
                    - effective_skill
                )
            )
        )

        success = int(
            self.rng.random()
            < success_probability
        )

        completion_time = (
            1.0
            + difficulty
            - effective_skill
            + abs(
                performance_noise
            )
        )

        completion_time = max(
            completion_time,
            0.1
        )

        error_rate = np.clip(
            difficulty
            - effective_skill
            + abs(
                performance_noise
            ),
            0.0,
            1.0
        )

        result = {

            "success": success,

            "completion_time":
                float(
                    completion_time
                ),

            "error_rate":
                float(
                    error_rate
                ),

            "true_skill":
                float(
                    self.true_skill
                ),

            "success_probability":
                float(
                    success_probability
                )
        }

        self.history.append(
            result
        )

        return result
