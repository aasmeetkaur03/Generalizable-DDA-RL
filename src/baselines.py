
import numpy as np


class StaticPolicy:
    """
    Always selects the same difficulty.
    """

    def __init__(self, difficulty):

        self.difficulty = float(
            np.clip(
                difficulty,
                0.0,
                1.0
            )
        )


    def reset(self):

        pass


    def select_action(
        self,
        observation
    ):

        return np.array(
            [self.difficulty],
            dtype=np.float32
        )


class RuleBasedDDA:
    """
    Simple rule-based Dynamic Difficulty Adjustment.

    If player performs above the target:
        increase difficulty.

    If player performs below the target:
        decrease difficulty.

    Otherwise:
        keep difficulty stable.
    """

    def __init__(
        self,
        target_win_rate=0.70,
        tolerance=0.05,
        adjustment_step=0.05,
        initial_difficulty=0.50
    ):

        self.target_win_rate = (
            target_win_rate
        )

        self.tolerance = tolerance

        self.adjustment_step = (
            adjustment_step
        )

        self.initial_difficulty = (
            initial_difficulty
        )

        self.reset()


    def reset(self):

        self.current_difficulty = (
            self.initial_difficulty
        )


    def select_action(
        self,
        observation
    ):

        recent_win_rate = float(
            observation[2]
        )


        upper_threshold = (
            self.target_win_rate
            +
            self.tolerance
        )


        lower_threshold = (
            self.target_win_rate
            -
            self.tolerance
        )


        if (
            recent_win_rate
            >
            upper_threshold
        ):

            self.current_difficulty += (
                self.adjustment_step
            )


        elif (
            recent_win_rate
            <
            lower_threshold
        ):

            self.current_difficulty -= (
                self.adjustment_step
            )


        self.current_difficulty = float(
            np.clip(
                self.current_difficulty,
                0.0,
                1.0
            )
        )


        return np.array(
            [
                self.current_difficulty
            ],
            dtype=np.float32
        )
