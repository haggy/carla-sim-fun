# The leaderboard kit may run on Python 3.7, where annotations must not be evaluated at definition time.
from __future__ import annotations

import matplotlib.axes
import numpy as np


class LongitudinalController:
    """Longitudinal control with a discrete PID controller on the speed error (assignment eqs. 5-7).

    Speeds may be in any unit as long as speed and target speed agree; the default gains were tuned in unit 4B on
    Town01 for speeds in km/h.
    """

    def __init__(
        self,
        KP: float = 0.2,
        KI: float = 0.01,
        KD: float = 1.0,
        max_error_sum: float = 80.0,
        max_action: float = 0.8,
    ) -> None:
        """Set the gains and reset the controller state.

        Parameters
        ----------
        KP : float
            Proportional gain K_p, per km/h of speed error.
        KI : float
            Integral gain K_i, per km/h of accumulated error (summed once per step).
        KD : float
            Derivative gain K_d, per km/h change of the error between two steps.
        max_error_sum : float
            Bound on |sum of errors| for the integral term (anti-windup), in km/h * steps. The default lets the
            integral term reach KI * max_error_sum = 0.8, the largest gas or brake value, and no more.
        max_action : float
            Largest gas or brake value sent to the car.
        """
        self.last_error = 0.0
        self.sum_error = 0.0
        self.last_control = 0.0
        self.speed_history: list[float] = []
        self.target_speed_history: list[float] = []
        self.step_history: list[int] = []

        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.max_error_sum = max_error_sum
        self.max_action = max_action

    def PID_step(self, speed: float, target_speed: float) -> float:
        """One step of the discretised PID control law (eqs. 5 and 6).

        Parameters
        ----------
        speed : float
            Current speed v(t).
        target_speed : float
            Target speed v_target.

        Returns
        -------
        float
            Control signal u(t); positive means gas, negative means brake.
        """
        error = target_speed - speed

        # Anti-windup: while the car cannot follow (e.g. at full gas from standstill) the plain sum keeps growing and
        # later forces a long overshoot until the opposite error has paid it back.
        self.sum_error = float(np.clip(self.sum_error + error, -self.max_error_sum, self.max_error_sum))

        control = self.KP * error + self.KD * (error - self.last_error) + self.KI * self.sum_error

        self.last_error = error
        self.last_control = control
        return control

    def control(self, speed: float, target_speed: float) -> tuple[float, float]:
        """Gas and brake for the current speed (eq. 7).

        Parameters
        ----------
        speed : float
            Current speed.
        target_speed : float
            Target speed.

        Returns
        -------
        tuple[float, float]
            (gas, brake), each in [0, max_action]; at most one of them is non-zero.
        """
        control = self.PID_step(speed, target_speed)
        brake = 0.0
        gas = 0.0

        if control >= 0:
            gas = float(np.clip(control, 0, self.max_action))
        else:
            brake = float(np.clip(-control, 0, self.max_action))

        return gas, brake

    def plot_speed(self, speed: float, target_speed: float, step: int, ax: matplotlib.axes.Axes) -> None:
        """Append the current speeds to the history and redraw it.

        Parameters
        ----------
        speed : float
            Current speed.
        target_speed : float
            Target speed.
        step : int
            Simulation step, the x axis.
        ax : matplotlib.axes.Axes
            Axis to draw into; it is cleared first.
        """
        self.speed_history.append(speed)
        self.target_speed_history.append(target_speed)
        self.step_history.append(step)
        ax.clear()
        ax.plot(self.step_history, self.speed_history, c="green", label="speed")
        ax.plot(self.step_history, self.target_speed_history, label="target speed")
        ax.set_xlabel("step")
        ax.set_ylabel("km/h")
        ax.legend(loc="lower right", fontsize=7)
