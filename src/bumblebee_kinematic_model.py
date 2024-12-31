from src.core import Scalar
from src.core import Angle
import numpy as np


def bumblebee_kinematics_model(
    number_time_steps: int,
    PHI: Scalar = 115.0,
    phi_m: Scalar = 24.0,
    dTau: Scalar = 0.00,
    alpha_down: Scalar = 70.0,
    alpha_up: Scalar = -40.0,
    tau: Scalar = 0.22,
    theta: Scalar = 12.55 / 2,
) -> (np.array, Angle, Angle, Angle):
    """
    Kinematics model for a bumblebee bombus terrestris [Engels et al PRL 2016, PRF 2019]

    Note motion starts with downstroke. Defaults are set to values used in [Engels et al PRL 2016, PRF 2019].

    Alpha is piecewise constant with sin transition, theta is constant and phi is sinusoidal.

    Parameters:
        number_time_steps : int
            Number of time steps
        PHI : float, scalar
            Stroke amplitude (deg)
        phi_m : float, scalar
            Mean stroke angle_rad (deg)
        dTau : float, scalar
            Delay parameter of supination/pronation
        alpha_down : float, scalar, optional
            Featherng angle_rad during downstroke. (deg)
        alpha_up : float, scalar, optional
            Feathering angle_rad during upstroke. (deg)
        tau :
            duration of wing rotation
        theta :
            constant deviation angle_rad (deg)


    Returns:
        time : np.array
            Time vector of shape (number_time_steps,)
        alpha : Angle
        phi : Angle
        theta : Angle
    """

    if not isinstance(number_time_steps, int):
        raise TypeError("number_time_steps must be an integer")

    time = np.linspace(0.0, 1.0, endpoint=False, num=number_time_steps)

    # phi is sinusoidal function with fixed phase (variable amplitude+offset)
    phi = phi_m + (PHI / 2.0) * np.sin(2.0 * np.pi * (time + 0.25))
    # theta is a constant value
    theta = np.zeros_like(time) + theta

    # alpha is a new function
    # d_tau is the timing of pronation and supination, which Muijres2016 identified as important parameter in the compensation
    # degrees
    alpha_tau = tau  # fixed parameters (rotation duration) upstroke->downstroke
    alpha_tau1 = tau  # downstroke->upstroke

    T1 = alpha_tau1 / 2.0
    T2 = 0.5 - alpha_tau / 2.0
    T3 = T2 + alpha_tau
    T4 = 1.0 - alpha_tau1 / 2.0

    pi = np.pi
    a = (alpha_up - alpha_down) / alpha_tau
    a1 = (alpha_up - alpha_down) / alpha_tau1

    alpha = np.zeros_like(time)

    for it, t in enumerate(time):
        if t < T1:
            alpha[it] = alpha_down - a1 * (
                t
                - alpha_tau1 / 2.0
                - (alpha_tau1 / 2.0 / pi)
                * np.sin(2.0 * pi * (t - alpha_tau1 / 2.0) / alpha_tau1)
            )

        elif t >= T1 and t < T2:
            alpha[it] = alpha_down

        elif t >= T2 and t < T3:
            alpha[it] = alpha_down + a * (
                t - T2 - (alpha_tau / 2 / pi) * np.sin(2 * pi * (t - T2) / alpha_tau)
            )

        elif t >= T3 and t < T4:
            alpha[it] = alpha_up

        elif t >= T4:
            TT = 1.0 - alpha_tau1 / 2.0
            alpha[it] = alpha_up - a1 * (
                t
                - TT
                - (alpha_tau1 / 2 / pi) * np.sin(2 * pi * ((t - TT) / alpha_tau1))
            )

    # this now is the important part that circularily shifts the entire vector.
    # it thus changes the "timing of pronation and supination"
    dt = time[1] - time[0]
    shift = int(np.round(dTau / dt))
    alpha = np.roll(alpha, shift)

    # integration with the Angle class
    alpha = Angle(alpha, "deg")
    phi = Angle(phi, "deg")
    theta = Angle(theta, "deg")

    return time, alpha, phi, theta
