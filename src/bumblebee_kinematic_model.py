from src.core import Scalar
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
) -> tuple: # todo : add tests
    """
    Kinematics model for a bumblebee bombus terrestris [Engels et al PRL 2016, PRF 2019]

    Note motion starts with downstroke. Defaults are set to values used in [Engels et al PRL 2016, PRF 2019].

    Alpha is piecewise constant with sin transition, theta is constant and phi is sinusoidal.

    Parameters
    ----------
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


    Returns
    -------
    time, alpha, phi, theta
    """

    time = np.linspace(0, 1, number_time_steps)

    pass  # todo : need to be adapted to work with Vector3D and Angle classes 
