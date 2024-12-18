
def bumblebee_kinematics_model(time=None, PHI=115.0, phi_m=24.0, dTau=0.00, alpha_down=70.0, alpha_up=-40.0, tau=0.22, theta=12.55/2):
    """
    Kinematics model for a bumblebee bombus terrestris [Engels et al PRL 2016, PRF 2019]

    Note motion starts with downstroke. Defaults are set to values used in [Engels et al PRL 2016, PRF 2019].
    
    Alpha is piecewise constant with sin transition, theta is constant and phi is sinusoidal.

    Parameters
    ----------
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
    time : vector of time
        Time vector
    

    Returns
    -------
    alpha, phi, theta
    """

    pass # todo : need to be adapted to work with Vector3D and Angle classes