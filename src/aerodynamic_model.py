from core import Angle
import numpy as np
from core import Scalar


def lift_coefficient(angle_of_attack: Angle, K1: Scalar, K2: Scalar) -> np.array:
    """Dickinson Model 1999"""
    if not isinstance(angle_of_attack, Angle):
        raise ValueError("angle_of_attack must be an Angle object.")

    angle_1 = Angle(2.13, "deg")
    angle_2 = Angle(7.2, "deg")
    angle = Angle(angle_1.degrees * angle_of_attack.degrees - angle_2.degrees, "deg")
    
    return K1 + K2 * np.sin(angle.radians)


def drag_coefficient(angle_of_attack: Angle, K3: Scalar, K4: Scalar) -> np.array:
    """Dickinson Model 1999"""
    if not isinstance(angle_of_attack, Angle):
        raise ValueError("angle_of_attack must be an Angle object.")

    angle_1 = Angle(2.04, "deg")
    angle_2 = Angle(9.82, "deg")
    angle = Angle(angle_1.degrees * angle_of_attack.degrees - angle_2.degrees, "deg")

    return K3 + K4 * np.cos(angle.radians)
