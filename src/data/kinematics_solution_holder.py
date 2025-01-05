from dataclasses import dataclass
from core import Angle, Vector3D
import numpy as np


@dataclass
class KinematicsSolutionHolder:
    # Time Vector
    time: np.array

    # Angles
    phi: Angle
    alpha: Angle
    theta: Angle
    eta: Angle
    psi: Angle
    beta: Angle
    gamma: Angle
    angle_of_attack: Angle

    # Angles time derivatives
    phi_dt: Angle
    alpha_dt: Angle
    theta_dt: Angle

    # Unit vectors
    ex: Vector3D
    ey: Vector3D
    ez: Vector3D

    # Velocities
    omega: Vector3D
    u_tip: Vector3D
    
    def __init__(self):
        pass

    # todo : add the other important quantities to be computed

    def save(file_name: str):
        pass  # todo save as csv, or npy (binary)

    def load(file_name: str):
        pass  # todo : load a csv, or npy (binary)
