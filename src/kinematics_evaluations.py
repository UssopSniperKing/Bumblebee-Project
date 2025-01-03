from bumblebee_kinematic_model import bumblebee_kinematics_model
from core import angle_time_derivative
from data import KinematicsSolutionHolder
from core import Vector3D
from core import Referential
from core import cross
import numpy as np

def evaluate_angles_kinematics(number_time_steps: int, Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the kinematics of the bumblebee model

    Args:
        number_time_steps (int): Number of time steps
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    Holder.time, Holder.alpha, Holder.phi, Holder.theta = bumblebee_kinematics_model(number_time_steps)
    Holder.alpha_dt = angle_time_derivative(Holder.time, Holder.alpha)
    Holder.phi_dt = angle_time_derivative(Holder.time, Holder.phi)
    Holder.theta_dt = angle_time_derivative(Holder.time, Holder.theta)

    return Holder


def define_unit_vectors(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:

    Holder.ex = Vector3D([1, 0, 0], Referential.STROKE)
    Holder.ey = Vector3D([0, 1, 0], Referential.STROKE)
    Holder.ez = Vector3D([0, 0, 1], Referential.STROKE)

    return Holder


def evaluate_angular_velocity(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:

    omega_stroke = np.empty((3, Holder.time.size))
    omega_stroke[0, :] = Holder.phi_dt.radians - np.sin(Holder.theta.radians) * Holder.alpha_dt.radians
    omega_stroke[1, :] = np.cos(Holder.phi.radians) * np.cos(Holder.theta.radians) * Holder.alpha_dt.radians - np.sin(Holder.phi.radians) * Holder.theta_dt.radians
    omega_stroke[2, :] = np.sin(Holder.phi.radians) * np.cos(Holder.theta.radians) * Holder.alpha_dt.radians + np.cos(Holder.phi.radians) * Holder.theta_dt.radians

    Holder.omega = Vector3D(omega_stroke, Referential.STROKE)

    return Holder


def evaluate_tip_velocity(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the velocities of the bumblebee model

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    # todo : add the other important quantities to be computed, u_tip, etc.

    omega_wing = Holder.omega.to_referential(Referential.WING)
    ey_wing = Holder.ey.to_referential(Referential.WING)

    Holder.u_tip = cross(omega_wing, ey_wing)

    return Holder