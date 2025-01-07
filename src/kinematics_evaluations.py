from bumblebee_kinematic_model import bumblebee_kinematics_model
from aerodynamic_model import drag_coefficient, lift_coefficient
from core import angle_time_derivative
from data import KinematicsSolutionHolder
from core import Vector3D
from core import Referential
from core import cross
from core import normalize
from core import Angle
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
    """Define the unit vectors of the bumblebee model

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution
    
    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    Holder.ex = Vector3D([1, 0, 0], Referential.WING)
    Holder.ey = Vector3D([0, 1, 0], Referential.WING)
    Holder.ez = Vector3D([0, 0, 1], Referential.WING)

    return Holder


def evaluate_angular_velocity(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the angular velocity of the bumblebee model

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
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
    Holder.omega.set_referential(Referential.WING)
    Holder.ey.set_referential(Referential.WING)

    Holder.u_tip = cross(Holder.omega, Holder.ey)

    return Holder


def define_aero_unit_vectors(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Define the aerodynamic unit vectors

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """

    u_tip_global = Holder.u_tip.set_referential(Referential.GLOBAL)
    ey_global = Holder.ey.set_referential(Referential.GLOBAL)
    
    u_tip_global_opposite_coords = - u_tip_global.coords
    u_tip_global_opposite = Vector3D(u_tip_global_opposite_coords, Referential.GLOBAL)

    e_drag_global = normalize(u_tip_global_opposite)
    e_lift_global = cross(ey_global, e_drag_global)

    sign_alpha = np.sign(Holder.alpha.radians)
    e_lift_global_coords = np.where(sign_alpha == -1, -e_lift_global.coords, e_lift_global.coords)

    Holder.e_drag = e_drag_global
    Holder.e_lift = Vector3D(e_lift_global_coords, Referential.GLOBAL)

    return Holder


def compute_angle_of_attack(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Compute the angle of attack as arctan²(-omega_wing_y, -omega_wing_z)

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    omega_wing = Holder.omega.set_referential(Referential.WING)
    omega_wing_y = omega_wing.coords[1,:]
    omega_wing_z = omega_wing.coords[2,:]

    aoa = np.atan2( -omega_wing_y, -omega_wing_z )
    Holder.angle_of_attack = Angle(aoa, "rad")

    return Holder


def compute_aerodynamic_coefficients(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Compute CL and CD"""

    K1 = 1
    K2 = 1
    K3 = 1
    K4 = 1
    # todo : constants needs to find their place in the holder (maybe a dictionnary)

    Holder.lift_coeff = lift_coefficient(Holder.angle_of_attack, K1, K2)
    Holder.drag_coeff = drag_coefficient(Holder.angle_of_attack, K3, K4)

    return Holder
