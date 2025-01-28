from bumblebee_kinematic_model import bumblebee_kinematics_model
from aerodynamic_model import drag_coefficient, lift_coefficient
from core import angle_time_derivative
from core import vector_time_derivative
from data import KinematicsSolutionHolder
from core import Vector3D
from core import Referential
from core import cross
from core import normalize
from core import Angle
import numpy as np
from forces_model import force_RC, force_AMx, force_AMz, force_RD, force_TC, force_TD
from core import dot

def evaluate_angles_kinematics(number_time_steps: int, Holder: KinematicsSolutionHolder, PHI: float = None) -> KinematicsSolutionHolder:
    """Evaluate the kinematics of the bumblebee model

    Args:
        number_time_steps (int): Number of time steps
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """

    if PHI is None:
        Holder.time, alpha, phi, theta = bumblebee_kinematics_model(number_time_steps)
    else:
        Holder.time, alpha, phi, theta = bumblebee_kinematics_model(number_time_steps, PHI=PHI)

    Holder.alpha = Angle(-alpha.radians, "rad")
    Holder.phi = Angle(-phi.radians, "rad")
    Holder.theta = theta

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
    ex_wing = Holder.ex.set_referential(Referential.GLOBAL)
    e_drag = Holder.e_drag.set_referential(Referential.GLOBAL)
    minus_e_drag = Vector3D(-e_drag.coords, Referential.GLOBAL)

    aoa = np.arccos(dot(ex_wing, minus_e_drag))
    Holder.angle_of_attack = Angle(aoa, "rad")

    return Holder


def compute_aerodynamic_coefficients(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Compute CL and CD"""
    
    # K1 = -0.2083
    # K2 = 0.4193
    # K3 = 0.2509
    # K4 = -0.2656
    
    K1 = 1
    K2 = 1
    K3 = 1
    K4 = 1
    # todo : constants needs to find their place in the holder (maybe a dictionnary)

    Holder.lift_coeff = lift_coefficient(Holder.angle_of_attack, K1, K2)
    Holder.drag_coeff = drag_coefficient(Holder.angle_of_attack, K3, K4)

    return Holder


def define_planar_angular_velocity(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """"""
    omega_wing = Holder.omega.set_referential(Referential.WING)
    omega_wing_planar_coords = omega_wing.coords.copy()
    omega_wing_planar_coords[1,:] = 0

    Holder.omega_planar = Vector3D(omega_wing_planar_coords, Referential.WING)

    return Holder


def compute_accelerations(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """"""
    Holder.u_tip.set_referential(Referential.GLOBAL)
    Holder.u_tip_dt = vector_time_derivative(Holder.time, Holder.u_tip)

    Holder.omega.set_referential(Referential.GLOBAL)
    Holder.omega_dt = vector_time_derivative(Holder.time, Holder.omega)

    return Holder

def compute_forces(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """"""

    # C_RC = 0.0972
    # C_RD = 0.0626
    
    # C_AMX1 = -0.0045
    # C_AMX2 = 0.0017

    # C_AMZ1 = 0.0211
    # C_AMZ2 = 0.0422
    # C_AMZ3 = 0.1479
    # C_AMZ4 = -0.1697
    # C_AMZ5 = -0.0063
    # C_AMZ6 = 0.0242

    C_RC = 1
    C_RD = 1
    
    C_AMX1 = 1
    C_AMX2 = 1

    C_AMZ1 = 1
    C_AMZ2 = 1
    C_AMZ3 = 1
    C_AMZ4 = 1
    C_AMZ5 = 1
    C_AMZ6 = 1

    omega_planar_wing = Holder.omega_planar.set_referential(Referential.WING)
    e_lift_global = Holder.e_lift.set_referential(Referential.GLOBAL)
    e_drag_global = Holder.e_drag.set_referential(Referential.GLOBAL)

    ez_global = Holder.ez.set_referential(Referential.GLOBAL)
    u_tip_global = Holder.u_tip.set_referential(Referential.GLOBAL)
    omega_wing = Holder.omega.set_referential(Referential.WING)

    u_tip_dt_wing = Holder.u_tip_dt.set_referential(Referential.WING)
    omega_dt_wing = Holder.omega_dt.set_referential(Referential.WING)
    ex_global = Holder.ex.set_referential(Referential.GLOBAL)

    Holder.force_TD = force_TD(Holder.drag_coeff, omega_planar_wing, e_drag_global)
    Holder.force_TC = force_TC(Holder.lift_coeff, omega_planar_wing, e_lift_global)

    Holder.force_RC = force_RC(u_tip_global, omega_wing, ez_global, C_RC)
    Holder.force_RD = force_RD(omega_wing, ez_global, C_RD)
    Holder.force_AMx = force_AMx(u_tip_dt_wing, ex_global, C_AMX1, C_AMX2)
    Holder.force_AMz = force_AMz(u_tip_dt_wing, omega_dt_wing, ez_global, C_AMZ1, C_AMZ2, C_AMZ3, C_AMZ4, C_AMZ5, C_AMZ6)

    Holder.force_QSM = Holder.force_AMx + Holder.force_AMz + Holder.force_RC + Holder.force_RD + Holder.force_TC + Holder.force_TD

    return Holder


def evaluate_power(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """"""

    position_vector = Vector3D([-0.26,0.59,0], Referential.WING)

    force_QSM = Holder.force_QSM.set_referential(Referential.WING)
    omega_wing = Holder.omega.set_referential(Referential.WING)

    Holder.power = - dot( cross(position_vector, force_QSM), omega_wing)

    return Holder