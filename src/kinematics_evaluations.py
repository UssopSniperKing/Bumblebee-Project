from src import bumblebee_kinematic_model, angle_time_derivative
from data import KinematicsSolutionHolder

def evaluate_angles_kinematics(number_time_steps: int, Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the kinematics of the bumblebee model

    Args:
        number_time_steps (int): Number of time steps
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    Holder.time, Holder.alpha, Holder.phi, Holder.theta = bumblebee_kinematic_model(number_time_steps)

    Holder.alpha_dt = angle_time_derivative(Holder.alpha, Holder.time)
    Holder.phi_dt = angle_time_derivative(Holder.phi, Holder.time)
    Holder.theta_dt = angle_time_derivative(Holder.theta, Holder.time)

    return Holder


def evaluate_velocities(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the velocities of the bumblebee model

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    # todo : add the other important quantities to be computed, u_tip, etc.
    return Holder