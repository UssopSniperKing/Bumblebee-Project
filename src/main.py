#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data import WING_CONTOUR_W_X, WING_CONTOUR_W_Y
from src.core import Angle
from src.data import KinematicsSolutionHolder
from src import bumblebee_kinematic_model

# ? : maybe move this to a separate file
def evaluate_angles_kinematics(number_time_steps: int, Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the kinematics of the bumblebee model

    Args:
        number_time_steps (int): Number of time steps
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    Holder.time, Holder.alpha, Holder.phi, Holder.theta = bumblebee_kinematic_model(number_time_steps)
    # todo : add the other important quantities to be computed, time derivative of alpha, phi, theta, etc.
    return Holder


# ? : maybe move this to a separate file
def evaluate_velocities(Holder: KinematicsSolutionHolder) -> KinematicsSolutionHolder:
    """Evaluate the velocities of the bumblebee model

    Args:
        Holder (KinematicsSolutionHolder): Holder for the kinematic solution

    Returns:
        KinematicsSolutionHolder: Holder for the kinematic solution
    """
    # todo : add the other important quantities to be computed, u_tip, etc.
    return Holder
    

def main() -> None:
    """Main function
    """
    NUMBER_TIME_STEPS = 1000
    Kinematics = KinematicsSolutionHolder()
    Kinematics = evaluate_angles_kinematics(NUMBER_TIME_STEPS, Kinematics)
    Kinematics = evaluate_velocities(Kinematics)


    print(WING_CONTOUR_W_X) # as a test for now

if __name__ == '__main__':
    main()
