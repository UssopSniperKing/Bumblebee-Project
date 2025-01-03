#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data import KinematicsSolutionHolder
from kinematics_evaluations import evaluate_angles_kinematics, define_unit_vectors, evaluate_angular_velocity, evaluate_tip_velocity
from initialize_transformations import initialize_transformations
from plot_kinematics import plot_kinematics

def main() -> None:
    """Main function
    """
    NUMBER_TIME_STEPS = 1000
    Kinematics = KinematicsSolutionHolder()
    Kinematics = evaluate_angles_kinematics(NUMBER_TIME_STEPS, Kinematics)
    initialize_transformations(Kinematics)

    Kinematics = define_unit_vectors(Kinematics)

    Kinematics = evaluate_angular_velocity(Kinematics)
    
    Kinematics = evaluate_tip_velocity(Kinematics)

    plot_kinematics(Kinematics)



if __name__ == '__main__':
    main()
