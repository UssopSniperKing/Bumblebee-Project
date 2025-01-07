#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data import KinematicsSolutionHolder
from initialize_transformations import initialize_transformations
from plot_kinematics import plot_kinematics
from kinematics_evaluations import (
    evaluate_angles_kinematics,
    define_unit_vectors,
    evaluate_angular_velocity,
    evaluate_tip_velocity,
    compute_angle_of_attack,
    compute_aerodynamic_coefficients,
    define_aero_unit_vectors
)


def main() -> None:
    """Main function"""
    NUMBER_TIME_STEPS = 400
    Kinematics = KinematicsSolutionHolder()
    Kinematics = evaluate_angles_kinematics(NUMBER_TIME_STEPS, Kinematics)
    initialize_transformations(Kinematics)

    Kinematics = define_unit_vectors(Kinematics)

    Kinematics = evaluate_angular_velocity(Kinematics)

    Kinematics = evaluate_tip_velocity(Kinematics)

    Kinematics = compute_angle_of_attack(Kinematics)

    Kinematics = compute_aerodynamic_coefficients(Kinematics)

    Kinematics = define_aero_unit_vectors(Kinematics)

    plot_kinematics(Kinematics)


if __name__ == "__main__":
    main()
