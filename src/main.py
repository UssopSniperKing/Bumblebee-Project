#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.data import KinematicsSolutionHolder
from src.kinematics_evaluations import evaluate_angles_kinematics, evaluate_velocities
from src.initialize_transformations import initialize_transformations


def main() -> None:
    """Main function
    """
    NUMBER_TIME_STEPS = 1000
    Kinematics = KinematicsSolutionHolder()
    Kinematics = evaluate_angles_kinematics(NUMBER_TIME_STEPS, Kinematics)
    Kinematics = evaluate_velocities(Kinematics)
    initialize_transformations(Kinematics)


if __name__ == '__main__':
    main()
