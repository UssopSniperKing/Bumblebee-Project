#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data import KinematicsSolutionHolder
from plot_kinematics import plot_kinematics
from kinematics_evaluations import evaluate_angles_kinematics
from workflows import (normal_workflow, optimization_workflow)


def main() -> None:
    """Main function"""
    NUMBER_TIME_STEPS = 400
    SHOW_FIGURES = False
    SAVE_FIGURES = True
    USE_OPTIMIZATION = False

    Kinematics = KinematicsSolutionHolder()


    if not USE_OPTIMIZATION:

        
        Kinematics = evaluate_angles_kinematics(NUMBER_TIME_STEPS, Kinematics)
        Kinematics = normal_workflow(Kinematics)
        plot_kinematics(Kinematics, SAVE_FIGURES, SHOW_FIGURES)

    else:

        Kinematics = optimization_workflow(Kinematics, NUMBER_TIME_STEPS)
        plot_kinematics(Kinematics, SAVE_FIGURES, SHOW_FIGURES)


if __name__ == "__main__":
    main()
