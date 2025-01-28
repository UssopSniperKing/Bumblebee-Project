from data import KinematicsSolutionHolder
from initialize_transformations import initialize_transformations
import scipy.optimize as opt
from core import Referential
from kinematics_evaluations import (
    evaluate_angles_kinematics,
    define_unit_vectors,
    evaluate_angular_velocity,
    evaluate_tip_velocity,
    compute_angle_of_attack,
    compute_aerodynamic_coefficients,
    define_aero_unit_vectors,
    define_planar_angular_velocity,
    compute_accelerations,
    compute_forces,
    evaluate_power
)
import numpy as np

def normal_workflow(Kinematics: KinematicsSolutionHolder) -> KinematicsSolutionHolder:

    initialize_transformations(Kinematics)
    Kinematics = define_unit_vectors(Kinematics)
    Kinematics = evaluate_angular_velocity(Kinematics)
    Kinematics = evaluate_tip_velocity(Kinematics)
    Kinematics = define_aero_unit_vectors(Kinematics)
    Kinematics = compute_angle_of_attack(Kinematics)
    Kinematics = compute_aerodynamic_coefficients(Kinematics)
    Kinematics = define_planar_angular_velocity(Kinematics)
    Kinematics = compute_accelerations(Kinematics)
    Kinematics = compute_forces(Kinematics)
    Kinematics = evaluate_power(Kinematics)

    return Kinematics


def optimization_workflow(Holder: KinematicsSolutionHolder, number_time_steps: int) -> KinematicsSolutionHolder:

    dim_parameters = {"mass": 175*10**(-6), "gravity": 9.81, "density": 1.2, "R":13.2*10**(-3)}

    def cost_function(parameters: np.array, Holder: KinematicsSolutionHolder) -> np.array:

        PHI = parameters[0]
        freq = parameters[1]

        Holder = evaluate_angles_kinematics(number_time_steps, Holder, PHI)
        Holder = normal_workflow(Holder)
        Holder.force_QSM.set_referential(Referential.GLOBAL)

        force_z_dim = Holder.force_QSM.coords[2,:] * dim_parameters["density"] * dim_parameters["R"]**4 * freq**2
        power_dim = Holder.power * dim_parameters["density"] * dim_parameters["R"]**5 * freq**3

        K = (np.mean(force_z_dim) - dim_parameters["mass"] * dim_parameters["gravity"] * 0.5)**2 + np.mean(power_dim)**2

        return K

    initial_guess = np.random.rand(2)
    optimization = opt.minimize(cost_function, args=(Holder), x0=initial_guess)

    print(f"======{optimization.success}========")
    print(f"PHI = {optimization.x[0]*np.pi/180}")
    print(f"FREQ = {optimization.x[1]}")

    return Holder

