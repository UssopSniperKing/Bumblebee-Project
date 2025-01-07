from core import Vector3D, Referential
import numpy as np

def force_TD(lift_coeff: np.array, omega_planar_wing: Vector3D, e_lift_global: Vector3D) -> Vector3D:
    """"""
    # todo : check referentials and types

    force_components = 0.5 * lift_coeff * omega_planar_wing.norm()**2 * e_lift_global.coords

    return Vector3D(force_components, Referential.GLOBAL)

def force_TC():
    pass

def force_RC():
    pass

def force_RD():
    pass

def force_AMx():
    pass

def force_AMz():
    pass