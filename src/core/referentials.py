from enum import Enum, auto
import numpy as np
from .angle import Angle
from .transform_func import (
    stroke_to_wing_matrix,
    global_to_body_matrix,
    global_to_wing_matrix,
    get_rotation_matrix_y,
    transpose
)


class Referential(Enum):
    GLOBAL = auto()
    WING = auto()
    BODY = auto()
    STROKE = auto()


class Transformations:
    _transformations = {}
    _is_initialized = False

    @staticmethod
    def initialize(
        phi: Angle,
        alpha: Angle,
        theta: Angle,
        eta: Angle,
        psi: Angle,
        beta: Angle,
        gamma: Angle,
    ):
        # check if the arguments are Angle objects
        if not (
            isinstance(phi, Angle)
            and isinstance(alpha, Angle)
            and isinstance(theta, Angle)
            and isinstance(eta, Angle)
            and isinstance(psi, Angle)
            and isinstance(beta, Angle)
            and isinstance(gamma, Angle)
        ):
            raise ValueError("Arguments must be Angle objects.")

        Transformations._transformations = {
            (Referential.STROKE, Referential.WING): stroke_to_wing_matrix(phi, alpha, theta),
            (Referential.GLOBAL, Referential.BODY): global_to_body_matrix(psi, beta, gamma),
            (Referential.GLOBAL, Referential.WING): global_to_wing_matrix(phi, alpha, theta, eta, psi, beta, gamma),
            (Referential.BODY, Referential.STROKE): get_rotation_matrix_y(eta)
        }
        Transformations._is_initialized = True

    @staticmethod
    def get_matrix(source: Referential, target: Referential):

        # check if the transformations are initialized
        if not Transformations._is_initialized:
            raise ValueError("Transformations must be initialized first.")
        
        # check if the referential is valid
        if not isinstance(source, Referential) or not isinstance(target, Referential):
            raise ValueError("Invalid referential type.")
        
        # check if the referential is the same
        # maybe useless but it's here just in case
        if source == target:
            return np.eye(3)
        
        # manage the inverse transformations
        if (source, target) not in Transformations._transformations:

            if (target, source) in Transformations._transformations:
                return transpose(Transformations._transformations[(target, source)])
            else:
                raise ValueError("Transformation not available.")

        return Transformations._transformations[(source, target)]
