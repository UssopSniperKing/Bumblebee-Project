from enum import Enum, auto
from .transform_func import stroke_to_body_matrix, body_to_wing_matrix, wing_to_global_matrix

class Referential(Enum):
    GLOBAL = auto()
    WING = auto()
    BODY = auto()
    STROKE = auto()


class Transformations(): # todo : add tests
 
    _transformations = {}
    _is_initialized = False

    @staticmethod
    def initialize(): # todo : add angles as arguments, perform checks on the angles
        Transformations._transformations = {
            (Referential.STROKE, Referential.BODY): stroke_to_body_matrix(),
            (Referential.BODY, Referential.WING): body_to_wing_matrix(),
            (Referential.WING, Referential.GLOBAL): wing_to_global_matrix()
        }
        Transformations._is_initialized = True

    @staticmethod
    def get_matrix(source: Referential, target: Referential): # todo : build something more robust
        return Transformations._transformations[(source, target)]
