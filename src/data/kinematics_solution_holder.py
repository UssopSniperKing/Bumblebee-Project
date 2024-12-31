from dataclasses import dataclass
from src.core import Angle, Vector3D
import numpy as np


@dataclass
class KinematicsSolutionHolder:
    time: np.array
    alpha: Angle
    phi: Angle
    theta: Angle
    u_tip: Vector3D
    

    # todo : add the other important quantities to be computed

    def save(file_name: str):
        pass  # todo save as csv, or npy (binary)

    def load(file_name: str):
        pass  # todo : load a csv, or npy (binary)
