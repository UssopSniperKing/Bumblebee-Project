from data import KinematicsSolutionHolder
from core import Transformations
from core import Angle

def initialize_transformations(Holder: KinematicsSolutionHolder) -> None:
    
    # Initialize some of the angles to zero, this may change 
    # and another implementation may be used.
    Holder.eta = Angle(-90, "deg")
    Holder.psi = Angle(0.0, "rad")
    Holder.beta = Angle(0.0, "rad")
    Holder.gamma = Angle(180, "deg")

    angles = {
        "phi": Holder.phi,
        "alpha": Holder.alpha,
        "theta": Holder.theta,
        "eta": Holder.eta,
        "psi": Holder.psi,
        "beta": Holder.beta,
        "gamma": Holder.gamma,
    }
    
    Transformations.initialize(**angles)
    return None