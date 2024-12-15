import numpy as np
from typing import Union, Callable, TypeAlias

UFunc: TypeAlias = Callable[[np.ndarray], np.ndarray]

class Angle:

    def __init__(self, values: Union[float, list, np.ndarray], unit: str = "rad"):
        """Initialize an angle or set of angles

        Args:
            values (Union[float, list, np.ndarray]): Angles values (scalar or array)
            unit (str, optional): Angle unit "rad" or "deg". Defaults to "rad".
        """     

        if unit not in {"rad", "deg"}:
            raise ValueError("Angle unit must be 'deg' or 'rad'.")

        self.values = np.asarray(values, dtype=np.float64)
        self.unit = unit

    
    @property
    def radians(self) -> np.ndarray:
        """Give the angles in radians.

        Returns:
            np.ndarray: Angles in radians
        """
        if self.unit == "deg":
            return np.radians(self.values)
        else:
            return self.values


    @property
    def degrees(self) -> np.ndarray:
        """Give the angles in degrees.

        Returns:
            np.ndarray: Angles in degrees
        """
        if self.unit == "rad":
            return np.degrees(self.values)
        else:
            return self.values

    
    def apply(self, func: UFunc) -> "Angle":
        """Apply a function on the angles (in radians).

        Parameters:
            func (Callable[[np.ndarray], np.ndarray]): Function to apply.

        Returns:
            Angle: New angles.
        """
        transformed_values = func(self.radians)
        return Angle(transformed_values, unit="rad")

    
    def __repr__(self):
        unit = "rad" if self.unit == "rad" else "deg"
        return f"Angle(values={self.values}, unit='{unit}')"


    def __add__(self, angle: "Angle") -> "Angle":

        if isinstance(angle, Angle):
            result_values = self.radians + angle.radians
            return Angle(result_values, unit="rad")
        else:
            raise ValueError('You can only add angles to angles.')


    def __sub__(self, angle: "Angle") -> "Angle":

        if isinstance(angle, Angle):
            result_values = self.radians - angle.radians
            return Angle(result_values, unit="rad")
        else:
            raise ValueError('You can only substract angles to angles.')

        