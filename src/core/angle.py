import numpy as np
from .types import UFunc, ArrayLike


class Angle:
    def __init__(self, values: ArrayLike, unit: str = "rad"):
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
    
    def set_unit(self, unit: str) -> None: # todo : add tests
        """Set the unit of the angles.

        Parameters:
            unit (str): Unit to set ("rad" or "deg").
        """
        if unit not in {"rad", "deg"}:
            raise ValueError("Angle unit must be 'deg' or 'rad'.")

        if self.unit != unit:
            self.values = self.radians if unit == "rad" else self.degrees
            self.unit = unit

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
            raise ValueError("You can only add angles to angles.")

    def __sub__(self, angle: "Angle") -> "Angle":
        if isinstance(angle, Angle):
            result_values = self.radians - angle.radians
            return Angle(result_values, unit="rad")
        else:
            raise ValueError("You can only substract angles to angles.")
    
    def __len__(self) -> int:
        """Return the number of angles."""
        return len(self.values)
