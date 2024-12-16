import numpy as np
from .referentials import Referential
from .types import ArrayLike, UFunc
from typing import Optional


class Vector3D:
    
    def __init__(self, array: ArrayLike, referential: Referential):
        # check if we have a valid referential
        if isinstance(referential, Referential):
            self._referential = referential
        else:
            raise ValueError("Invalid Referential")

        # check if we have an array_like
        if not isinstance(array, (list, tuple, np.ndarray)):
            raise ValueError("Invalid Type")

        array = np.array(array, dtype=float)

        if array.ndim == 1:  # If we have a 1D array
            if len(array) != 3:
                raise ValueError(
                    f"Invalid length for 1D array: Provided {len(array)}. Expected 3"
                )

            self._coords = array.reshape(3, 1)

        elif array.ndim == 2:  # If we have a 2D array
            if array.shape[0] == 3:  # Shape (3, N) -> keep it
                self._coords = array
            elif array.shape[1] == 3:  # Shape (N, 3) -> transpose
                self._coords = np.transpose(array)
            else:
                raise ValueError(
                    f"Invalid shape: {array.shape}. Expected (3,N) or (N,3)"
                )
        else:
            raise ValueError(
                f"Invalid array dimension: {array.ndim}D. Expected 1D or 2D"
            )

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def referential(self) -> Referential:
        return self._referential

    @referential.setter
    def referential(self, new_referential: Optional[Referential]) -> None:
        if isinstance(new_referential, Referential):
            self._referential = new_referential
        else:
            raise ValueError("Invalid Referential")

    def __mul__(self, scalar: float) -> "Vector3D":
        new_coords = self.coords * scalar
        return Vector3D(new_coords, self.referential)

    def __sub__(self, vector: "Vector3D") -> "Vector3D":  # todo
        pass

    def __add__(self, vector: "Vector3D") -> "Vector3D":  # todo
        pass

    def __repr__(self) -> str:
        return f"Vector3D(Coordinates=\n{self.coords}, referential={self.referential})"

    def __len__(self) -> int:
        """Number N of vectors in the array (3, N)."""
        return self.coords.shape[1]

    def __array__(self, dtype: type = None) -> np.ndarray:
        """
        Support for np.asarray().
        """
        return self.coords if dtype is None else self.coords.astype(dtype)

    # Intégration avec les ufuncs NumPy
    def __array_ufunc__(self, ufunc: UFunc, method, *inputs, **kwargs):
        """
        Permet l'utilisation directe des ufuncs NumPy comme np.sin ou np.linalg.norm.

        Parameters:
            ufunc: La ufunc NumPy (par ex. np.sin, np.exp).
            method: Méthode appelée sur la ufunc.
            inputs: Entrées passées à la ufunc.
            kwargs: Arguments additionnels.
        """
        coords = [arg.coords if isinstance(arg, Vector3D) else arg for arg in inputs]
        result = getattr(ufunc, method)(*coords, **kwargs)
        if isinstance(result, np.ndarray) and result.ndim >= 2 and result.shape[0] == 3:
            return Vector3D(result, frame=self.frame)
        return result
