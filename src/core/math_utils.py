from .vector import Vector3D
import numpy as np
from .referentials import Referential


def time_derivative(time, vector: Vector3D) -> Vector3D:  # todo : add tests
    """Get the time derivative of a Vector3D object of shape (3,N),
    assuming it's a time series on N. A centered difference scheme of order 2 is used.
    First and last points are computed with a forward and backward difference scheme of order 1.

    Args:
        time (np.ndarray): Time array of shape (N,)
        vector (Vector3D): Vector3D object of shape (3,N)

    Returns:
        Vector3D: Time derivative of the input vector
    """

    # Check if the vector is a Vector3D object
    if not isinstance(vector, Vector3D):
        raise ValueError("Invalid vector type.")

    # Check if the time and vector shape match
    if vector.coords.shape[1] != time.shape[0]:
        raise ValueError("Time and vector shape mismatch.")

    # Check if the vector is in the GLOBAL referential
    if vector.referential != Referential.GLOBAL:
        raise ValueError("The vector must be in the GLOBAL referential.")

    dt = time[1] - time[0]  # Time step
    N = time.shape[0]  # Number of time steps
    vector_coords = np.empty((3, N))  # Empty array to store the time derivative

    # First point
    vector_coords[:, 0] = (vector.coords[:, 1] - vector.coords[:, 0]) / dt

    # Middle points
    for i in range(1, N - 1):
        vector_coords[:, i] = (vector.coords[:, i + 1] - vector.coords[:, i - 1]) / (2 * dt)

    # Last point
    vector_coords[:, N - 1] = (vector.coords[:, N - 1] - vector.coords[:, N - 2]) / dt

    return Vector3D(vector_coords, Referential.GLOBAL)


def cross(u: Vector3D, v: Vector3D) -> Vector3D:  # todo : add tests
    """Compute the cross product between a set of vectors or individual vectors.

    Args:
        u (Vector3D): First vector
        v (Vector3D): Second vector

    Returns:
        Vector3D: Set or single vector orthogonal to the given u, v vectors
    """
    if not isinstance(u, Vector3D) or not isinstance(v, Vector3D):
        raise ValueError("u, v must be a Vector3D object.")

    if u.referential != v.referential:
        raise ValueError("Referentials mismatch.")

    vec_coords = np.cross(u.coords, v.coords, axis=0)

    return Vector3D(vec_coords, u.referential)


def dot(u: Vector3D, v: Vector3D) -> np.ndarray:  # todo : add tests
    """Compute the term by term product of two vectors or two set
    of vectors (i.e. dot product)

    Args:
        u (Vector3D): First vector
        v (Vector3D): Second vector

    Returns:
        np.ndarray: Array of scalars
    """
    if not isinstance(u, Vector3D) or not isinstance(v, Vector3D):
        raise ValueError("u, v must be a Vector3D object.")

    if u.referential != v.referential:
        raise ValueError("Referentials mismatch.")

    return np.sum(u.coords * v.coords, axis=0)


def normalize(u: Vector3D) -> Vector3D:  # todo : add tests
    """Vector normalization function.

    Args:
        u (Vector3D): The vector to get it's normalized vector from.

    Returns:
        Vector3D: Set or single normalized vectors.
    """
    if not isinstance(u, Vector3D):
        raise ValueError("u, v must be a Vector3D object.")

    vec_coords = u.coords / u.norm()

    return Vector3D(vec_coords, u.referential)
