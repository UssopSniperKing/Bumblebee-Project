import numpy as np
from angle import Angle

def get_rotation_matrix_z(angle: Angle) -> np.ndarray: # todo : add tests
    """Get the rotation matrix around the z-axis. If multiple angles are given, 
    the function returns a (3,3,N) array of rotation matrices.

    Args:
        angle (Angle): Angle object representing the rotation around the z-axis.

    Raises:
        ValueError: If the angle is not an Angle object.

    Returns:
        np.ndarray: Rotation matrix of shape (3,3) or (3,3,N) depending on the number of angles.
    """    
    if not isinstance(angle, Angle):
        raise ValueError("angle must be an Angle object.")

    cosine = np.cos(angle.radians)
    sine = np.sin(angle.radians)

    N = len(angle)
    rotation_matrices = np.empty((3, 3, N))

    for i in range(N):
        rotation_matrices[:, :, i] = np.array([[cosine[i], sine[i], 0],
                                               [0, 1, 0],
                                               [sine[i], 0, cosine[i]]])
    
    if N == 1:
        return rotation_matrices[:, :, 0] # return the rotation matrix of shape (3,3)

    return rotation_matrices # return the rotation matrix of shape (3,3,N)
         


def get_rotation_matrix_y(angle: Angle) -> np.ndarray:  # todo : add tests
    """Get the rotation matrix around the y-axis. If multiple angles are given, 
    the function returns a (3,3,N) array of rotation matrices.

    Args:
        angle (Angle): Angle object representing the rotation around the y-axis.

    Raises:
        ValueError: If the angle is not an Angle object.

    Returns:
        np.ndarray: Rotation matrix of shape (3,3) or (3,3,N) depending on the number of angles.
    """  
    if not isinstance(angle, Angle):
        raise ValueError("angle must be an Angle object.")

    cosine = np.cos(angle.radians)
    sine = np.sin(angle.radians)

    N = len(angle)
    rotation_matrices = np.empty((3, 3, N))

    for i in range(N):
        rotation_matrices[:, :, i] = np.array([[cosine[i], 0, -sine[i]],
                                               [0, 1, 0],
                                               [sine[i], 0, cosine[i]]])
    
    if N == 1:
        return rotation_matrices[:, :, 0] # return the rotation matrix of shape (3,3)
    
    return rotation_matrices # return the rotation matrix of shape (3,3,N)


def get_rotation_matrix_x(angle: Angle) -> np.ndarray:  # todo : add tests
    """Get the rotation matrix around the x-axis. If multiple angles are given, 
    the function returns a (3,3,N) array of rotation matrices.

    Args:
        angle (Angle): Angle object representing the rotation around the x-axis.

    Raises:
        ValueError: If the angle is not an Angle object.

    Returns:
        np.ndarray: Rotation matrix of shape (3,3) or (3,3,N) depending on the number of angles.
    """  
    if not isinstance(angle, Angle):
        raise ValueError("angle must be an Angle object.")
    
    cosine = np.cos(angle.radians)
    sine = np.sin(angle.radians)

    N = len(angle)
    rotation_matrices = np.empty((3, 3, N))

    for i in range(N):
        rotation_matrices[:, :, i] = np.array([[1, 0, 0],
                                               [0, cosine[i], sine[i]],
                                               [0, -sine[i], cosine[i]]])
    
    if N == 1:
        return rotation_matrices[:, :, 0] # return the rotation matrix of shape (3,3)
    
    return rotation_matrices # return the rotation matrix of shape (3,3,N)


def wing_to_global_matrix() -> np.ndarray: # todo : use einsum to multiply the rotation matrices (need to understand einsum first)
    pass


def body_to_wing_matrix() -> np.ndarray:
    pass


def stroke_to_body_matrix() -> np.ndarray:
    pass
