import numpy as np
from .angle import Angle


def get_rotation_matrix_z(angle: Angle) -> np.ndarray:
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
        rotation_matrices[:, :, i] = np.array(
            [[cosine[i], sine[i], 0], [-sine[i], cosine[i], 0], [0, 0, 1]]
        )

    if N == 1:
        return rotation_matrices[:, :, 0]  # return the rotation matrix of shape (3,3)

    return rotation_matrices  # return the rotation matrix of shape (3,3,N)


def get_rotation_matrix_y(angle: Angle) -> np.ndarray:
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
        rotation_matrices[:, :, i] = np.array(
            [[cosine[i], 0, -sine[i]], [0, 1, 0], [sine[i], 0, cosine[i]]]
        )

    if N == 1:
        return rotation_matrices[:, :, 0]  # return the rotation matrix of shape (3,3)

    return rotation_matrices  # return the rotation matrix of shape (3,3,N)


def get_rotation_matrix_x(angle: Angle) -> np.ndarray:
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
        rotation_matrices[:, :, i] = np.array(
            [[1, 0, 0], [0, cosine[i], sine[i]], [0, -sine[i], cosine[i]]]
        )

    if N == 1:
        return rotation_matrices[:, :, 0]  # return the rotation matrix of shape (3,3)

    return rotation_matrices  # return the rotation matrix of shape (3,3,N)


def stroke_to_wing_matrix(phi: Angle, alpha: Angle, theta: Angle) -> np.ndarray:
    """Get the rotation matrix from the stroke referential to the wing referential.
    Computes the rotation matrix as Ry(alpha) @ Rz(theta) @ Rx(phi).

    Args:
        phi (Angle): Rotation around the x-axis.
        alpha (Angle): Rotation around the y-axis.
        theta (Angle): Rotation around the z-axis.

    Raises:
        ValueError: If the angles are not Angle objects or if the angles arrays are not of the same length.

    Returns:
        np.ndarray: Rotation matrix of shape (3,3) or (3,3,N) depending on the number of angles.
    """

    # check if the angles are Angle objects
    for argument, name in [(phi, "phi"), (alpha, "alpha"), (theta, "theta")]:
        if not isinstance(argument, Angle):
            raise ValueError(f"{name} must be an Angle object")

    # check if the angles are of length 1 or same length
    lengths = [len(angle) for angle in (phi, alpha, theta)]
    max_len = max(lengths)

    if not all(length in (1, max_len) for length in lengths):
        raise ValueError("Angle arrays must either be of length 1 or same length")

    # broadcast the angles to the same length
    def broadcast_to_max_len(angle: Angle) -> np.ndarray:
        return np.broadcast_to(angle, (max_len,))

    phi_broadcasted = phi.apply(broadcast_to_max_len)
    alpha_broadcasted = alpha.apply(broadcast_to_max_len)
    theta_broadcasted = theta.apply(broadcast_to_max_len)

    # get the rotation matrices, at this point all matrices are of shape (3,3,max_len)
    # and max_len is either 1 or the length of the angles
    Rx = get_rotation_matrix_x(phi_broadcasted)
    Ry = get_rotation_matrix_y(alpha_broadcasted)
    Rz = get_rotation_matrix_z(theta_broadcasted)

    if max_len == 1:
        output_matrix = Ry @ Rz @ Rx
    else:
        output_matrix = np.empty((3, 3, max_len))

        # Use np.einsum with a single string to handle all indices directly
        output_matrix = np.einsum("ijn,jkn,kln->iln", Ry, Rz, Rx, optimize=True)

        # equivalent to the following loop but way faster
        # for i in range(max_len):
        #    output_matrix[:, :, i] = Ry[:, :, i] @ Rz[:, :, i] @ Rx[:, :, i]

    return output_matrix


def global_to_body_matrix(psi: Angle, beta: Angle, gamma: Angle) -> np.ndarray:
    """Get the rotation matrix from the global referential to the body referential.
    Computes the rotation matrix as Rx(psi) @ Ry(beta) @ Rz(gamma).

    Args:
        psi (Angle): Rotation around the x-axis.
        beta (Angle): Rotation around the y-axis.
        gamma (Angle): Rotation around the z-axis.

    Raises:
        ValueError: If the angles are not Angle objects or if the angles arrays are not of the same length.

    Returns:
        np.ndarray: Rotation matrix of shape (3,3) or (3,3,N) depending on the number of angles.
    """

    # check if the angles are Angle objects
    for argument, name in [(psi, "psi"), (beta, "beta"), (gamma, "gamma")]:
        if not isinstance(argument, Angle):
            raise ValueError(f"{name} must be an Angle object")

    # check if the angles are of length 1 or same length
    lengths = [len(angle) for angle in (psi, beta, gamma)]
    max_len = max(lengths)

    if not all(length in (1, max_len) for length in lengths):
        raise ValueError("Angle arrays must either be of length 1 or same length")

    # broadcast the angles to the same length
    def broadcast_to_max_len(angle: Angle) -> np.ndarray:
        return np.broadcast_to(angle, (max_len,))

    psi_broadcasted = psi.apply(broadcast_to_max_len)
    beta_broadcasted = beta.apply(broadcast_to_max_len)
    gamma_broadcasted = gamma.apply(broadcast_to_max_len)

    # get the rotation matrices, at this point all matrices are of shape (3,3,max_len)
    # and max_len is either 1 or the length of the angles
    Rx = get_rotation_matrix_x(psi_broadcasted)
    Ry = get_rotation_matrix_y(beta_broadcasted)
    Rz = get_rotation_matrix_z(gamma_broadcasted)

    if max_len == 1:
        output_matrix = Rx @ Ry @ Rz
    else:
        output_matrix = np.empty((3, 3, max_len))

        # Use np.einsum with a single string to handle all indices directly
        output_matrix = np.einsum("ijn,jkn,kln->iln", Rx, Ry, Rz, optimize=True)

        # equivalent to the following loop but way faster
        # for i in range(max_len):
        # output_matrix[:, :, i] = Rx[:, :, i] @ Ry[:, :, i] @ Rz[:, :, i]

    return output_matrix


def global_to_wing_matrix(phi, alpha, theta, eta, psi, beta, gamma) -> np.ndarray:
    """Get the rotation matrix from the global referential to the wing referential.
    Computes the rotation matrix as R_s2w @ R_b2s @ R_g2b.

    Args:
        phi (Angle): Rotation around the x-axis in the stroke referential.
        alpha (Angle): Rotation around the y-axis in the stroke referential.
        theta (Angle): Rotation around the z-axis in the stroke referential.
        eta (Angle): Rotation around the y-axis in the body referential.
        psi (Angle): Rotation around the x-axis in the body referential.
        beta (Angle): Rotation around the y-axis in the body referential.
        gamma (Angle): Rotation around the z-axis in the body referential.

    Raises:
        ValueError: If the angles are not Angle objects or if the angles arrays are not of the same length.

    Returns:
        np.ndarray: Rotation matrix of shape (3,3) or (3,3,N) depending on the number of angles.
    """

    # check if the angles are Angle objects
    for argument, name in [
        (phi, "phi"),
        (alpha, "alpha"),
        (theta, "theta"),
        (eta, "eta"),
        (psi, "psi"),
        (beta, "beta"),
        (gamma, "gamma"),
    ]:
        if not isinstance(argument, Angle):
            raise ValueError(f"{name} must be an Angle object")

    # check if the angles are of length 1 or same length
    lengths = [len(angle) for angle in (phi, alpha, theta, eta, psi, beta, gamma)]
    max_len = max(lengths)

    if not all(length in (1, max_len) for length in lengths):
        raise ValueError("Angle arrays must either be of length 1 or same length")

    # broadcast the angles to the same length
    def broadcast_to_max_len(angle: Angle) -> np.ndarray:
        return np.broadcast_to(angle, (max_len,))

    phi_broadcasted = phi.apply(broadcast_to_max_len)
    alpha_broadcasted = alpha.apply(broadcast_to_max_len)
    theta_broadcasted = theta.apply(broadcast_to_max_len)
    eta_broadcasted = eta.apply(broadcast_to_max_len)
    psi_broadcasted = psi.apply(broadcast_to_max_len)
    beta_broadcasted = beta.apply(broadcast_to_max_len)
    gamma_broadcasted = gamma.apply(broadcast_to_max_len)

    # get the rotation matrices, at this point all matrices are of shape (3,3,max_len)
    # and max_len is either 1 or the length of the angles
    R_s2w = stroke_to_wing_matrix(phi_broadcasted, alpha_broadcasted, theta_broadcasted)
    R_g2b = global_to_body_matrix(psi_broadcasted, beta_broadcasted, gamma_broadcasted)
    R_b2s = get_rotation_matrix_y(eta_broadcasted)

    if max_len == 1:
        output_matrix = R_s2w @ R_b2s @ R_g2b
    else:
        output_matrix = np.empty((3, 3, max_len))

        # Use np.einsum with a single string to handle all indices directly
        output_matrix = np.einsum(
            "ijn,jkn,kln->iln", R_s2w, R_b2s, R_g2b, optimize=True
        )

        # equivalent to the following loop but way faster
        # for i in range(max_len):
        #    output_matrix[:, :, i] = R_s2w[:, :, i] @ R_b2s[:, :, i] @ R_g2b[:, :, i]

    return output_matrix


def transpose(matrix: np.ndarray) -> np.ndarray:  # todo : add tests
    """Transpose the input matrix. If the input matrix is of shape (3,3,N), the function
    returns a matrix of shape (3,3,N).

    Args:
        matrix (np.ndarray): Input matrix of shape (3,3) or (3,3,N).

    Raises:
        ValueError: If the input matrix is not of shape (3,3) or (3,3,N).

    Returns:
        np.ndarray: Transposed matrix of shape (3,3) or (3,3,N).
    """
    if matrix.shape == (3, 3):
        return matrix.T
    elif matrix.ndim == 3:
        return matrix.transpose((1, 0, 2))
    else:
        raise ValueError("Input matrix must be of shape (3,3) or (3,3,N).")
