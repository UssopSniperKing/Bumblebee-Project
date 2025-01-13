import pytest
import numpy as np
from src.core import (
    Angle,
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
    stroke_to_wing_matrix,
    global_to_body_matrix,
    global_to_wing_matrix,
    transpose,
)


@pytest.fixture
def single_angle():
    return Angle(np.pi / 6, "rad")  # 30 degrees


@pytest.fixture
def array_angle():
    return Angle([0, np.pi / 6, np.pi / 4], "rad")  # 0, 30, 45 degrees


def test_get_rotation_matrix_x_single():
    # Test rotation of 30 degrees (pi/6 radians)
    angle = Angle(np.pi / 6, "rad")
    result = get_rotation_matrix_x(angle)

    # Expected rotation matrix around x-axis for 30 degrees
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])

    np.testing.assert_array_almost_equal(result, expected)


def test_get_rotation_matrix_y_single():
    # Test rotation of 30 degrees (pi/6 radians)
    angle = Angle(np.pi / 6, "rad")
    result = get_rotation_matrix_y(angle)

    # Expected rotation matrix around y-axis for 30 degrees
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    expected = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    np.testing.assert_array_almost_equal(result, expected)


def test_get_rotation_matrix_z_single():
    # Test rotation of 30 degrees (pi/6 radians)
    angle = Angle(np.pi / 6, "rad")
    result = get_rotation_matrix_z(angle)

    # Expected rotation matrix around z-axis for 30 degrees
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    expected = np.array([[c, s, 0.0], [0.0, 1.0, 0.0], [s, 0.0, c]])

    np.testing.assert_array_almost_equal(result, expected)


def test_get_rotation_matrix_x_array():
    # Test for [0, 30, 45] degrees
    angles = Angle([0, np.pi / 6, np.pi / 4], "rad")
    result = get_rotation_matrix_x(angles)

    # Expected matrices for each angle
    expected = np.empty((3, 3, 3))
    for i, angle in enumerate([0, np.pi / 6, np.pi / 4]):
        c, s = np.cos(angle), np.sin(angle)
        expected[:, :, i] = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])

    np.testing.assert_array_almost_equal(result, expected)


def test_stroke_to_wing_matrix_known_values():
    # Test with known angles: phi=30°, alpha=45°, theta=60°
    phi = Angle(np.pi / 6, "rad")  # 30°
    alpha = Angle(np.pi / 4, "rad")  # 45°
    theta = Angle(np.pi / 3, "rad")  # 60°

    result = stroke_to_wing_matrix(phi, alpha, theta)

    # Compute individual rotation matrices
    cx, sx = np.cos(np.pi / 6), np.sin(np.pi / 6)
    cy, sy = np.cos(np.pi / 4), np.sin(np.pi / 4)
    cz, sz = np.cos(np.pi / 3), np.sin(np.pi / 3)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])

    Ry = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]])

    Rz = np.array([[cz, sz, 0.0], [0.0, 1.0, 0.0], [sz, 0.0, cz]])

    # Expected result: Ry @ Rz @ Rx
    expected = Ry @ Rz @ Rx

    np.testing.assert_array_almost_equal(result, expected)


def test_stroke_to_wing_matrix_einsum_vs_matmul(single_angle):
    # Test with single angle using einsum and matmul
    phi = single_angle
    alpha = single_angle
    theta = single_angle

    result_einsum = stroke_to_wing_matrix(phi, alpha, theta)
    result_matmul = (
        get_rotation_matrix_y(alpha)
        @ get_rotation_matrix_z(theta)
        @ get_rotation_matrix_x(phi)
    )

    np.testing.assert_array_almost_equal(result_einsum, result_matmul)


def test_global_to_body_matrix_einsum_vs_matmul(single_angle):
    # Test with single angle using einsum and matmul
    psi = single_angle
    beta = single_angle
    gamma = single_angle

    result_einsum = global_to_body_matrix(psi, beta, gamma)
    result_matmul = (
        get_rotation_matrix_x(psi)
        @ get_rotation_matrix_y(beta)
        @ get_rotation_matrix_z(gamma)
    )

    np.testing.assert_array_almost_equal(result_einsum, result_matmul)


def test_global_to_wing_matrix_einsum_vs_matmul(single_angle):
    # Test with single angle using einsum and matmul
    psi = single_angle
    beta = single_angle
    gamma = single_angle
    eta = single_angle
    phi = single_angle
    alpha = single_angle
    theta = single_angle

    R_s2w = stroke_to_wing_matrix(phi, alpha, theta)
    R_g2b = global_to_body_matrix(psi, beta, gamma)
    R_b2s = get_rotation_matrix_y(eta)

    result_einsum = global_to_wing_matrix(psi, beta, gamma, eta, phi, alpha, theta)
    result_matmul = R_s2w @ R_b2s @ R_g2b

    np.testing.assert_array_almost_equal(result_einsum, result_matmul)


def test_global_to_body_matrix_known_values():
    # Test with known angles: psi=30°, beta=45°, gamma=60°
    psi = Angle(np.pi / 6, "rad")  # 30°
    beta = Angle(np.pi / 4, "rad")  # 45°
    gamma = Angle(np.pi / 3, "rad")  # 60°

    result = global_to_body_matrix(psi, beta, gamma)

    # Compute individual rotation matrices
    cx, sx = np.cos(np.pi / 6), np.sin(np.pi / 6)
    cy, sy = np.cos(np.pi / 4), np.sin(np.pi / 4)
    cz, sz = np.cos(np.pi / 3), np.sin(np.pi / 3)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])

    Ry = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]])

    Rz = np.array([[cz, sz, 0.0], [0.0, 1.0, 0.0], [sz, 0.0, cz]])

    # Expected result: Rx @ Ry @ Rz
    expected = Rx @ Ry @ Rz

    np.testing.assert_array_almost_equal(result, expected)


def test_global_to_body_matrix_array():
    # Test with array of angles: [30°, 45°, 60°] for each rotation
    psi = Angle([np.pi / 6, np.pi / 4, np.pi / 3], "rad")  # [30°, 45°, 60°]
    beta = Angle([np.pi / 6, np.pi / 4, np.pi / 3], "rad")  # [30°, 45°, 60°]
    gamma = Angle([np.pi / 6, np.pi / 4, np.pi / 3], "rad")  # [30°, 45°, 60°]

    result = global_to_body_matrix(psi, beta, gamma)

    # Compute expected matrices for each set of angles
    expected = np.empty((3, 3, 3))  # Shape: (3, 3, 3) for 3 sets of angles
    angles = [np.pi / 6, np.pi / 4, np.pi / 3]  # 30°, 45°, 60°

    for i, angle in enumerate(angles):
        # Individual rotation matrices for current angle
        cx, sx = np.cos(angle), np.sin(angle)

        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])

        Ry = np.array([[cx, 0.0, -sx], [0.0, 1.0, 0.0], [sx, 0.0, cx]])

        Rz = np.array([[cx, sx, 0.0], [0.0, 1.0, 0.0], [sx, 0.0, cx]])

        # Expected result for current angle: Rx @ Ry @ Rz
        expected[:, :, i] = Rx @ Ry @ Rz

    np.testing.assert_array_almost_equal(result, expected)


def test_stroke_to_wing_matrix_array():
    # Test with array of angles: [30°, 45°, 60°] for each rotation
    phi = Angle([np.pi / 6, np.pi / 4, np.pi / 3], "rad")  # [30°, 45°, 60°]
    alpha = Angle([np.pi / 6, np.pi / 4, np.pi / 3], "rad")  # [30°, 45°, 60°]
    theta = Angle([np.pi / 6, np.pi / 4, np.pi / 3], "rad")  # [30°, 45°, 60°]

    result = stroke_to_wing_matrix(phi, alpha, theta)

    # Compute expected matrices for each set of angles
    expected = np.empty((3, 3, 3))  # Shape: (3, 3, 3) for 3 sets of angles
    angles = [np.pi / 6, np.pi / 4, np.pi / 3]  # 30°, 45°, 60°

    for i, angle in enumerate(angles):
        # Individual rotation matrices for current angle
        cx, sx = np.cos(angle), np.sin(angle)

        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])

        Ry = np.array([[cx, 0.0, -sx], [0.0, 1.0, 0.0], [sx, 0.0, cx]])

        Rz = np.array([[cx, sx, 0.0], [0.0, 1.0, 0.0], [sx, 0.0, cx]])

        # Expected result for current angle: Ry @ Rz @ Rx
        expected[:, :, i] = np.einsum("ij,jk,kl->il", Ry, Rz, Rx)

    np.testing.assert_array_almost_equal(result, expected)


def test_mixed_length_arrays():
    # Test with mixed length arrays - one scalar angle and two array angles
    phi = Angle(np.pi / 6, "rad")  # Single angle (30°)
    alpha = Angle([np.pi / 4, np.pi / 3], "rad")  # Two angles [45°, 60°]
    theta = Angle([np.pi / 3, np.pi / 2], "rad")  # Two angles [60°, 90°]

    result = stroke_to_wing_matrix(phi, alpha, theta)

    # Expected shape should be (3, 3, 2) - broadcast to match longest input
    assert result.shape == (3, 3, 2)

    # Compute expected matrices
    expected = np.empty((3, 3, 2))
    phi_val = np.pi / 6
    alpha_vals = [np.pi / 4, np.pi / 3]
    theta_vals = [np.pi / 3, np.pi / 2]

    for i in range(2):
        # Rotation matrices for each set
        cx, sx = np.cos(phi_val), np.sin(phi_val)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])

        cy, sy = np.cos(alpha_vals[i]), np.sin(alpha_vals[i])
        Ry = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]])

        cz, sz = np.cos(theta_vals[i]), np.sin(theta_vals[i])
        Rz = np.array([[cz, sz, 0.0], [0.0, 1.0, 0.0], [sz, 0.0, cz]])

        expected[:, :, i] = Ry @ Rz @ Rx

    np.testing.assert_array_almost_equal(result, expected)


def test_transpose():
    # Test single matrix
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = transpose(matrix)
    expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    np.testing.assert_array_equal(result, expected)

    # Test array of matrices
    matrices = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
        ]
    )
    result = transpose(matrices)
    expected = np.array(
        [
            [[1, 2], [7, 8], [13, 14]],
            [[3, 4], [9, 10], [15, 16]],
            [[5, 6], [11, 12], [17, 18]],
        ]
    )
    np.testing.assert_array_equal(result, expected)


# Additional tests for error handling
def test_invalid_angle_type():
    with pytest.raises(ValueError, match="angle must be an Angle object"):
        get_rotation_matrix_x(0.5)


def test_stroke_to_wing_matrix_invalid_lengths():
    phi = Angle([0.1, 0.2], "rad")
    alpha = Angle([0.2, 0.3, 0.4], "rad")
    theta = Angle(0.3, "rad")
    with pytest.raises(
        ValueError, match="Angle arrays must either be of length 1 or same length"
    ):
        stroke_to_wing_matrix(phi, alpha, theta)


def test_transpose_invalid_shape():
    matrix = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Input matrix must be of shape"):
        transpose(matrix)
