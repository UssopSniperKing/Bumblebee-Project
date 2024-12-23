import pytest
import numpy as np
from src.core.transform_func import (
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
    stroke_to_wing_matrix,
    global_to_body_matrix,
    global_to_wing_matrix,
)
from src.core.angle import Angle

def test_get_rotation_matrix_x():
    angle = Angle([0, np.pi / 2, np.pi], unit="rad")
    rotation_matrix = get_rotation_matrix_x(angle)

    assert rotation_matrix.shape == (3, 3, 3)
    expected_matrix_0 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    expected_matrix_1 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    expected_matrix_2 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    assert np.allclose(rotation_matrix[:, :, 0], expected_matrix_0, atol=1e-8)
    assert np.allclose(rotation_matrix[:, :, 1], expected_matrix_1, atol=1e-8)
    assert np.allclose(rotation_matrix[:, :, 2], expected_matrix_2, atol=1e-8)

def test_get_rotation_matrix_y():
    angle = Angle([0, np.pi / 2, np.pi], unit="rad")
    rotation_matrix = get_rotation_matrix_y(angle)

    assert rotation_matrix.shape == (3, 3, 3)
    expected_matrix_0 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    expected_matrix_1 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    expected_matrix_2 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    assert np.allclose(rotation_matrix[:, :, 0], expected_matrix_0, atol=1e-8)
    assert np.allclose(rotation_matrix[:, :, 1], expected_matrix_1, atol=1e-8)
    assert np.allclose(rotation_matrix[:, :, 2], expected_matrix_2, atol=1e-8)

def test_get_rotation_matrix_z():
    angle = Angle([0, np.pi / 2, np.pi], unit="rad")
    rotation_matrix = get_rotation_matrix_z(angle)

    assert rotation_matrix.shape == (3, 3, 3)
    expected_matrix_0 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    expected_matrix_1 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    expected_matrix_2 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    assert np.allclose(rotation_matrix[:, :, 0], expected_matrix_0, atol=1e-8)
    assert np.allclose(rotation_matrix[:, :, 1], expected_matrix_1, atol=1e-8)
    assert np.allclose(rotation_matrix[:, :, 2], expected_matrix_2, atol=1e-8)

def test_stroke_to_wing_matrix():
    phi = Angle([0], unit="rad")
    alpha = Angle([np.pi / 4], unit="rad")
    theta = Angle([np.pi / 2], unit="rad")

    rotation_matrix = stroke_to_wing_matrix(phi, alpha, theta)
    assert rotation_matrix.shape == (3, 3)

def test_global_to_body_matrix():
    psi = Angle([np.pi / 6], unit="rad")
    beta = Angle([np.pi / 3], unit="rad")
    gamma = Angle([np.pi / 4], unit="rad")

    rotation_matrix = global_to_body_matrix(psi, beta, gamma)
    assert rotation_matrix.shape == (3, 3)

def test_global_to_wing_matrix():
    phi = Angle([0], unit="rad")
    alpha = Angle([np.pi / 4], unit="rad")
    theta = Angle([np.pi / 6], unit="rad")
    eta = Angle([np.pi / 3], unit="rad")
    psi = Angle([np.pi / 4], unit="rad")
    beta = Angle([np.pi / 6], unit="rad")
    gamma = Angle([np.pi / 3], unit="rad")

    rotation_matrix = global_to_wing_matrix(phi, alpha, theta, eta, psi, beta, gamma)
    assert rotation_matrix.shape == (3, 3)

    # Test shape with multiple angles
    phi = Angle([0, np.pi / 6], unit="rad")
    alpha = Angle([np.pi / 4, np.pi / 3], unit="rad")
    theta = Angle([np.pi / 6, np.pi / 4], unit="rad")
    eta = Angle([np.pi / 3, np.pi / 6], unit="rad")
    psi = Angle([np.pi / 4, np.pi / 3], unit="rad")
    beta = Angle([np.pi / 6, np.pi / 4], unit="rad")
    gamma = Angle([np.pi / 3, np.pi / 6], unit="rad")

    rotation_matrix = global_to_wing_matrix(phi, alpha, theta, eta, psi, beta, gamma)
    assert rotation_matrix.shape == (3, 3, 2)
