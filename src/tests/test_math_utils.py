import pytest
import numpy as np
from src.core.math_utils import time_derivative, cross, dot, normalize
from src.core.vector import Vector3D
from src.core.referentials import Referential

def test_time_derivative():
    time = np.array([0, 1, 2, 3])
    coords = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    vector = Vector3D(coords, Referential.GLOBAL)

    derivative = time_derivative(time, vector)
    expected_coords = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert isinstance(derivative, Vector3D)
    assert np.allclose(derivative.coords, expected_coords)
    assert derivative.referential == Referential.GLOBAL

    # Test shape mismatch
    invalid_time = np.array([0, 1])
    with pytest.raises(ValueError):
        time_derivative(invalid_time, vector)

    # Test invalid referential
    vector_body = Vector3D(coords, Referential.BODY)
    with pytest.raises(ValueError):
        time_derivative(time, vector_body)

def test_cross_product():
    coords_u = np.array([[1, 0], [0, 1], [0, 0]])
    coords_v = np.array([[0, 1], [1, 0], [0, 0]])
    u = Vector3D(coords_u, Referential.GLOBAL)
    v = Vector3D(coords_v, Referential.GLOBAL)

    result = cross(u, v)
    expected_coords = np.array([[0, 0], [0, 0], [1, -1]])
    assert isinstance(result, Vector3D)
    assert np.allclose(result.coords, expected_coords)
    assert result.referential == Referential.GLOBAL

    # Test referential mismatch
    v_body = Vector3D(coords_v, Referential.BODY)
    with pytest.raises(ValueError):
        cross(u, v_body)

def test_dot_product():
    coords_u = np.array([[1, 2], [3, 4], [5, 6]])
    coords_v = np.array([[6, 5], [4, 3], [2, 1]])
    u = Vector3D(coords_u, Referential.GLOBAL)
    v = Vector3D(coords_v, Referential.GLOBAL)

    result = dot(u, v)
    expected_result = np.array([28, 28])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected_result)

    # Test referential mismatch
    v_body = Vector3D(coords_v, Referential.BODY)
    with pytest.raises(ValueError):
        dot(u, v_body)

def test_normalize():
    coords = np.array([[3, 1], [4, 0], [0, 2]])
    vector = Vector3D(coords, Referential.GLOBAL)

    result = normalize(vector)
    expected_coords = np.array([[0.6, 0.4472136], [0.8, 0], [0, 0.89442719]])
    assert isinstance(result, Vector3D)
    assert np.allclose(result.coords, expected_coords)
    assert result.referential == Referential.GLOBAL

    # Test zero vector
    zero_coords = np.array([[0, 0], [0, 0], [0, 0]])
    zero_vector = Vector3D(zero_coords, Referential.GLOBAL)
    with pytest.raises(ZeroDivisionError):
        normalize(zero_vector)
