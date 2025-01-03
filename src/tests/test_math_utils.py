import pytest
import numpy as np
from src.core import Vector3D, Referential, vector_time_derivative, cross, dot, normalize

@pytest.fixture
def time_array():
    return np.array([0, 0.1, 0.2, 0.3])

@pytest.fixture
def vector_time_series():
    coords = np.array([[1, 2, 3, 4],
                      [2, 3, 4, 5],
                      [3, 4, 5, 6]])
    return Vector3D(coords, Referential.GLOBAL)

def test_vector_time_derivative_validation(time_array, vector_time_series):
    # Invalid vector type
    with pytest.raises(ValueError, match="Invalid vector type"):
        vector_time_derivative(time_array, "not_a_vector")
    
    # Time and vector shape mismatch
    wrong_time = np.array([0, 0.1, 0.2])
    with pytest.raises(ValueError, match="Time and vector shape mismatch"):
        vector_time_derivative(wrong_time, vector_time_series)
    
    # Non-global referential
    local_vector = Vector3D([[1], [2], [3]], Referential.WING)
    with pytest.raises(ValueError, match="The vector must be in the GLOBAL referential"):
        vector_time_derivative(np.array([0]), local_vector)

def test_vector_time_derivative_calculation(time_array, vector_time_series):
    result = vector_time_derivative(time_array, vector_time_series)
    
    # Check type and referential
    assert isinstance(result, Vector3D)
    assert result.referential == Referential.GLOBAL
    
    # Check shape
    assert result.coords.shape == vector_time_series.coords.shape
    
    # First point (forward difference)
    expected_first = np.array([[10], [10], [10]])  # (point2 - point1)/dt
    np.testing.assert_array_almost_equal(
        result.coords[:, 0:1], expected_first
    )
    
    # Middle points (centered difference)
    dt = 0.1
    for i in range(1, len(time_array)-1):
        expected = (vector_time_series.coords[:, i+1] - vector_time_series.coords[:, i-1]) / (2*dt)
        np.testing.assert_array_almost_equal(
            result.coords[:, i], expected
        )
    
    # Last point (backward difference)
    expected_last = np.array([[10], [10], [10]])  # (pointN - pointN-1)/dt
    np.testing.assert_array_almost_equal(
        result.coords[:, -1:], expected_last
    )

def test_cross_product_validation():
    v1 = Vector3D([1, 0, 0], Referential.GLOBAL)
    
    # Invalid vector type
    with pytest.raises(ValueError, match="must be a Vector3D object"):
        cross(v1, "not_a_vector")
    with pytest.raises(ValueError, match="must be a Vector3D object"):
        cross("not_a_vector", v1)
    
    # Referential mismatch
    v2 = Vector3D([0, 1, 0], Referential.WING)
    with pytest.raises(ValueError, match="Referentials mismatch"):
        cross(v1, v2)

def test_cross_product_calculation():
    # Single vectors
    v1 = Vector3D([1, 0, 0], Referential.GLOBAL)
    v2 = Vector3D([0, 1, 0], Referential.GLOBAL)
    result = cross(v1, v2)
    
    assert isinstance(result, Vector3D)
    assert result.referential == Referential.GLOBAL
    np.testing.assert_array_almost_equal(
        result.coords, np.array([[0], [0], [1]])
    )
    
    # Multiple vectors
    v3 = Vector3D([[1, 2], [0, 0], [0, 0]], Referential.GLOBAL)
    v4 = Vector3D([[0, 0], [1, 1], [0, 0]], Referential.GLOBAL)
    result = cross(v3, v4)
    expected = np.array([[0, 0], [0, 0], [1, 2]])
    np.testing.assert_array_almost_equal(result.coords, expected)

def test_dot_product_validation():
    v1 = Vector3D([1, 0, 0], Referential.GLOBAL)
    
    # Invalid vector type
    with pytest.raises(ValueError, match="must be a Vector3D object"):
        dot(v1, "not_a_vector")
    with pytest.raises(ValueError, match="must be a Vector3D object"):
        dot("not_a_vector", v1)
    
    # Referential mismatch
    v2 = Vector3D([0, 1, 0], Referential.WING)
    with pytest.raises(ValueError, match="Referentials mismatch"):
        dot(v1, v2)

def test_dot_product_calculation():
    # Single vectors
    v1 = Vector3D([1, 0, 0], Referential.GLOBAL)
    v2 = Vector3D([1, 1, 0], Referential.GLOBAL)
    result = dot(v1, v2)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, np.array([1]))
    
    # Multiple vectors
    v3 = Vector3D([[1, 2], [1, 0], [0, 1]], Referential.GLOBAL)
    v4 = Vector3D([[1, 1], [1, 0], [1, 0]], Referential.GLOBAL)
    result = dot(v3, v4)
    expected = np.array([2, 2])
    np.testing.assert_array_almost_equal(result, expected)

def test_normalize_validation():
    # Invalid vector type
    with pytest.raises(ValueError, match="must be a Vector3D object"):
        normalize("not_a_vector")


def test_normalize_calculation():
    # Single vector
    v1 = Vector3D([3, 0, 0], Referential.GLOBAL)
    result = normalize(v1)
    assert isinstance(result, Vector3D)
    assert result.referential == v1.referential
    np.testing.assert_array_almost_equal(
        result.coords, np.array([[1], [0], [0]])
    )
    
    # Multiple vectors
    v2 = Vector3D([[3, 0], [0, 4], [0, 0]], Referential.GLOBAL)
    result = normalize(v2)
    expected = np.array([[1, 0], [0, 1], [0, 0]])
    np.testing.assert_array_almost_equal(result.coords, expected)
    
    # Check unit length
    assert np.allclose(np.linalg.norm(result.coords, axis=0), 1.0)
