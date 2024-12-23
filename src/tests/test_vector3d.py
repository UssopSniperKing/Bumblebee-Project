import pytest
import numpy as np
from src.core.vector import Vector3D
from src.core.referentials import Referential
from src.core.types import Scalar


def test_vector_creation():
    # Test valid 1D array
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)
    assert v1.coords.shape == (3, 1)
    assert v1.referential == Referential.GLOBAL

    # Test valid 2D array (3, N)
    v2 = Vector3D([[1, 2], [3, 4], [5, 6]], Referential.BODY)
    assert v2.coords.shape == (3, 2)
    assert v2.referential == Referential.BODY

    # Test valid 2D array (N, 3)
    v3 = Vector3D([[1, 2, 3], [4, 5, 6]], Referential.WING)
    assert v3.coords.shape == (3, 2)
    assert v3.referential == Referential.WING

    # Test invalid referential
    with pytest.raises(ValueError):
        Vector3D([1, 2, 3], "INVALID")

    # Test invalid array shape
    with pytest.raises(ValueError):
        Vector3D([1, 2], Referential.GLOBAL)

    # Test invalid array type
    with pytest.raises(ValueError):
        Vector3D("invalid", Referential.BODY)

def test_vector_properties():
    v = Vector3D([1, 2, 3], Referential.GLOBAL)
    assert np.allclose(v.coords, np.array([[1], [2], [3]]))
    assert v.referential == Referential.GLOBAL

def test_vector_norm():
    v = Vector3D([[3, 4], [0, 0], [0, 0]], Referential.WING)
    assert np.allclose(v.norm(), [3, 4])

def test_vector_addition():
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)
    v2 = Vector3D([4, 5, 6], Referential.GLOBAL)
    v_sum = v1 + v2
    assert v_sum.coords.shape == (3, 1)
    assert np.allclose(v_sum.coords, np.array([[5], [7], [9]]))

    # Test referential mismatch
    v3 = Vector3D([7, 8, 9], Referential.BODY)
    with pytest.raises(ValueError):
        v1 + v3

def test_vector_subtraction():
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)
    v2 = Vector3D([4, 5, 6], Referential.GLOBAL)
    v_diff = v1 - v2
    assert v_diff.coords.shape == (3, 1)
    assert np.allclose(v_diff.coords, np.array([[-3], [-3], [-3]]))

    # Test referential mismatch
    v3 = Vector3D([7, 8, 9], Referential.BODY)
    with pytest.raises(ValueError):
        v1 - v3

def test_vector_multiplication():
    v = Vector3D([1, 2, 3], Referential.GLOBAL)
    v_scaled = v * 2
    assert v_scaled.coords.shape == (3, 1)
    assert np.allclose(v_scaled.coords, np.array([[2], [4], [6]]))

    # Test invalid scalar type
    with pytest.raises(ValueError):
        v * "invalid"

def test_vector_division():
    v = Vector3D([2, 4, 6], Referential.GLOBAL)
    v_scaled = v / 2
    assert v_scaled.coords.shape == (3, 1)
    assert np.allclose(v_scaled.coords, np.array([[1], [2], [3]]))

    # Test division by zero
    with pytest.raises(ValueError):
        v / 0

    # Test invalid scalar type
    with pytest.raises(ValueError):
        v / "invalid"

def test_vector_length():
    v = Vector3D([[1, 2], [3, 4], [5, 6]], Referential.WING)
    assert len(v) == 2

def test_vector_numpy_integration():
    v = Vector3D([[1, 2], [3, 4], [5, 6]], Referential.GLOBAL)
    # Test np.asarray
    array = np.asarray(v)
    assert array.shape == (3, 2)
    assert np.allclose(array, v.coords)

    # Test NumPy ufunc (e.g., np.sin)
    sin_v = np.sin(v)
    assert isinstance(sin_v, Vector3D)
    assert sin_v.coords.shape == (3, 2)
