import pytest
import numpy as np
from src.core import Vector3D, Referential


@pytest.fixture
def single_vector():
    return Vector3D([1, 2, 3], Referential.GLOBAL)


@pytest.fixture
def multiple_vectors():
    return Vector3D([[1, 4], [2, 5], [3, 6]], Referential.GLOBAL)


def test_init_validation():
    # Valid initializations
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)
    assert v1.coords.shape == (3, 1)

    v2 = Vector3D([[1, 4], [2, 5], [3, 6]], Referential.GLOBAL)
    assert v2.coords.shape == (3, 2)

    v3 = Vector3D([[1, 2, 3]], Referential.GLOBAL)  # Shape (1, 3)
    assert v3.coords.shape == (3, 1)

    # Invalid initializations
    with pytest.raises(ValueError):
        Vector3D([1, 2], Referential.GLOBAL)  # Wrong length

    with pytest.raises(ValueError):
        Vector3D([[[1, 2, 3]]], Referential.GLOBAL)  # 3D array

    with pytest.raises(ValueError):
        Vector3D([1, 2, 3], "GLOBAL")  # Invalid referential


def test_properties(single_vector, multiple_vectors):
    # Test coords property
    assert single_vector.coords.shape == (3, 1)
    assert multiple_vectors.coords.shape == (3, 2)

    # Test referential property
    assert single_vector.referential == Referential.GLOBAL
    assert multiple_vectors.referential == Referential.GLOBAL


def test_norm(single_vector, multiple_vectors):
    # Single vector norm
    expected_norm = np.sqrt(14)  # sqrt(1^2 + 2^2 + 3^2)
    assert np.isclose(single_vector.norm(), expected_norm)

    # Multiple vectors norm
    expected_norms = np.array(
        [np.sqrt(14), np.sqrt(77)]
    )  # For vectors [1,2,3] and [4,5,6]
    assert np.allclose(multiple_vectors.norm(), expected_norms)


def test_scalar_operations(single_vector):
    # Multiplication
    result = single_vector * 2
    assert np.array_equal(result.coords, np.array([[2], [4], [6]]))
    assert result.referential == single_vector.referential

    # Division
    result = single_vector / 2
    assert np.array_equal(result.coords, np.array([[0.5], [1], [1.5]]))
    assert result.referential == single_vector.referential

    # Invalid operations
    with pytest.raises(ValueError):
        single_vector * "2"
    with pytest.raises(ValueError):
        single_vector / 0


def test_vector_operations():
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)
    v2 = Vector3D([4, 5, 6], Referential.GLOBAL)

    # Addition
    result = v1 + v2
    assert np.array_equal(result.coords, np.array([[5], [7], [9]]))

    # Subtraction
    result = v2 - v1
    assert np.array_equal(result.coords, np.array([[3], [3], [3]]))

    # Operations with different referentials
    v3 = Vector3D([1, 2, 3], Referential.WING)
    with pytest.raises(ValueError):
        v1 + v3
    with pytest.raises(ValueError):
        v1 - v3


def test_broadcasting_operations():
    # Test broadcasting between (3,1) and (3,N) shapes
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)  # Shape (3,1)
    v2 = Vector3D([[4, 7], [5, 8], [6, 9]], Referential.GLOBAL)  # Shape (3,2)

    # Addition with broadcasting
    result = v1 + v2
    expected = np.array([[5, 8], [7, 10], [9, 12]])
    assert np.array_equal(result.coords, expected)

    # Multiplication with scalar
    result = v2 * 2
    expected = np.array([[8, 14], [10, 16], [12, 18]])
    assert np.array_equal(result.coords, expected)


def test_len_operation(single_vector, multiple_vectors):
    assert len(single_vector) == 1
    assert len(multiple_vectors) == 2


def test_numpy_integration(single_vector):
    # Test numpy array conversion
    arr = np.asarray(single_vector)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 1)

    # Test ufunc operations
    result = np.sin(single_vector)
    assert isinstance(result, Vector3D)
    assert result.coords.shape == (3, 1)
    assert result.referential == single_vector.referential


def test_shape_combinations():
    # Test operations between different shapes
    v1 = Vector3D([[1], [2], [3]], Referential.GLOBAL)  # (3,1)
    v2 = Vector3D([[1, 4, 7], [2, 5, 8], [3, 6, 9]], Referential.GLOBAL)  # (3,3)

    # Addition
    result = v1 + v2
    expected = np.array([[2, 5, 8], [4, 7, 10], [6, 9, 12]])
    assert np.array_equal(result.coords, expected)

    # Test scalar multiplication with different shapes
    result = v2 * 2
    expected = np.array([[2, 8, 14], [4, 10, 16], [6, 12, 18]])
    assert np.array_equal(result.coords, expected)
