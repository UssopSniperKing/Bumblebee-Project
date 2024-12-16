import pytest
import numpy as np
from src.core.referentials import Referential
from src.core.vector import Vector3D


def test_vector3d_init_valid_1d_array():
    """Test initialization with a 1D array of 3 elements."""
    v = Vector3D([1, 2, 3], Referential.GLOBAL)

    assert isinstance(v.coords, np.ndarray)
    assert v.coords.shape == (3, 1)
    assert np.array_equal(v.coords, np.array([[1], [2], [3]]))
    assert v.referential == Referential.GLOBAL


def test_vector3d_init_valid_2d_array_3xn():
    """Test initialization with a 2D array of shape (3, N)."""
    input_array = np.array([[1, 4], [2, 5], [3, 6]])
    v = Vector3D(input_array, Referential.WING)

    assert v.coords.shape == (3, 2)
    assert np.array_equal(v.coords, input_array)
    assert v.referential == Referential.WING


def test_vector3d_init_valid_2d_array_nx3():
    """Test initialization with a 2D array of shape (N, 3)."""
    input_array = np.array([[1, 2, 3], [4, 5, 6]])
    v = Vector3D(input_array, Referential.BODY)

    assert v.coords.shape == (3, 2)
    assert np.array_equal(v.coords, input_array.T)
    assert v.referential == Referential.BODY


def test_vector3d_invalid_referential():
    """Test initialization with an invalid referential."""
    with pytest.raises(ValueError, match="Invalid Referential"):
        Vector3D([1, 2, 3], "not a referential")


def test_vector3d_invalid_array_type():
    """Test initialization with an invalid array type."""
    with pytest.raises(ValueError, match="Invalid Type"):
        Vector3D(42, Referential.STROKE)


def test_vector3d_invalid_array_length():
    """Test initialization with an incorrect array length."""
    with pytest.raises(ValueError, match="Invalid length for 1D array"):
        Vector3D([1, 2], Referential.GLOBAL)


def test_vector3d_invalid_array_dimension():
    """Test initialization with an invalid array dimension."""
    with pytest.raises(ValueError, match="Invalid array dimension"):
        Vector3D(np.zeros((2, 2, 2)), Referential.WING)


def test_vector3d_invalid_array_shape():
    """Test initialization with an invalid array shape."""
    with pytest.raises(ValueError, match="Invalid shape"):
        Vector3D(np.zeros((4, 4)), Referential.BODY)


def test_vector3d_referential_setter():
    """Test the referential setter method."""
    v = Vector3D([1, 2, 3], Referential.GLOBAL)

    v.referential = Referential.WING
    assert v.referential == Referential.WING


def test_vector3d_referential_setter_invalid():
    """Test the referential setter with an invalid referential."""
    v = Vector3D([1, 2, 3], Referential.GLOBAL)

    with pytest.raises(ValueError, match="Invalid Referential"):
        v.referential = "invalid"


def test_vector3d_multiplication():
    """Test vector multiplication by a scalar."""
    v = Vector3D([1, 2, 3], Referential.BODY)
    result = v * 2

    assert isinstance(result, Vector3D)
    assert np.array_equal(result.coords, np.array([[2], [4], [6]]))
    assert result.referential == Referential.BODY


def test_vector3d_len():
    """Test the __len__ method."""
    v1 = Vector3D([1, 2, 3], Referential.GLOBAL)
    assert len(v1) == 1

    v2 = Vector3D(np.array([[1, 4], [2, 5], [3, 6]]), Referential.WING)
    assert len(v2) == 2


def test_vector3d_array_conversion():
    """Test conversion to NumPy array."""
    v = Vector3D([1, 2, 3], Referential.STROKE)

    np_array = np.asarray(v)
    assert np.array_equal(np_array, v.coords)


# Note: Cannot fully test __sub__ and __add__ as they are not implemented (todo)


def test_vector3d_repr():
    """Test the string representation of Vector3D."""
    v = Vector3D([1, 2, 3], Referential.WING)

    repr_str = repr(v)
    assert "Vector3D" in repr_str
    assert "Coordinates" in repr_str
    assert "referential" in repr_str
    assert str(Referential.WING) in repr_str
