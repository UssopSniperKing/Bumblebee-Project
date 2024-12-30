import pytest
import numpy as np
from src.core import Angle

@pytest.fixture
def single_angle():
    return Angle(45, "deg")

@pytest.fixture
def multiple_angles():
    return Angle([0, 45, 90], "deg")

def test_init_validation():
    # Valid initializations
    a1 = Angle(45, "deg")
    assert isinstance(a1._values, np.ndarray)
    assert a1._values.size == 1
    
    a2 = Angle([0, 45, 90], "deg")
    assert a2._values.shape == (3,)

    a3 = Angle(np.array([0, 45, 90]), "deg")
    assert a3._values.shape == (3,)

    a4 = Angle([[0, 45], [90, 135]], "deg")
    assert a4._values.shape == (4,)
    
    # Invalid initializations
    with pytest.raises(ValueError):
        Angle(45, "invalid_unit")


def test_unit_conversion():
    # Test degree to radian conversion
    a1 = Angle(180, "deg")
    np.testing.assert_almost_equal(a1.radians, np.pi)
    
    # Test radian to degree conversion
    a2 = Angle(np.pi, "rad")
    np.testing.assert_almost_equal(a2.degrees, 180)
    
    # Test multiple angles conversion
    a3 = Angle([0, 90, 180], "deg")
    expected_rad = np.array([0, np.pi/2, np.pi])
    np.testing.assert_array_almost_equal(a3.radians, expected_rad)

def test_set_unit():
    a1 = Angle(90, "deg")
    
    # Test deg to rad conversion
    a1.set_unit("rad")
    assert a1._unit == "rad"
    np.testing.assert_almost_equal(a1._values, np.pi/2)
    
    # Test rad to deg conversion
    a1.set_unit("deg")
    assert a1._unit == "deg"
    np.testing.assert_almost_equal(a1._values, 90)
    
    # Test invalid unit
    with pytest.raises(ValueError):
        a1.set_unit("invalid")

def test_apply():
    angles = Angle([0, 90, 180], "deg")
    
    # Test sin function
    result = angles.apply(np.sin)
    expected = np.array([0, 1, 0])
    np.testing.assert_array_almost_equal(result.radians, expected)
    
    # Test custom function
    def double(x): return 2 * x
    result = angles.apply(double)
    expected = np.array([0, np.pi, 2*np.pi])
    np.testing.assert_array_almost_equal(result.radians, expected)

def test_arithmetic_operations():
    a1 = Angle(45, "deg")
    a2 = Angle(45, "deg")
    
    # Addition
    result = a1 + a2
    assert result._unit == "rad"
    np.testing.assert_almost_equal(result.degrees, 90)
    
    # Subtraction
    result = a1 - a2
    assert result._unit == "rad"
    np.testing.assert_almost_equal(result.degrees, 0)
    
    # Invalid operations
    with pytest.raises(ValueError):
        a1 + 45  # Can only add Angle objects
    with pytest.raises(ValueError):
        a1 - "45"  # Can only subtract Angle objects

def test_length_operations(single_angle, multiple_angles):
    assert isinstance(single_angle._values, np.ndarray)
    assert single_angle._values.size == 1
    assert len(multiple_angles) == 3

def test_broadcasting():
    # Test operations with different lengths
    a1 = Angle(45, "deg")
    a2 = Angle([0, 45, 90], "deg")
    
    # Addition
    result = a1 + a2
    expected = np.array([45, 90, 135])
    np.testing.assert_array_almost_equal(result.degrees, expected)
    
    # Subtraction
    result = a2 - a1
    expected = np.array([-45, 0, 45])
    np.testing.assert_array_almost_equal(result.degrees, expected)

def test_repr(single_angle):
    assert repr(single_angle) == f"Angle(values={single_angle._values}, unit='deg')"

def test_array_shapes():
    # Test scalar input
    a1 = Angle(45, "deg")
    assert isinstance(a1._values, np.ndarray)
    assert a1._values.size == 1
    
    # Test list input
    a2 = Angle([0, 45, 90], "deg")
    assert a2._values.shape == (3,)
    
    # Test numpy array input
    a3 = Angle(np.array([0, 45, 90]), "deg")
    assert a3._values.shape == (3,)