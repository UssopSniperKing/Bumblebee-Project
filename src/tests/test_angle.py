import numpy as np
import numpy.testing as npt
import math
from src.core.angle import Angle
import pytest


# Pytest Test Cases
def test_init_with_scalar():
    """Test initialization with a scalar value"""
    # Test radian initialization
    angle_rad = Angle(math.pi, unit="rad")
    assert angle_rad.unit == "rad"
    npt.assert_almost_equal(angle_rad.values, math.pi)

    # Test degree initialization
    angle_deg = Angle(180, unit="deg")
    assert angle_deg.unit == "deg"
    npt.assert_almost_equal(angle_deg.values, 180)


def test_init_with_list():
    """Test initialization with a list of values"""
    angle_rad = Angle([0, math.pi / 2, math.pi], unit="rad")
    npt.assert_almost_equal(angle_rad.values, [0, math.pi / 2, math.pi])
    assert angle_rad.unit == "rad"


def test_init_with_ndarray():
    """Test initialization with a numpy array of values"""
    angle_rad = Angle(np.array([0, math.pi / 2, math.pi]), unit="rad")
    npt.assert_almost_equal(angle_rad.values, np.array([0, math.pi / 2, math.pi]))
    assert angle_rad.unit == "rad"


def test_init_invalid_unit():
    """Test that invalid unit raises a ValueError"""
    with pytest.raises(ValueError):
        Angle(45, unit="grad")


def test_radians_property():
    """Test the radians property conversion"""
    # From degrees to radians
    angle_deg = Angle(180, unit="deg")
    npt.assert_almost_equal(angle_deg.radians, math.pi)

    # Already in radians
    angle_rad = Angle(math.pi, unit="rad")
    npt.assert_almost_equal(angle_rad.radians, math.pi)


def test_degrees_property():
    """Test the degrees property conversion"""
    # From radians to degrees
    angle_rad = Angle(math.pi, unit="rad")
    npt.assert_almost_equal(angle_rad.degrees, 180)

    # Already in degrees
    angle_deg = Angle(180, unit="deg")
    npt.assert_almost_equal(angle_deg.degrees, 180)


def test_apply_method():
    """Test the apply method with numpy functions"""
    angle = Angle([0, math.pi / 2, math.pi], unit="rad")

    # Test sine function
    sine_result = angle.apply(np.sin)
    npt.assert_almost_equal(sine_result.values, [0, 1, 0])
    assert sine_result.unit == "rad"

    # Test cosine function
    cosine_result = angle.apply(np.cos)
    npt.assert_almost_equal(cosine_result.values, [1, 0, -1])
    assert cosine_result.unit == "rad"


def test_repr_method():
    """Test the __repr__ method with flexible matching"""
    # Test for radian representation
    angle_rad = Angle(math.pi, unit="rad")
    repr_rad = repr(angle_rad)
    assert "Angle" in repr_rad
    assert "values=" in repr_rad
    assert str(math.pi) in repr_rad
    assert "unit='rad'" in repr_rad

    # Test for degree representation
    angle_deg = Angle(180, unit="deg")
    repr_deg = repr(angle_deg)
    assert "Angle" in repr_deg
    assert "values=" in repr_deg
    assert "180" in repr_deg  # Works for both 180 and 180.0
    assert "unit='deg'" in repr_deg


def test_add_method():
    """Test angle addition"""
    # Test addition of angles in the same unit
    angle1 = Angle(math.pi / 2, unit="rad")
    angle2 = Angle(math.pi / 2, unit="rad")
    result = angle1 + angle2
    npt.assert_almost_equal(result.values, math.pi)
    assert result.unit == "rad"

    # Test addition of angles in different units
    angle_deg = Angle(180, unit="deg")
    angle_rad = Angle(math.pi / 2, unit="rad")
    result = angle_deg + angle_rad
    npt.assert_almost_equal(result.values, 3 * math.pi / 2)
    assert result.unit == "rad"


def test_sub_method():
    """Test angle subtraction"""
    # Test subtraction of angles in the same unit
    angle1 = Angle(math.pi, unit="rad")
    angle2 = Angle(math.pi / 2, unit="rad")
    result = angle1 - angle2
    npt.assert_almost_equal(result.values, math.pi / 2)
    assert result.unit == "rad"


def test_invalid_add_sub():
    """Test that adding or subtracting non-Angle objects raises an error"""
    angle = Angle(math.pi, unit="rad")

    with pytest.raises(ValueError):
        angle + 45

    with pytest.raises(ValueError):
        angle - 45
