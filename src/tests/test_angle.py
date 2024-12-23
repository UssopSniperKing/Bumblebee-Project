import pytest
import numpy as np
from src.core.angle import Angle

def test_angle_initialization():
    # Test valid initialization
    angle_rad = Angle([0, np.pi / 2, np.pi], unit="rad")
    assert np.allclose(angle_rad.radians, [0, np.pi / 2, np.pi])
    assert angle_rad.degrees.shape == angle_rad.radians.shape

    angle_deg = Angle([0, 90, 180], unit="deg")
    assert np.allclose(angle_deg.degrees, [0, 90, 180])
    assert angle_deg.radians.shape == angle_deg.degrees.shape

    # Test multidimensional initialization
    angle_multi = Angle([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]], unit="rad")
    assert np.allclose(angle_multi.radians, [[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
    assert angle_multi.radians.shape == (2, 2)

    # Test invalid unit
    with pytest.raises(ValueError):
        Angle([0, 1], unit="invalid")

def test_angle_conversion():
    angle = Angle([0, 90, 180], unit="deg")
    assert np.allclose(angle.radians, [0, np.pi / 2, np.pi])

    angle.set_unit("rad")
    assert np.allclose(angle.radians, [0, np.pi / 2, np.pi])

    angle.set_unit("deg")
    assert np.allclose(angle.degrees, [0, 90, 180])

    with pytest.raises(ValueError):
        angle.set_unit("invalid")

    # Test conversion with multidimensional input
    angle_multi = Angle([[0, 90], [180, 270]], unit="deg")
    assert np.allclose(angle_multi.radians, [[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
    angle_multi.set_unit("rad")
    assert np.allclose(angle_multi.degrees, [[0, 90], [180, 270]])

def test_angle_operations():
    angle1 = Angle([0, 90, 180], unit="deg")
    angle2 = Angle([0, np.pi / 2, np.pi], unit="rad")

    angle_sum = angle1 + angle2
    assert np.allclose(angle_sum.radians, [0, np.pi, 2 * np.pi])

    angle_diff = angle1 - angle2
    assert np.allclose(angle_diff.radians, [0, 0, 0])

    # Test operations with multidimensional inputs
    angle_multi1 = Angle([[0, 90], [180, 270]], unit="deg")
    angle_multi2 = Angle([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]], unit="rad")

    angle_sum_multi = angle_multi1 + angle_multi2
    assert angle_sum_multi.radians.shape == (2, 2)
    assert np.allclose(angle_sum_multi.radians, [[0, np.pi], [2 * np.pi, 3 * np.pi]])

    angle_diff_multi = angle_multi1 - angle_multi2
    assert angle_diff_multi.radians.shape == (2, 2)
    assert np.allclose(angle_diff_multi.radians, [[0, 0], [0, 0]])

    with pytest.raises(ValueError):
        angle1 + [1, 2, 3]  # Adding non-Angle object

    with pytest.raises(ValueError):
        angle1 - [1, 2, 3]  # Subtracting non-Angle object

def test_angle_apply():
    angle = Angle([0, np.pi / 2, np.pi], unit="rad")
    applied_angle = angle.apply(np.sin)
    assert np.allclose(applied_angle.radians, [0, 1, 0])
    assert applied_angle.radians.shape == angle.radians.shape

    # Test apply with multidimensional input
    angle_multi = Angle([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]], unit="rad")
    applied_angle_multi = angle_multi.apply(np.cos)
    assert np.allclose(applied_angle_multi.radians, [[1, 0], [-1, 0]])
    assert applied_angle_multi.radians.shape == angle_multi.radians.shape

def test_angle_length():
    angle = Angle([0, 90, 180], unit="deg")
    assert len(angle) == 3

    # Test length with multidimensional input
    angle_multi = Angle([[0, 90], [180, 270]], unit="deg")
    assert len(angle_multi) == 2
