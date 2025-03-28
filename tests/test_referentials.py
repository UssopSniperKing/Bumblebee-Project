import pytest
import numpy as np
from src.core import Referential, Transformations, Angle


@pytest.fixture
def simple_angles():
    return {
        "phi": Angle(0.1, "rad"),
        "alpha": Angle(0.2, "rad"),
        "theta": Angle(0.3, "rad"),
        "eta": Angle(0.4, "rad"),
        "psi": Angle(0.5, "rad"),
        "beta": Angle(0.6, "rad"),
        "gamma": Angle(0.7, "rad"),
    }


@pytest.fixture
def array_angles():
    return {
        "phi": Angle([0.1, 0.2], "rad"),
        "alpha": Angle([0.2, 0.3], "rad"),
        "theta": Angle([0.3, 0.4], "rad"),
        "eta": Angle([0.4, 0.5], "rad"),
        "psi": Angle([0.5, 0.6], "rad"),
        "beta": Angle([0.6, 0.7], "rad"),
        "gamma": Angle([0.7, 0.8], "rad"),
    }


def test_referential_enum():
    assert isinstance(Referential.GLOBAL, Referential)
    assert isinstance(Referential.WING, Referential)
    assert isinstance(Referential.BODY, Referential)
    assert isinstance(Referential.STROKE, Referential)


def test_transformations_not_initialized():
    with pytest.raises(ValueError, match="Transformations must be initialized first"):
        Transformations.get_matrix(Referential.GLOBAL, Referential.WING)


def test_transformations_initialize_invalid_types():
    with pytest.raises(ValueError, match="Arguments must be Angle objects"):
        Transformations.initialize(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def test_transformations_initialize(simple_angles):
    Transformations.initialize(**simple_angles)
    assert Transformations._is_initialized


def test_get_matrix_invalid_referential(simple_angles):
    Transformations.initialize(**simple_angles)
    with pytest.raises(ValueError, match="Invalid referential type"):
        Transformations.get_matrix("GLOBAL", Referential.WING)


def test_get_matrix_same_referential(simple_angles):
    Transformations.initialize(**simple_angles)
    result = Transformations.get_matrix(Referential.GLOBAL, Referential.GLOBAL)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result, np.eye(3))


def test_transformation_roundtrip(simple_angles):
    """Test that transforming a point from STROKE to WING and back preserves coordinates"""
    Transformations.initialize(**simple_angles)

    # Test points (unit vectors for simplicity)
    test_points = np.eye(3)  # [[1,0,0], [0,1,0], [0,0,1]]

    for point in test_points:
        # Transform to WING
        transformed = (
            Transformations.get_matrix(Referential.GLOBAL, Referential.WING) @ point
        )
        # Transform back to GLOBAL
        roundtrip = (
            Transformations.get_matrix(Referential.WING, Referential.GLOBAL)
            @ transformed
        )
        # Check if we get back approximately the same point
        np.testing.assert_allclose(roundtrip, point, rtol=1e-5, atol=1e-5)


def test_length_preservation(simple_angles):
    """Test that transformations preserve vector lengths"""
    Transformations.initialize(**simple_angles)

    test_vector = np.array([1.0, 2.0, 3.0])
    original_length = np.linalg.norm(test_vector)

    transformed = (
        Transformations.get_matrix(Referential.GLOBAL, Referential.WING) @ test_vector
    )
    transformed_length = np.linalg.norm(transformed)

    np.testing.assert_allclose(transformed_length, original_length, rtol=1e-5)


def test_get_matrix_stroke_to_wing_shape(simple_angles):
    Transformations.initialize(**simple_angles)
    result = Transformations.get_matrix(Referential.STROKE, Referential.WING)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_get_matrix_stroke_to_wing_array_shape(array_angles):
    Transformations.initialize(**array_angles)
    result = Transformations.get_matrix(Referential.STROKE, Referential.WING)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3, 2)


def test_get_matrix_global_to_body_shape(simple_angles):
    Transformations.initialize(**simple_angles)
    result = Transformations.get_matrix(Referential.GLOBAL, Referential.BODY)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_get_matrix_global_to_body_array_shape(array_angles):
    Transformations.initialize(**array_angles)
    result = Transformations.get_matrix(Referential.GLOBAL, Referential.BODY)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3, 2)


def test_get_matrix_global_to_wing_shape(simple_angles):
    Transformations.initialize(**simple_angles)
    result = Transformations.get_matrix(Referential.GLOBAL, Referential.WING)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_get_matrix_global_to_wing_array_shape(array_angles):
    Transformations.initialize(**array_angles)
    result = Transformations.get_matrix(Referential.GLOBAL, Referential.WING)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3, 2)


def test_get_matrix_body_to_stroke_shape(simple_angles):
    Transformations.initialize(**simple_angles)
    result = Transformations.get_matrix(Referential.BODY, Referential.STROKE)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_get_matrix_body_to_stroke_array_shape(array_angles):
    Transformations.initialize(**array_angles)
    result = Transformations.get_matrix(Referential.BODY, Referential.STROKE)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3, 2)


def test_mixed_shapes(simple_angles, array_angles):
    # Test mixing single and array angles
    mixed_angles = simple_angles.copy()
    mixed_angles["phi"] = array_angles["phi"]

    Transformations.initialize(**mixed_angles)
    result = Transformations.get_matrix(Referential.STROKE, Referential.WING)
    assert result.shape == (3, 3, 2)


def test_unavailable_transformation(simple_angles):
    Transformations.initialize(**simple_angles)
    with pytest.raises(ValueError, match="Transformation not available"):
        Transformations.get_matrix(Referential.WING, Referential.BODY)
