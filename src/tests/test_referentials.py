import pytest
import numpy as np
from src.core.referentials import Referential, Transformations
from src.core.angle import Angle
from src.core.vector import Vector3D

def test_referential_enum():
    # Test if referentials are correctly defined
    assert Referential.GLOBAL.name == "GLOBAL"
    assert Referential.WING.name == "WING"
    assert Referential.BODY.name == "BODY"
    assert Referential.STROKE.name == "STROKE"

def test_transformations_initialization():
    phi = Angle([0], unit="rad")
    alpha = Angle([0], unit="rad")
    theta = Angle([0], unit="rad")
    eta = Angle([0], unit="rad")
    psi = Angle([0], unit="rad")
    beta = Angle([0], unit="rad")
    gamma = Angle([0], unit="rad")

    # Initialize transformations
    Transformations.initialize(phi, alpha, theta, eta, psi, beta, gamma)
    assert Transformations._is_initialized

    # Test invalid arguments
    with pytest.raises(ValueError):
        Transformations.initialize("invalid", alpha, theta, eta, psi, beta, gamma)

def test_get_matrix():
    phi = Angle([0], unit="rad")
    alpha = Angle([0], unit="rad")
    theta = Angle([0], unit="rad")
    eta = Angle([0], unit="rad")
    psi = Angle([0], unit="rad")
    beta = Angle([0], unit="rad")
    gamma = Angle([0], unit="rad")

    Transformations.initialize(phi, alpha, theta, eta, psi, beta, gamma)

    # Test identity matrix for same source and target
    matrix = Transformations.get_matrix(Referential.GLOBAL, Referential.GLOBAL)
    assert np.allclose(matrix, np.eye(3))

    # Test invalid referential
    with pytest.raises(ValueError):
        Transformations.get_matrix("invalid", Referential.WING)

    # Test uninitialized transformations
    Transformations._is_initialized = False
    with pytest.raises(ValueError):
        Transformations.get_matrix(Referential.GLOBAL, Referential.WING)
    Transformations._is_initialized = True

def test_inverse_transformation():
    phi = Angle([0], unit="rad")
    alpha = Angle([0], unit="rad")
    theta = Angle([0], unit="rad")
    eta = Angle([0], unit="rad")
    psi = Angle([0], unit="rad")
    beta = Angle([0], unit="rad")
    gamma = Angle([0], unit="rad")

    Transformations.initialize(phi, alpha, theta, eta, psi, beta, gamma)

    # Test inverse transformation
    matrix = Transformations.get_matrix(Referential.GLOBAL, Referential.BODY)
    inverse_matrix = Transformations.get_matrix(Referential.BODY, Referential.GLOBAL)
    assert np.allclose(matrix.T, inverse_matrix)

def test_transformation_with_vector():
    phi = Angle([0], unit="rad")
    alpha = Angle([0], unit="rad")
    theta = Angle([0], unit="rad")
    eta = Angle([0], unit="rad")
    psi = Angle([0], unit="rad")
    beta = Angle([0], unit="rad")
    gamma = Angle([0], unit="rad")

    Transformations.initialize(phi, alpha, theta, eta, psi, beta, gamma)

    # Create a vector in GLOBAL referential
    vector = Vector3D([1, 0, 0], Referential.GLOBAL)

    # Transform it to BODY referential
    matrix = Transformations.get_matrix(Referential.GLOBAL, Referential.BODY)
    transformed_coords = matrix @ vector.coords
    assert transformed_coords.shape == (3, 1)
