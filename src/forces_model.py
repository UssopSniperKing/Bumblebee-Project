from core import Vector3D, Referential, Scalar
import numpy as np


def force_TC(
    lift_coeff: np.array, omega_planar_wing: Vector3D, e_lift_global: Vector3D
) -> Vector3D:
    """"""
    # todo : check referentials and types

    force_components = (
        0.5 * lift_coeff * omega_planar_wing.norm() ** 2 * e_lift_global.coords
    )

    return Vector3D(force_components, Referential.GLOBAL)


def force_TD(
    drag_coeff: np.array, omega_planar_wing: Vector3D, e_drag_global: Vector3D
) -> Vector3D:
    """"""
    # todo : check referentials and types

    force_components = (
        0.5 * drag_coeff * omega_planar_wing.norm() ** 2 * e_drag_global.coords
    )

    return Vector3D(force_components, Referential.GLOBAL)


def force_RC(
    u_tip_global: Vector3D, omega_wing: Vector3D, ez_global: Vector3D, C_RC: Scalar
):
    """"""
    # todo : check referentials and types

    force_components = (
        C_RC * u_tip_global.norm() * omega_wing.coords[1, :] * ez_global.coords
    )

    return Vector3D(force_components, Referential.GLOBAL)


def force_RD(omega_wing: Vector3D, ez_global: Vector3D, C_RD: Scalar):
    """"""
    # todo : check referentials and types

    force_components = (
        (-1 / 6)
        * C_RD
        * np.abs(omega_wing.coords[1, :])
        * omega_wing.coords[1, :]
        * ez_global.coords
    )

    return Vector3D(force_components, Referential.GLOBAL)


def force_AMx(
    u_tip_dt_wing: Vector3D, ex_global: Vector3D, C_AMX1: Scalar, C_AMX2: Scalar
):
    force_components = (
        C_AMX1 * u_tip_dt_wing.coords[0, :] + C_AMX2 * u_tip_dt_wing.coords[2, :]
    ) * ex_global.coords

    return Vector3D(force_components, Referential.GLOBAL)


def force_AMz(
    u_tip_dt_wing: Vector3D,
    omega_dt_wing: Vector3D,
    ez_global: Vector3D,
    C_AMZ1: Scalar,
    C_AMZ2: Scalar,
    C_AMZ3: Scalar,
    C_AMZ4: Scalar,
    C_AMZ5: Scalar,
    C_AMZ6: Scalar,
):
    """"""
    # todo : check referentials and types

    force_components = (
        C_AMZ1 * u_tip_dt_wing.coords[0, :]
        + C_AMZ2 * u_tip_dt_wing.coords[1, :]
        + C_AMZ3 * u_tip_dt_wing.coords[2, :]
        + C_AMZ4 * omega_dt_wing.coords[0, :]
        + C_AMZ5 * omega_dt_wing.coords[1, :]
        + C_AMZ6 * omega_dt_wing.coords[2, :]
    ) * ez_global.coords

    return Vector3D(force_components, Referential.GLOBAL)
