"""
quaternion_ops.py — Thin wrapper over C-optimised quaternion backends.

This module provides a clean, consistent project-level API for quaternion
operations while delegating all heavy computation to:
  • numpy-quaternion  (C-extension for quaternion arithmetic)
  • scipy.spatial.transform  (Rotation, Slerp)

Convention: scalar-first [w, x, y, z] throughout this module.
scipy internally uses scalar-last [x, y, z, w] — conversions are handled
transparently in every function that touches scipy.
"""

import quaternion  # numpy-quaternion package (C backend)
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerance for near-zero checks
# ---------------------------------------------------------------------------
_EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. create
# ---------------------------------------------------------------------------
def create(w: float, x: float, y: float, z: float) -> np.quaternion:
    """Create a single quaternion from scalar components.

    Args:
        w: Scalar (real) part.
        x: First imaginary component.
        y: Second imaginary component.
        z: Third imaginary component.

    Returns:
        A numpy quaternion object.
    """
    # Delegates to the C-backed np.quaternion constructor.
    return np.quaternion(w, x, y, z)


# ---------------------------------------------------------------------------
# 2. from_array
# ---------------------------------------------------------------------------
def from_array(arr: np.ndarray) -> Union[np.quaternion, np.ndarray]:
    """Convert a float array to quaternion(s).

    Args:
        arr: Array of shape (4,) for a single quaternion or (N, 4) for a batch.
             Component order is [w, x, y, z].

    Returns:
        A single ``np.quaternion`` when *arr* has shape (4,), or a numpy array
        of quaternions when *arr* has shape (N, 4).

    Raises:
        ValueError: If *arr* does not have a trailing dimension of size 4.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] != 4:
        raise ValueError(f"Expected array of length 4, got {arr.shape[0]}")
    if arr.ndim == 2 and arr.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array, got shape {arr.shape}")
    # C-backed conversion from float array to quaternion dtype.
    return quaternion.from_float_array(arr)


# ---------------------------------------------------------------------------
# 3. to_array
# ---------------------------------------------------------------------------
def to_array(q: Union[np.quaternion, np.ndarray]) -> np.ndarray:
    """Convert quaternion(s) to a float array.

    Args:
        q: A single quaternion or a numpy array of quaternions.

    Returns:
        A float array of shape (4,) or (N, 4) with component order [w, x, y, z].
    """
    # C-backed conversion from quaternion dtype to float array.
    return quaternion.as_float_array(q)


# ---------------------------------------------------------------------------
# 4. hamilton_product
# ---------------------------------------------------------------------------
def hamilton_product(
    q1: Union[np.quaternion, np.ndarray],
    q2: Union[np.quaternion, np.ndarray],
) -> Union[np.quaternion, np.ndarray]:
    """Compute the Hamilton product of two quaternion(s).

    Args:
        q1: Left operand (single quaternion or array).
        q2: Right operand (single quaternion or array).

    Returns:
        The Hamilton product ``q1 * q2``.  Note: quaternion multiplication
        is **not** commutative.
    """
    # C-optimised Hamilton product via operator overload.
    return q1 * q2


# ---------------------------------------------------------------------------
# 5. conjugate
# ---------------------------------------------------------------------------
def conjugate(
    q: Union[np.quaternion, np.ndarray],
) -> Union[np.quaternion, np.ndarray]:
    """Return the conjugate of quaternion(s).

    For q = w + xi + yj + zk, the conjugate is w - xi - yj - zk.

    Args:
        q: A single quaternion or array of quaternions.

    Returns:
        The conjugated quaternion(s).
    """
    # np.conjugate dispatches to the C-backed quaternion conjugate.
    return np.conjugate(q)


# ---------------------------------------------------------------------------
# 6. norm
# ---------------------------------------------------------------------------
def norm(q: Union[np.quaternion, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute the norm (magnitude) of quaternion(s).

    Args:
        q: A single quaternion or array of quaternions.

    Returns:
        The L2 norm as a float (single) or ndarray (batch).
    """
    # np.abs dispatches to the C-backed quaternion absolute value.
    return np.abs(q)


# ---------------------------------------------------------------------------
# 7. normalize
# ---------------------------------------------------------------------------
def normalize(
    q: Union[np.quaternion, np.ndarray],
) -> Union[np.quaternion, np.ndarray]:
    """Normalise quaternion(s) to unit length.

    If the norm is near zero the identity quaternion (1, 0, 0, 0) is returned
    instead, avoiding division-by-zero.

    Args:
        q: A single quaternion or array of quaternions.

    Returns:
        Unit quaternion(s).
    """
    n = np.abs(q)  # C-backed norm

    if np.ndim(n) == 0:
        # Single quaternion
        if n < _EPS:
            logger.warning("Quaternion norm near zero; returning identity.")
            return np.quaternion(1.0, 0.0, 0.0, 0.0)
        return q / n

    # Array of quaternions — element-wise safety check
    safe = n > _EPS
    result = np.empty_like(q)
    identity = np.quaternion(1.0, 0.0, 0.0, 0.0)

    for i in range(len(q)):
        if safe[i]:
            result[i] = q[i] / n[i]
        else:
            logger.warning("Quaternion at index %d has near-zero norm; using identity.", i)
            result[i] = identity

    return result


# ---------------------------------------------------------------------------
# 8. inverse
# ---------------------------------------------------------------------------
def inverse(
    q: Union[np.quaternion, np.ndarray],
) -> Union[np.quaternion, np.ndarray]:
    """Compute the multiplicative inverse of quaternion(s).

    For a quaternion q the inverse is conjugate(q) / norm(q)².
    If the norm is near zero the identity quaternion is returned.

    Args:
        q: A single quaternion or array of quaternions.

    Returns:
        The inverse quaternion(s).
    """
    conj = np.conjugate(q)  # C-backed conjugate
    n = np.abs(q)            # C-backed norm

    if np.ndim(n) == 0:
        n2 = n * n
        if n2 < _EPS:
            logger.warning("Quaternion norm near zero; returning identity for inverse.")
            return np.quaternion(1.0, 0.0, 0.0, 0.0)
        return conj / n2

    # Batch path
    n2 = n * n
    safe = n2 > _EPS
    result = np.empty_like(q)
    identity = np.quaternion(1.0, 0.0, 0.0, 0.0)

    for i in range(len(q)):
        if safe[i]:
            result[i] = conj[i] / n2[i]
        else:
            logger.warning("Quaternion at index %d has near-zero norm; using identity.", i)
            result[i] = identity

    return result


# ---------------------------------------------------------------------------
# 9. exp
# ---------------------------------------------------------------------------
def exp(q: Union[np.quaternion, np.ndarray]) -> Union[np.quaternion, np.ndarray]:
    """Compute the quaternion exponential.

    Args:
        q: A single quaternion or array of quaternions.

    Returns:
        exp(q) computed by the C backend.
    """
    # C-backed quaternion exponential.
    return np.exp(q)


# ---------------------------------------------------------------------------
# 10. log
# ---------------------------------------------------------------------------
def log(q: Union[np.quaternion, np.ndarray]) -> Union[np.quaternion, np.ndarray]:
    """Compute the quaternion logarithm.

    Args:
        q: A single quaternion or array of quaternions.

    Returns:
        log(q) computed by the C backend.
    """
    # C-backed quaternion logarithm.
    return np.log(q)


# ---------------------------------------------------------------------------
# Helper: scalar-first ↔ scalar-last conversion
# ---------------------------------------------------------------------------
def _wxyz_to_xyzw(arr: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] to scipy's [x, y, z, w] convention."""
    if arr.ndim == 1:
        return arr[[1, 2, 3, 0]]
    return arr[:, [1, 2, 3, 0]]


def _xyzw_to_wxyz(arr: np.ndarray) -> np.ndarray:
    """Convert scipy's [x, y, z, w] back to [w, x, y, z]."""
    if arr.ndim == 1:
        return arr[[3, 0, 1, 2]]
    return arr[:, [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# 11. slerp
# ---------------------------------------------------------------------------
def slerp(q1: np.quaternion, q2: np.quaternion, t: float) -> np.quaternion:
    """Spherical linear interpolation between two quaternions.

    Uses ``scipy.spatial.transform.Slerp`` for a robust, C-optimised
    interpolation.

    CRITICAL: scipy expects scalar-last [x, y, z, w] while numpy-quaternion
    uses scalar-first [w, x, y, z].  The conversion is handled here.

    Args:
        q1: Start quaternion.
        q2: End quaternion.
        t: Interpolation parameter in [0, 1].  0 → q1, 1 → q2.

    Returns:
        Interpolated unit quaternion.
    """
    # Convert to float arrays in [w,x,y,z] order via C backend …
    arr1 = quaternion.as_float_array(q1)  # shape (4,), [w,x,y,z]
    arr2 = quaternion.as_float_array(q2)

    # … then reorder to scipy's [x,y,z,w]
    xyzw1 = _wxyz_to_xyzw(arr1)
    xyzw2 = _wxyz_to_xyzw(arr2)

    # Build scipy Rotation objects and the Slerp interpolator
    key_rots = Rotation.from_quat(np.array([xyzw1, xyzw2]))  # scalar-last
    interpolator = Slerp([0.0, 1.0], key_rots)

    # Interpolate — result is a Rotation object
    interp_rot = interpolator([t])

    # Extract quaternion in [x,y,z,w] and convert back to [w,x,y,z]
    xyzw_result = interp_rot.as_quat()[0]  # scalar-last
    wxyz_result = _xyzw_to_wxyz(xyzw_result)

    # Convert back to numpy-quaternion via C backend
    return quaternion.from_float_array(wxyz_result)


# ---------------------------------------------------------------------------
# 12. rotate_vector
# ---------------------------------------------------------------------------
def rotate_vector(q: np.quaternion, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-D vector by a unit quaternion.

    Uses ``scipy.spatial.transform.Rotation`` for the actual rotation (BLAS /
    C-optimised path under the hood).

    Args:
        q: A unit quaternion representing the rotation.
        v: A 3-D vector (shape (3,)) or batch of vectors (shape (N, 3)).

    Returns:
        The rotated vector(s), same shape as *v*.
    """
    # Convert to [w,x,y,z] float array then reorder to scipy [x,y,z,w]
    wxyz = quaternion.as_float_array(q)
    xyzw = _wxyz_to_xyzw(wxyz)

    # scipy Rotation.apply handles single and batch vectors
    rot = Rotation.from_quat(xyzw)  # scalar-last convention
    return rot.apply(np.asarray(v, dtype=np.float64))


# ---------------------------------------------------------------------------
# 13. batch_from_array
# ---------------------------------------------------------------------------
def batch_from_array(arr_2d: np.ndarray) -> np.ndarray:
    """Convert an (N, 4) float array to a numpy array of quaternions.

    Args:
        arr_2d: Float array of shape (N, 4) with component order [w, x, y, z].

    Returns:
        A 1-D numpy array of quaternion dtype, length N.

    Raises:
        ValueError: If *arr_2d* is not 2-D or the second dimension is not 4.
    """
    arr_2d = np.asarray(arr_2d, dtype=np.float64)
    if arr_2d.ndim != 2 or arr_2d.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array, got shape {arr_2d.shape}")
    # C-backed batch conversion.
    return quaternion.from_float_array(arr_2d)


# ---------------------------------------------------------------------------
# 14. batch_to_array
# ---------------------------------------------------------------------------
def batch_to_array(q_arr: np.ndarray) -> np.ndarray:
    """Convert a numpy array of quaternions to an (N, 4) float array.

    Args:
        q_arr: 1-D numpy array with quaternion dtype.

    Returns:
        Float array of shape (N, 4) with component order [w, x, y, z].
    """
    # C-backed batch conversion.
    return quaternion.as_float_array(q_arr)


# ---------------------------------------------------------------------------
# 15. relative_rotation
# ---------------------------------------------------------------------------
def relative_rotation(
    q1: Union[np.quaternion, np.ndarray],
    q2: Union[np.quaternion, np.ndarray],
) -> Union[np.quaternion, np.ndarray]:
    """Compute the relative rotation from q1 to q2.

    The result is ``inverse(q1) * q2``, i.e. the rotation that, when applied
    *after* q1, yields q2.  Useful for computing path deltas.

    Args:
        q1: Reference quaternion(s).
        q2: Target quaternion(s).

    Returns:
        The relative rotation quaternion(s).
    """
    # Uses inverse() and hamilton_product(), both C-backed.
    return hamilton_product(inverse(q1), q2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("quaternion_ops  —  demo")
    print("=" * 60)

    # --- Create ---
    q1 = create(1.0, 0.0, 0.0, 0.0)
    q2 = create(0.0, 1.0, 0.0, 0.0)
    print(f"\nq1 (identity)  : {q1}")
    print(f"q2 (pure-i)    : {q2}")

    # --- Hamilton product ---
    q12 = hamilton_product(q1, q2)
    print(f"\nq1 * q2        : {q12}")

    # --- Norm & normalise ---
    q_unnorm = create(1.0, 2.0, 3.0, 4.0)
    print(f"\nUnnormalised    : {q_unnorm}   norm = {norm(q_unnorm):.6f}")
    q_normed = normalize(q_unnorm)
    print(f"Normalised      : {q_normed}   norm = {norm(q_normed):.6f}")

    # --- Inverse ---
    q_inv = inverse(q_normed)
    product = hamilton_product(q_normed, q_inv)
    print(f"\nInverse         : {q_inv}")
    print(f"q * q_inv       : {product}   (should ≈ identity)")

    # --- Slerp ---
    qa = normalize(create(1.0, 0.0, 0.0, 0.0))
    qb = normalize(create(0.0, 0.0, 1.0, 0.0))
    print(f"\nSlerp endpoints : {qa}  →  {qb}")
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        qs = slerp(qa, qb, t_val)
        print(f"  t={t_val:.2f} : {qs}")

    # --- Rotate vector ---
    q_rot = normalize(create(0.7071, 0.7071, 0.0, 0.0))  # ~90° around x
    v = np.array([0.0, 1.0, 0.0])
    v_rot = rotate_vector(q_rot, v)
    print(f"\nRotate {v} by ~90° around x-axis:")
    print(f"  Result: {v_rot}")

    # --- Relative rotation ---
    r1 = normalize(create(1.0, 0.0, 0.0, 0.0))
    r2 = normalize(create(0.7071, 0.7071, 0.0, 0.0))
    delta = relative_rotation(r1, r2)
    print(f"\nRelative rotation (identity → r2): {delta}")

    print("\n" + "=" * 60)
    print("Done.")
