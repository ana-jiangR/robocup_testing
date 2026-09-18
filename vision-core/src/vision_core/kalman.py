"""A small linear Kalman filter, shared by everything that tracks over time.

Knows nothing about balls, tags, gravity or the field. It holds a state vector
and its covariance and does the two textbook steps -- predict through a
transition matrix, update against a measurement -- plus the two details that
are the same for every tracker and worth getting right once:

- a Mahalanobis gate, so a measurement that is impossible given the current
  estimate is refused instead of dragging the estimate towards it;
- the Joseph-form covariance update, which stays symmetric positive-definite
  even when the gain is poor.

State dimensions declared as angles (radians) get their residual wrapped, so a
heading measured at +179 degrees against a prediction of -179 is a 2 degree
error, not 358.

Physics belongs to the caller: which state dimensions exist, what F/Q/H/R are,
and anything like gravity or a change of measurement model. The two helpers
below build the constant-velocity F and Q that both current callers use.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap radians into [-pi, pi). Works on scalars and arrays."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def cv_transition(dt: float, n: int) -> np.ndarray:
    """Constant-velocity transition for a state laid out [p_1..p_n, v_1..v_n]."""
    F = np.eye(2 * n)
    F[:n, n:] = np.eye(n) * dt
    return F


def cv_process_noise(dt: float, sigma_a) -> np.ndarray:
    """Piecewise white acceleration noise for the same layout.

    `sigma_a` is one acceleration sigma per position dimension (m/s^2, or
    rad/s^2 for an angle), so a robot can be allowed to turn far more sharply
    than it accelerates.
    """
    q = np.asarray(sigma_a, np.float64).reshape(-1) ** 2
    n = q.size
    D = np.diag(q)
    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = D * (dt**4) / 4.0
    Q[:n, n:] = D * (dt**3) / 2.0
    Q[n:, :n] = D * (dt**3) / 2.0
    Q[n:, n:] = D * (dt**2)
    return Q


class KalmanFilter:
    """State `x`, covariance `P`, and nothing else."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        gate: float = 9.0,
        angle_dims: tuple[int, ...] = (),
    ) -> None:
        self.x = np.asarray(x0, np.float64).reshape(-1).copy()
        self.P = np.asarray(P0, np.float64).copy()
        #: Mahalanobis threshold per measurement dimension. ~9 is the 99% point
        #: for one degree of freedom; the test below scales it by len(z).
        self.gate = float(gate)
        self.angle_dims = tuple(angle_dims)

    def _wrap_state(self) -> None:
        for d in self.angle_dims:
            self.x[d] = wrap_angle(self.x[d])

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self._wrap_state()

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> bool:
        """One update. Returns False, and changes nothing, if `z` fails the gate."""
        z = np.asarray(z, np.float64).reshape(-1)
        y = z - H @ self.x
        for i in range(len(y)):
            if any(H[i, d] != 0.0 for d in self.angle_dims):
                y[i] = wrap_angle(y[i])
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        if float(y @ S_inv @ y) > self.gate * len(z):
            return False
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        A = np.eye(len(self.x)) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self._wrap_state()
        return True
