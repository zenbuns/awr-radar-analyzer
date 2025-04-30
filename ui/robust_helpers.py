from functools import partial
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
import numba

def robust_cr(points_xyz: np.ndarray, max_dt=0.01):
    # ① Mahalanobis gate (3‑D)
    mean = np.mean(points_xyz, axis=0)
    Σ = np.cov(points_xyz, rowvar=False)
    Σ_inv = np.linalg.inv(Σ + 1e-6*np.eye(3))
    d2 = np.einsum('ij,jk,ik->i', points_xyz-mean, Σ_inv, points_xyz-mean)
    keep = d2 < 5.99            # 95 % χ²(3) ellipse
    points_xyz = points_xyz[keep]

    # ③ DBSCAN to ensure single compact cluster
    labels = DBSCAN(eps=0.05, min_samples=4).fit_predict(points_xyz)
    core = points_xyz[labels == stats.mode(labels)[0][0]]
    return core.mean(0), core

@numba.njit(fastmath=True)
def project_fast(X, R, T, K):
    X_cam = R @ X.T + T     # (3,N)
    z = X_cam[2] + 1e-9
    uv = (K[:2,:2] @ (X_cam[:2] / z)).T + K[:2,2]
    return uv, z > 0
