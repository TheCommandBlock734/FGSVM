import time as systime

import cvxpy as cp
import numpy as np
from sklearn.metrics import accuracy_score, recall_score


def runClassifier(C, r, y, tX, ty, C_xi=1, tol=1e-5, solver=cp.CLARABEL):
    y = y.copy()
    y[y != 1] = -1

    start = systime.time()
    cp_w = cp.Variable(C.shape[0])
    cons = [
        cp.norm((y.reshape(-1, 1) * C).T @ cp_w, 2) <= r @ cp_w + 1,
        y @ cp_w == 0,
        cp_w >= 0,
        cp_w <= C_xi,
    ]
    obj = cp.Maximize(cp.sum(cp_w))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)
    alpha = cp_w.value
    v = (y.reshape(-1, 1) * C).T @ alpha
    support = np.where((alpha > tol) * (alpha < C_xi - tol))[0]
    Z = C[support] @ v
    A = np.stack([Z * y[support] - np.linalg.norm(v) * r[support], y[support]]).T
    k, b = np.linalg.lstsq(A, np.ones(len(support)))[0]
    w = k * v
    pred_ty = np.where(tX @ w + b > 0, 1, 0)
    end = systime.time()

    return (
        end - start,
        accuracy_score(ty, pred_ty, normalize=True),
        recall_score(ty, pred_ty),
        w,
        b,
        support.shape[0],
        cons[0].expr.value,
    )
