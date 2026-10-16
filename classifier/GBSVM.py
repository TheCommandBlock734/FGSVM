"""
Author: Lasse Regin Nielsen
"""

from __future__ import division, print_function

import math
import os
import random as rnd
import time as systime

import numpy as np
from numpy import sqrt
from sklearn.metrics import accuracy_score, recall_score


class SVM:
    """
    Simple implementation of a Support Vector Machine using the
    Sequential Minimal Optimization (SMO) algorithm for training.
    """

    def __init__(
        self, max_iter=2000, kernel_type="quadratic", C=1.0, epsilon=0.00001, N_data=[]
    ):
        self.kernels = {
            "linear": self.kernel_linear,
            "quadratic": self.kernel_quadratic,
            "gaussian": self.kernel_gaussian,
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.N_data = N_data
        self.Ker = self.K()
        # self.Ker_jian = self.K_jian()

    def K(self):  ##核函数矩阵
        kernel = self.kernels[self.kernel_type]
        KK = kernel(self.N_data[:, :-1], self.N_data[:, :-1])
        return KK

    # 计算两个半径的内积，把循环提到了最外层，不需要每次迭代都计算
    def C_C(self, granular_balls):
        c_c = np.zeros((len(granular_balls), len(granular_balls)))
        for i in range(len(granular_balls)):
            for j in range(len(granular_balls)):
                x_i = granular_balls[i]
                x_j = granular_balls[j]
                x_i = np.array(x_i)
                x_j = np.array(x_j)
                K_i = self.Ker[np.array(x_i[:, -1], dtype=int), :]
                K = K_i[:, np.array(x_j[:, -1], dtype=int)]
                c_c[i, j] = K.sum() / (len(x_i) * len(x_j))
        return c_c

    ##计算半径R向量
    def R_jian(self, granular_balls, C_C):
        k = len(granular_balls)
        R = np.zeros(k)
        kernel = self.kernels[self.kernel_type]
        for i1 in range(k):  # 遍历每一个球
            ball = granular_balls[i1]
            ball = np.array(ball)
            K_1 = self.Ker[np.array(ball[:, -1], dtype=int), :]
            K_11 = K_1[:, np.array(ball[:, -1], dtype=int)]
            R_1 = sqrt(
                [K_11[i][i] for i in range(len(K_11))]
                + C_C[i1, i1]
                - 2 * K_11.sum(axis=1) / len(ball)
            ).sum()  ##向量+ self.CC(i1, i1, granular_balls)
            R[i1] = R_1 / len(ball)
        return R

    def fit(self, y, granular_balls):
        # Initialization初始化
        n = len(granular_balls)
        alpha = np.zeros(n)
        count = 0
        C_C = self.C_C(granular_balls)
        # R = self.R(granular_balls)
        R = self.R_jian(granular_balls, C_C)
        R_sum = R.sum()
        # print("R", R)
        # print("R_2", R_2)
        ##中心点之间的矩阵计算

        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n - 1, j)  # Get random int i~=j 获取随机
                y_i, y_j = y[i], y[j]
                k11 = C_C[i, i]
                k12 = C_C[i, j]
                k22 = C_C[j, j]
                ##############################
                R_ii = R[i] * R[i]
                R_ij = R[i] * R[j]
                R_jj = R[j] * R[j]
                # print("000000000",i ,j, R[i],R[j])
                # sum_C1 = 0  # sum C1i
                # sum_C2 = 0  # sum C2i
                # for ii in range(len(alpha)):
                #     if (ii != i) & (ii != j):
                #         sum_C1 = sum_C1 + alpha[ii] * y[ii] * self.CC(i, ii, granular_balls)
                #         sum_C2 = sum_C2 + alpha[ii] * y[ii] * self.CC(j, ii, granular_balls)

                sum_R1 = 0  # sum R1i
                sum_R2 = 0  # sum R2i
                s = y_i * y_j
                # for jj in range(len(y)):
                #     if (jj != i) & (jj != j):
                #         sum_R1 = sum_R1 + R[i] * R[jj]
                #         sum_R2 = sum_R2 + R[j] * R[jj]
                # # print("GEZHONGZH_pre ",sum_R1,sum_R2)
                #######################
                R_3 = R_sum - R[i] - R[j]
                sum_R1 = R[i] * R_3  # sum R1i
                sum_R2 = R[j] * R_3  # sum R2i
                # print("GEZHONGZH", sum_R1, sum_R2)
                ########################
                la_1 = 0
                la_2 = 0
                # lambda
                #####################
                la_2 = (alpha * R).sum()
                la_11 = (alpha * y) * C_C
                la_12 = la_11.T * (alpha * y)
                la_1 = la_12.sum()
                # print("la_1_xin",la_1, la_2)
                # la_1 = 0
                # la_2 = 0
                ########################
                # for k1 in range(len(alpha)):
                #     la_2 = la_2 + alpha[k1] * R[k1]
                #     for k2 in range(len(alpha)):
                #         la_1 = la_1 + alpha[k1] * y[k1] * alpha[k2] * y[k2] * self.CC(k1, k2, granular_balls)
                # print("la_1", la_1,la_2)
                lambda1 = (2 * math.sqrt(la_1) / (la_2 + 0.000001)) - 1

                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]

                L, H = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters计算模型参数，只能计算出b
                # self.w = self.calc_w(alpha, y, X, R)
                # self.b = self.calc_b(alpha, y, R, granular_balls)
                ################
                self.b = self.calc_b_jian(alpha, y, R, C_C)  ##
                # print("self.bself.b", self.b, b_jian)
                ###############
                #####Compute f_i, f_j,与SVM不同，因为w的表达式较复杂一些
                # f_i = self.E(alpha, y, i,  granular_balls, R, self.b)
                # f_j = self.E(alpha, y, j,  granular_balls, R, self.b)
                ###########################
                f_i = self.E_jian(alpha, y, i, C_C, R, self.b)
                f_j = self.E_jian(alpha, y, j, C_C, R, self.b)
                # print("alpha_prime_j", f_i,f_ii)
                # print("alpha_prime_j", f_j,f_jj)
                ###########################
                signa = alpha_prime_i * y_i + alpha_prime_j * y_j
                # print("nannannanananan",k11,k12,k22,lambda1,sum_R1, sum_R2,sum_C1,sum_C2)
                # Set new alpha values设置新的alpha值
                eta = (
                    k11
                    + k22
                    - 2 * k12
                    - lambda1 * R_ii
                    - lambda1 * R_jj
                    + 2 * lambda1 * y_i * y_j * R_ij
                )
                # a2_1 = y_j * signa * (k11 - k12) + y_j * (
                #             sum_C1 - sum_C2) - lambda1 * signa * y_j * R_ii + lambda1 * signa * y_i * R_ij - lambda1 * s * sum_R1 + lambda1 * sum_R2 - s + 1
                # print("alpha[j]",signa,f_i,f_j,sum_R1,sum_R2)
                # 关于f_i,与f_j的更新公式
                a2_1 = (
                    y_j * signa * (k11 - k12)
                    + y_j
                    * (
                        f_i
                        - f_j
                        + signa * (k12 - k11)
                        + alpha_prime_j * y_j * (k11 + k22 - 2 * k12)
                    )
                    - lambda1
                    * (signa * y_j * R_ii - signa * y_i * R_ij + s * sum_R1 - sum_R2)
                    - s
                    + 1
                )

                # alpha_prime_j * (k11 - 2 * k12 + k22 - lambda1 * R_ii - s * R_ij) + y_j * (
                #         f_i - f_j) - lambda1 * (
                #                alpha_prime_i * s * R_ii + s * sum_R1 - sum_R2)
                # print("alpha_prime_j",alpha_prime_j, "sum_R1",sum_R1,"sum_R2",sum_R1, "eta", eta)
                alpha[j] = a2_1 / eta
                # print("alpha[j]",alpha[j],a2_1,eta )
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                # 更新alpha_i
                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])
                # print("alpha", alpha)
            # # # 验证;
            # A_1 = 0
            # B_1 = 0
            # C_1 = 0
            # for i in range(len(alpha)):
            #     C_1 += alpha[i]
            #     for j in range(len(alpha)):
            #         A_1 += alpha[i] * y[i] * alpha[j] * y[j] * self.CC(i, j, granular_balls)
            #         B_1 += alpha[i] * alpha[j] * R[i] * R[j]
            # L_11 = (A_1 / 2) - (B_1 / 2) - C_1   # 目标函数原始的
            # print("原始目标函数值：", L_11)
            # print("000000000")
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print(
                    "Iteration number exceeded the max of %d iterations"
                    % (self.max_iter)
                )
                break
        # Compute final model parameters linear
        # self.b = self.calc_b(alpha, y, R, granular_balls)
        self.b = self.calc_b_jian(alpha, y, R, C_C)
        # print("b:", self.b)
        return alpha, R, C_C, count

        # Prediction error

    # 求解损失
    def E_jian(self, alpha, y, x_i, C_C, R, b):
        ##求解||∑α_i* y_i*c_i||
        A = 0
        la_11 = (alpha * y) * C_C
        la_12 = la_11.T * (alpha * y)
        A = la_12.sum()
        # for i in range(len(alpha)):
        #     for j in range(len(alpha)):
        #         A += alpha[i] * y[i] * alpha[j] * y[j] * self.CC(i, j, granular_balls)
        A = math.sqrt(A)
        ###求解 ∑α_i* r_i
        B = (alpha * R).sum()
        # for ii in range(len(alpha)):
        #     B = B + alpha[ii] * R[ii]
        ###求解 (∑α_i* y_i*c_i )c_i
        C = 0
        C = ((alpha * y) * C_C[:, x_i]).sum()
        # for i2 in range(len(alpha)):
        #     C = C + alpha[i2] * y[i2] * self.CC(i2, x_i, granular_balls)
        y_h = ((A - B) * C) / (A + 0.000001) + b
        return y_h * (A / (A - B + 0.00000001))

    def predict(self, alpha, y, R, granular_balls, C_C, train_x):
        return self.h(alpha, y, R, granular_balls, train_x, self.b, C_C)

    # 计算b
    def calc_b_jian(self, alpha, y, R, C_C):
        ##求解||∑α_i* y_i*c_i ||        ###求解 ∑α_i* r_i = B
        B = 0
        A = 0
        B = (alpha * R).sum()
        la_11 = (alpha * y) * C_C
        la_12 = la_11.T * (alpha * y)
        A = la_12.sum()
        A = math.sqrt(A)
        ###求解 (∑α_i* y_i*c_i )c_i
        b_tmp = 0
        b_tmp = y - ((A - B) * ((alpha * y) * np.sum(C_C, axis=0))) / (A + 0.000001)
        return (b_tmp.sum()) / len(alpha)

    # def calc_w(self, alpha, y, X, R):
    #     # print('alpha***********',alpha)
    #     A_2 = np.zeros(len(X[0]))
    #     B_2 = 0
    #     t_x2 = 0
    #     for k in range(len(alpha)):
    #         A_2 = A_2 + alpha[k] * y[k] * X[k]
    #         B_2 = B_2 + alpha[k] * R[k]
    #         t_x2 += alpha[k]
    #     # A_1 = np.sqrt(np.sum([i * i for i in A]))
    #     A_22 = np.sqrt(np.sum(A_2 ** 2))
    #     w = ((A_22 - B_2) * A_2) / A_22
    #     return w
    ##求解C*X
    def CX(self, i, x, granular_balls):
        kernel = self.kernels[self.kernel_type]
        C_X = 0
        for jj in range(len(granular_balls[i])):
            C_X = C_X + kernel(granular_balls[i][jj][:-2], x)
        return C_X

    # Prediction
    def h(self, alpha, y, R, granular_balls, x, b, C_C):
        # print("alpha", alpha, R)
        ##求解||∑α_i* y_i*c_i||
        A = 0
        # for i in range(len(alpha)):
        #     for j in range(len(alpha)):
        #         A += alpha[i] * y[i] * alpha[j] * y[j] * self.CC(i, j, granular_balls)
        la_11 = (alpha * y) * C_C
        la_12 = la_11.T * (alpha * y)
        A = la_12.sum()
        A = math.sqrt(A)
        ###求解 ∑α_i* r_i
        B = 0
        B = (alpha * R).sum()
        # for ii in range(len(alpha)):
        #     # print("B", B)
        #     B = B + alpha[ii] * R[ii]
        ###求解 (∑α_i* y_i*c_i )x_i
        y_h = np.zeros(len(x))
        for i1 in range(len(x)):
            C = 0
            for i2 in range(len(alpha)):
                C = C + alpha[i2] * y[i2] * self.CX(i2, x[i1], granular_balls)
            y_h[i1] = np.sign(((A - B) * C) / (A + 0.000001) + b).astype(int)
            # print("A,B,C",  A, B, C, ((A - B) * C) / (A + 0.000001) + b)
        return y_h

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if y_i != y_j:
            return (
                max(0, alpha_prime_j - alpha_prime_i),
                min(C, max(C - alpha_prime_i + alpha_prime_j, 0)),
            )
        else:
            return (
                max(0, alpha_prime_i + alpha_prime_j - C),
                min(C, max(alpha_prime_i + alpha_prime_j, 0)),
            )

        # if (y_i != y_j):
        #     return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        # else:
        #     return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a, b, z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt + 1
        return i

    # Define kernels定义核函数
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return np.dot(x1, x2.T) ** 2

    def kernel_gaussian(self, x, y, sigma=1):
        # 高斯核函数
        """Returns the gaussian similarity of arrays 'x' and 'y' with
        kernel width paramenter 'sigma' (set to 1 by default)"""

        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(-((np.linalg.norm(x - y, 2)) ** 2) / (2 * sigma**2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (
            np.ndim(x) == 1 and np.ndim(y) > 1
        ):
            result = np.exp(-(np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma**2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(
                -(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2)
                / (2 * sigma**2)
            )
        return result


def runClassifier(C, r, y, points, tX, ty, max_iter=2000, C_xi=1, tol=1e-5):
    y = y.copy()
    y[y != 1] = -1
    cur = 0
    granular_balls = []
    for gb in points:
        n = gb.shape[0]
        py = gb[:, 0].copy().reshape(-1, 1)
        py[py != 1] = -1
        data = np.hstack([gb[:, 1:], py, np.arange(cur, cur + n).reshape(-1, 1)])
        granular_balls.append(data.tolist())
        cur += n
    N_data = np.vstack(points)
    py = N_data[:, 0].copy().reshape(-1, 1)
    py[py != 1] = -1
    N_data = np.hstack([N_data[:, 1:], py])
    start = systime.time()
    model = SVM(
        max_iter=max_iter, kernel_type="linear", C=C_xi, epsilon=tol, N_data=N_data
    )
    alpha, R, C_C, count = model.fit(y, granular_balls)
    pred_ty = model.predict(alpha, y, R, granular_balls, C_C, tX)
    end = systime.time()
    pred_ty[pred_ty != 1] = 0

    A = np.sum((alpha * y)[:, None] * C, axis=0)
    B = np.sum(alpha * r)
    w = (1 - B / np.linalg.norm(A)) * A
    return (
        end - start,
        accuracy_score(ty, pred_ty, normalize=True),
        recall_score(ty, pred_ty),
        w,
        model.b,
        count,
    )
