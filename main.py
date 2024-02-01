# Optimal Estimation - HW1 - LQR Design

import numpy as np
from numpy import cos, sin
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def build_A(nt: float):

    A = np.array([[0, 0, 0, 1, 0, 0], 
                  [0, 0, 0, 0, 1, 0], 
                  [0, 0, 0, 0, 0, 1], 
                  [3 * nt**2, 0, 0, 0, 2 * nt, 0], 
                  [0, 0, 0, -2 * nt, 0, 0], 
                  [0, 0, -nt**2, 0, 0, 0]])

    return A

def build_B():

    B = np.array([[0, 0, 0], 
                  [0, 0, 0], 
                  [0, 0, 0], 
                  [1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, 1]])

    return B

def build_F(nt: float, dt: float):

    ntdt = nt * dt
    F = np.array([[4 - 3 * cos(ntdt), 0, 0, nt**-1 * sin(ntdt), 2 * nt**-1 * (1 - cos(ntdt)), 0], 
                  [6 * (sin(ntdt) - ntdt), 1, 0, -2 * nt**-1 * (1 - cos(ntdt)), nt**-1 * (4 * sin(ntdt) - 3 * ntdt), 0], 
                  [0, 0, cos(ntdt), 0, 0, nt**-1 * sin(ntdt)], 
                  [3 * nt * sin (ntdt), 0, 0, cos(ntdt), 2 * sin(ntdt), 0], 
                  [-6 * nt * (1 - cos(ntdt)), 0, 0, -2 * sin(ntdt), 4 * cos(ntdt) - 3, 0], 
                  [0, 0, -nt * sin(ntdt), 0, 0, cos(ntdt)]])

    return F

def build_G(nt: float, dt: float):

    ntdt = nt * dt
    G = np.array([[nt**-1 * sin(ntdt), 2 * nt**-1 * (1 - cos(ntdt)), 0], 
                  [-2 * nt**-1 * (1 - cos(ntdt)), nt**-1 * (4 * sin(ntdt) - 3 * ntdt), 0], 
                  [0, 0, nt**-1 * sin(ntdt)], 
                  [cos(ntdt), 2 * sin(ntdt), 0], 
                  [-2 * sin(ntdt), 4 * cos(ntdt) - 3, 0], 
                  [0, 0, cos(ntdt)]])

    return G  

def FH_DLQR(N: int, dt: int, nt: float, Q: np.array, R: np.array, x_0: np.array):

    P_list = []
    x_hist = x_0.T
    u_hist = np.atleast_2d(np.array([0, 0, 0]))

    P_old = np.zeros((6, 6))
    F = build_F(nt, dt)
    G = build_G(nt, dt)

    while N > 0:

        P = F.T @ P_old @ F + Q \
            - (F.T @ P_old @ G) @ la.inv(G.T @ P_old @ G + R) @ (G.T @ P_old @ F)
        P_old = P
        P_list.append(P)
        N-=1

    x_old = x_0

    for i, P in enumerate(P_list[::-1]):

        if i == 1:
            pass
        x = (F - G @ la.inv(G.T @ P @ G + R) @ (G.T @ P @ F)) @ x_old
        K = la.inv(G.T @ P @ G + R) @ G.T @ P @ F
        u = -K @ x_old

        x_hist = np.vstack((x_hist, x.T))
        u_hist = np.vstack((u_hist, u.T))

        x_old = x

    u_hist = np.delete(u_hist, 0, axis=0)

    return x_hist, u_hist

def main():

    # Constants
    mu = 3.986004418e14     # m^3/s^2
    rt = 6783000            # m
    
    x_0 = np.array([[1000], [1000], [1000], [0], [0], [0]])

    nt = np.sqrt(mu / rt**3)
    A = build_A(nt)
    B = build_B()

    val, vec = la.eig(A)
    # print(val)

    control_mat = np.block([B, A @ B, A**2 @ B, A**3 @ B, A**4 @ B, A**5 @ B])

    # print(np.linalg.matrix_rank(control_mat))

    Q = np.eye(6)
    R = np.eye(3)

    x, u = FH_DLQR(1500, 1, nt, Q, R, x_0)

    time = np.arange(0, 1501)

    # plt.plot(x[:,0], x[:, 1])
    plt.plot(time, x[:, 2])

    plt.show()

    return 1

if __name__ == '__main__':
    main()