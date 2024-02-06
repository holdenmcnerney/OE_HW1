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
    G = np.array([[nt**-2 * (1 - cos(ntdt)), 2 * nt**-2 * (ntdt - sin(ntdt)), 0], 
                  [-2 * nt**-2 * (ntdt - sin(ntdt)), 4 * nt**-2 * (1 - cos(ntdt)) - 3 / 2 * dt**2, 0], 
                  [0, 0, nt**-2 * (1 - cos(ntdt))], 
                  [nt**-1 * sin(ntdt), 2 * nt**-1 * (1 - cos(ntdt)), 0], 
                  [-2 * nt**-1 * (1- cos(ntdt)), 4 * nt**-1 * sin(ntdt) - 3 * dt, 0], 
                  [0, 0, nt**-1 * sin(ntdt)]])
    # G = np.array([[nt**-1 * sin(ntdt), 2 * nt**-1 * (1 - cos(ntdt)), 0], 
    #               [-2 * nt**-1 * (1 - cos(ntdt)), nt**-1 * (4 * sin(ntdt) - 3 * ntdt), 0], 
    #               [0, 0, nt**-1 * sin(ntdt)], 
    #               [cos(ntdt), 2 * sin(ntdt), 0], 
    #               [-2 * sin(ntdt), 4 * cos(ntdt) - 3, 0], 
    #               [0, 0, cos(ntdt)]])

    return G  

def FH_CLQR(tf: int, nt: float, Q: np.array, R: np.array, x_0: np.array):

    A = build_A(nt)
    B = build_B()
    time_range = np.arange(0, tf)
    time_range_flip = np.flip(np.arange(0, tf))
    P_0 = np.zeros((6, 6))
    K_list = []
    x_hist = np.atleast_2d(np.array([0, 0, 0, 0, 0, 0]))
    u_hist = np.atleast_2d(np.array([0, 0, 0]))

    def riccati(t: int, P: np.array):

        P = P.reshape(6, 6)
        return (- P @ A - A.T @ P + P @ B @ la.inv(R) @ B.T @ P - Q).flatten()

    ric_sol = sp.integrate.solve_ivp(riccati, [tf, 0], P_0.flatten(), rtol = 1e-3, t_eval = time_range_flip)
    P_sol = ric_sol.y.T
    for P in P_sol:
        P = P.reshape(6, 6)
        K_list.append(la.inv(R) @ B.T @ P)

    def cl_st_dyn(t: int, x: np.array):
            t = int(t)
            return ((A - B @ K_list[t - 1]) @ x).flatten()

    state_sol = sp.integrate.solve_ivp(cl_st_dyn, [0, tf], x_0.flatten(), rtol = 1e-3, t_eval = time_range)
    x_sol = state_sol.y.T

    for K, x in zip(K_list, x_sol):
        x_hist = np.vstack((x_hist, x))
        u_hist = np.vstack((u_hist, - K @ x.T))

    x_hist = np.delete(x_hist, 0, axis=0)
    u_hist = np.delete(u_hist, 0, axis=0)

    return x_hist, u_hist

def IH_CLQR(nt: float, Q: np.array, R: np.array, x_0: np.array):

    A = build_A(nt)
    B = build_B()
    x_hist = np.atleast_2d(np.array([0, 0, 0, 0, 0, 0]))
    u_hist = np.atleast_2d(np.array([0, 0, 0]))

    P = la.solve_continuous_are(A, B, Q, R)
    K = la.inv(R) @ B.T @ P

    def cl_st_dyn(t: int, x: np.array):
            t = int(t)
            return ((A - B @ K) @ x).flatten()

    state_sol = sp.integrate.solve_ivp(cl_st_dyn, [0, 1e5], x_0.flatten(), rtol = 1e-2)
    x_sol = state_sol.y.T
    t = state_sol.t

    for x in x_sol:
        x_hist = np.vstack((x_hist, x))
        u_hist = np.vstack((u_hist, - K @ x.T))

    x_hist = np.delete(x_hist, 0, axis=0)
    u_hist = np.delete(u_hist, 0, axis=0)

    return x_hist, u_hist, t

def FH_DLQR(N: int, dt: int, nt: float, Q: np.array, R: np.array, x_0: np.array):
    x_old = x_0
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

def IH_DLQR(dt: int, nt: float, Q: np.array, R: np.array, x_0: np.array):

    x_hist = x_0.T
    u_hist = np.atleast_2d(np.array([0, 0, 0]))
    F = build_F(nt, dt)
    G = build_G(nt, dt)
    P = la.solve_discrete_are(F, G, Q, R)
    K = la.inv(G.T @ P @ G + R) @ G.T @ P @ F
    x_old = x_0

    while la.norm(x_old) > 1e-5:

        x = (F - G @ la.inv(G.T @ P @ G + R) @ (G.T @ P @ F)) @ x_old
        u = -K @ x_old
        x_hist = np.vstack((x_hist, x.T))
        u_hist = np.vstack((u_hist, u.T))
        x_old = x

    return x_hist, u_hist

def main():

    # Constants
    mu = 3.986004418e14     # m^3/s^2
    rt = 6783000            # m
    # Initial conditions
    x_0 = np.array([[1000], [1000], [1000], [0], [0], [0]])
    nt = np.sqrt(mu / rt**3)
    A = build_A(nt)
    B = build_B()
    val, vec = la.eig(A)
    print(f'Eigenvalues: {val}')
    control_mat = np.block([B, A @ B, A**2 @ B, A**3 @ B, A**4 @ B, A**5 @ B])
    print(f'Controllability matrix rank: {np.linalg.matrix_rank(control_mat)}')
    print(f'Controllability matrix: {control_mat}')
    input_dict = {'Case_1':[np.eye(6), np.eye(3) * 1e3],
                  'Case_2':[np.eye(6), 100 * np.eye(3) * 1e3],
                  'Case_3':[np.eye(6), 10000 * np.eye(3) * 1e3]}
    plotting = True
    if plotting:
        # Plotting finite-horizon LQR for the continuous-time LTI system
        for key in input_dict.keys():
            x_fc, u_fc = FH_CLQR(800, nt, 
                                input_dict[key][0], 
                                input_dict[key][1], 
                                x_0)
            time = np.arange(0, len(x_fc[:, 2]))

            fig, ax = plt.subplots(1, 3)
            fig.suptitle(f'''Finite Horizon LQR - Continuous - {key}''')
            ax[0].set_title('X vs Y')
            ax[0].set_xlabel(r'X ($m$)')
            ax[0].set_ylabel(r'Y ($m$)')
            ax[0].plot(x_fc[:, 0], x_fc[:, 1])

            ax[1].set_title('Height vs Time')
            ax[1].set_ylabel(r'Height ($m$)')
            ax[1].set_xlabel(r'Time ($s$)')
            ax[1].plot(time, x_fc[:, 2])
            
            ax[2].set_title('Acceleration Input vs Time')
            ax[2].set_ylabel(r'Acceleartion ($m/s^2$)')
            ax[2].set_xlabel(r'Time ($s$)')
            ax[2].plot(time, u_fc[:, 0], '--', label='u_x')
            ax[2].plot(time, u_fc[:, 1], '--', label='u_y')
            ax[2].plot(time, u_fc[:, 2], '--', label='u_z')
            ax[2].legend()

            fig.set_size_inches(16, 4)
            fig.savefig(f'continuous_finite_horizon_{key}.png', dpi=400)
            plt.close(fig)

        # Plotting infinite-horizon LQR for the continuous-time LTI system
        for key in input_dict.keys():
            x_ic, u_ic, t_ic = IH_CLQR(nt, 
                                input_dict[key][0], 
                                input_dict[key][1], 
                                x_0)
            time = np.arange(0, len(x_ic[:, 2]))

            fig, ax = plt.subplots(1, 3)
            fig.suptitle(f'''Infinite Horizon LQR - Continuous - {key}''')
            ax[0].set_title('X vs Y')
            ax[0].set_xlabel(r'X ($m$)')
            ax[0].set_ylabel(r'Y ($m$)')
            ax[0].plot(x_ic[:, 0], x_ic[:, 1])

            ax[1].set_title('Height vs Time')
            ax[1].set_ylabel(r'Height ($m$)')
            ax[1].set_xlabel(r'Time ($s$)')
            ax[1].set_xlim(-25, 825)
            ax[1].plot(t_ic, x_ic[:, 2])
            
            ax[2].set_title('Acceleration Input vs Time')
            ax[2].set_ylabel(r'Acceleartion ($m/s^2$)')
            ax[2].set_xlabel(r'Time ($s$)')
            ax[2].set_xlim(-25, 825)
            ax[2].plot(t_ic, u_ic[:, 0], '--', label='u_x')
            ax[2].plot(t_ic, u_ic[:, 1], '--', label='u_y')
            ax[2].plot(t_ic, u_ic[:, 2], '--', label='u_z')
            ax[2].legend()

            fig.set_size_inches(16, 4)
            fig.savefig(f'continuous_infinite_horizon_{key}.png', dpi=400)
            plt.close(fig)

        # Plotting finite-horizon LQR for the discrete-time LTI system
        for key in input_dict.keys():
            x_fd, u_fd = FH_DLQR(800, 1, nt, 
                                input_dict[key][0], 
                                input_dict[key][1], 
                                x_0)
            time_x = np.arange(0, len(x_fd[:, 2]))
            time_u = np.arange(0, len(u_fd[:, 0]))

            fig, ax = plt.subplots(1, 3)
            fig.suptitle(f'''Finite Horizon LQR - Discrete - {key}''')
            ax[0].set_title('X vs Y')
            ax[0].set_xlabel(r'X ($m$)')
            ax[0].set_ylabel(r'Y ($m$)')
            ax[0].plot(x_fd[:, 0], x_fd[:, 1])

            ax[1].set_title('Height vs Time')
            ax[1].set_ylabel(r'Height ($m$)')
            ax[1].set_xlabel(r'Time ($s$)')
            ax[1].plot(time_x, x_fd[:, 2])
            
            ax[2].set_title('Acceleration Input vs Time')
            ax[2].set_ylabel(r'Acceleartion ($m/s^2$)')
            ax[2].set_xlabel(r'Time ($s$)')
            ax[2].plot(time_u, u_fd[:, 0], '--', label='u_x')
            ax[2].plot(time_u, u_fd[:, 1], '--', label='u_y')
            ax[2].plot(time_u, u_fd[:, 2], '--', label='u_z')
            ax[2].legend()

            fig.set_size_inches(16, 4)
            fig.savefig(f'discrete_finite_horizon_{key}.png', dpi=400)
            plt.close(fig)

        # Plotting infinite-horizon LQR for the discrete-time LTI system
        for key in input_dict.keys():
            x_id, u_id = IH_DLQR(1, nt, 
                                input_dict[key][0], 
                                input_dict[key][1], 
                                x_0)
            time_x = np.arange(0, len(x_id[:, 2]))
            time_u = np.arange(0, len(u_id[:, 0]))

            fig, ax = plt.subplots(1, 3)
            fig.suptitle(f'''Infinite Horizon LQR - Discrete - {key}''')
            ax[0].set_title('X vs Y')
            ax[0].set_xlabel(r'X ($m$)')
            ax[0].set_ylabel(r'Y ($m$)')
            ax[0].plot(x_id[:, 0], x_id[:, 1])

            ax[1].set_title('Height vs Time')
            ax[1].set_ylabel(r'Height ($m$)')
            ax[1].set_xlabel(r'Time ($s$)')
            ax[1].plot(time_x, x_id[:, 2])
            
            ax[2].set_title('Acceleration Input vs Time')
            ax[2].set_ylabel(r'Acceleartion ($m/s^2$)')
            ax[2].set_xlabel(r'Time ($s$)')
            ax[2].plot(time_u, u_id[:, 0], '--', label='u_x')
            ax[2].plot(time_u, u_id[:, 1], '--', label='u_y')
            ax[2].plot(time_u, u_id[:, 2], '--', label='u_z')
            ax[2].legend()

            fig.set_size_inches(16, 4)
            fig.savefig(f'discrete_infinite_horizon_{key}.png', dpi=400)
            plt.close(fig)

    return 1

if __name__ == '__main__':
    main()