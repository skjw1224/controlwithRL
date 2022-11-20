import numpy as np
import casadi as ca

class CstrEnv(object):
    def __init__(self):
        self.envname = 'CSTR'
        self.E1 = -9758.3
        self.E2 = -9758.3
        self.E3 = -8560.

        self.rho = 0.9342  # (KG / L)
        self.Cp = 3.01  # (KJ / KG K)
        self.kw = 4032.  # (KJ / h M ^ 2 K)
        self.AR = 0.215  # (M ^ 2)
        self.VR = 10.  # L
        self.mk = 5.  # (KG)
        self.CpK = 2.0  # (KJ / KG K)

        self.CA0 = 5.1  # mol / L
        self.T0 = 378.05  # K

        self.k10 = 1.287e+12
        self.k20 = 1.287e+12
        self.k30 = 9.043e+9

        self.delHRab = 4.2  # (KJ / MOL)
        self.delHRbc = -11.0  # (KJ / MOL)
        self.delHRad = -41.85  # (KJ / MOL)

        self.s_dim = 7
        self.a_dim = 2
        self.o_dim = 1

        # MX variable for dae function object (no SX)
        self.state_var = ca.MX.sym('x', self.s_dim)
        self.action_var = ca.MX.sym('u', self.a_dim)

        self.t0 = 0.
        self.dt = 20 / 3600.  # hour
        self.tT = 3600 / 3600.  # terminal time

        self.x0 = np.array([[0., 2.1404, 1.4, 387.34, 386.06, 14.19, -1113.5]]).T
        self.u0 = np.array([[0., 0.]]).T
        self.nT = int(self.tT / self.dt) + 1  # episode length

        self.xmin = np.array([[self.t0, 0.001, 0.001, 353.15, 363.15, 3., -9000.]]).T
        self.xmax = np.array([[self.tT, 3.5, 1.8, 413.15, 408.15, 35., 0.]]).T
        self.umin = np.array([[-1., -1000.]]).T / self.dt
        self.umax = np.array([[1., 1000.]]).T / self.dt
        self.ymin = self.xmin[2]
        self.ymax = self.xmax[2]

        self.sym_expressions()

        self.reset()

    def reset(self):
        x0 = self.scale(self.x0, self.xmin, self.xmax)
        t0 = self.t0
        u0 = self.scale(self.u0, self.umin, self.umax)
        y0 = self.y_fnc(x0, u0).full()
        return t0, x0, y0, u0

    def ref_traj(self):
        return np.array([0.95])

    def step(self, time, state, action):
        # Scaled state, action, output
        t = round(time, 7)
        x = np.clip(state, -2, 2)
        u = action

        # Identify episode terminal
        is_term = False
        if self.tT - self.dt < t <= self.tT:
            is_term = True

        # Integrate ODE
        tplus = t + self.dt
        res = self.I_fnc(x0=x, p=u)
        xplus = res['xf'].full()
        cost = res['qf'].full()

        # Compute output
        xplus = np.clip(xplus, -2, 2)
        yplus = self.y_fnc(xplus, u).full()

        return tplus, xplus, yplus, cost, is_term

    def system_functions(self, x, u):
        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = ca.fmax(x, self.xmin)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        delHRab, delHRbc, delHRad = self.delHRab, self.delHRbc, self.delHRad
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        t, CA, CB, T, TK, VdotVR, QKdot = ca.vertsplit(x)
        dVdotVR, dQKdot = ca.vertsplit(u)

        k1 = k10 * ca.exp(E1 / T)
        k2 = k20 * ca.exp(E2 / T)
        k3 = k30 * ca.exp(E3 / T)

        dx = [1.,
              VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
              -VdotVR * CB + k1 * CA - k2 * CB,
              VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) /
              (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T),
              (QKdot + (kw * AR) * (T - TK)) / (mk * CpK),
              dVdotVR,
              dQKdot]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(CB)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, x, u):
        Q = np.diag([5.])
        R = np.diag([0.1, 0.1])

        y = self.y_fnc(x, u)
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)

        cost = 0.5 * (y - ref).T @ Q @ (y - ref) + 0.5 * u.T @ R @ u

        return cost

    def sym_expressions(self):
        """Syms: :Symbolic expressions, Fncs: Symbolic input/output structures"""

        # lists of sym_vars
        self.path_sym_args = [self.state_var, self.action_var]

        self.path_sym_args_str = ['x', 'u']

        "Symbolic functions of f, y"
        self.f_sym, self.y_sym = self.system_functions(self.state_var, self.action_var)
        self.f_fnc = ca.Function('f_fnc', [self.state_var, self.action_var], [self.f_sym], ['x', 'u'], ['f'])
        self.y_fnc = ca.Function('y_fnc', [self.state_var, self.action_var], [self.y_sym], ['x', 'u'], ['y'])

        "Symbolic function of c"
        self.c_sym = self.cost_functions(self.state_var, self.action_var)
        self.c_fnc = ca.Function('c_fnc', [self.state_var, self.action_var], [self.c_sym], ['x', 'u'], ['c'])

        "Symbolic function of dae solver"
        dae = {'x': self.state_var, 'p': self.action_var, 'ode': self.f_sym, 'quad': self.c_sym}
        opts = {'t0': 0., 'tf': self.dt}
        self.I_fnc = ca.integrator('I', 'cvodes', dae, opts)

    def scale(self, var, min, max, shift=True):
        shifting_factor = max + min if shift else 0.
        scaled_var = (2. * var - shifting_factor) / (max - min)

        return scaled_var

    def descale(self, scaled_var, min, max):
        var = (max - min) / 2 * scaled_var + (max + min) / 2

        return var