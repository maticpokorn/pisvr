import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
from cvxpy import Problem, Minimize, Variable, quad_form, psd_wrap
 
jax.config.update("jax_enable_x64", True)
 
 
def _arr(X):
    return jnp.asarray(X, dtype=jnp.float64)
 
 
class PISVR:
 
    def __init__(self, m, lambda_, eps1, C1, eps2, C2, L, f, pde_params=None):
        self.m = m
        self.lambda_ = lambda_
        self.eps1 = eps1
        self.C1 = C1
        self.eps2 = eps2
        self.C2 = C2
        self.L = L
        self.f = f
        self.pde_params = {} if pde_params is None else pde_params
 
        self.x1 = sp.Matrix(sp.symbols(f"x1_1:{m+1}", real=True))
        self.x2 = sp.Matrix(sp.symbols(f"x2_1:{m+1}", real=True))
        self.lam = sp.Symbol("lam", positive=True)
        k = sp.exp(-self.lam * (self.x1 - self.x2).dot(self.x1 - self.x2))
        args = (list(self.x1), list(self.x2), self.lam)
 
        self._K = self._gram_fn(sp.lambdify(args, k, "jax"))
        self._Lx1K = self._gram_fn(sp.lambdify(args, L(k, self.x1), "jax"))
        self._Lx2K = self._gram_fn(sp.lambdify(args, L(k, self.x2), "jax"))
        self._Lx1Lx2K = self._gram_fn(sp.lambdify(args, L(L(k, self.x1), self.x2), "jax"))
        self.L1 = float(L(sp.Integer(1), self.x1))
 
    def _hp(self, hp=None):
        d = dict(lam=self.lambda_, eps1=self.eps1, C1=self.C1, eps2=self.eps2, C2=self.C2)
        if hp:
            d.update(hp)
        return d
    
    @staticmethod
    def _gram_fn(fun):
        row = jax.vmap(fun, in_axes=(None, 0, None))
        return jax.jit(jax.vmap(row, in_axes=(0, None, None)))
 
    def _lam(self, lam):
        return self.lambda_ if lam is None else lam
 
    def RBF(self, X1, X2, lam=None):
        return self._K(_arr(X1), _arr(X2), self._lam(lam))
 
    def Lx1RBF(self, X1, X2, lam=None):
        return self._Lx1K(_arr(X1), _arr(X2), self._lam(lam))
 
    def Lx2RBF(self, X1, X2, lam=None):
        return self._Lx2K(_arr(X1), _arr(X2), self._lam(lam))
 
    def Lx1Lx2RBF(self, X1, X2, lam=None):
        return self._Lx1Lx2K(_arr(X1), _arr(X2), self._lam(lam))
 
    @staticmethod
    def block(M):
        return jnp.block([[M, -M], [-M, M]])
 
    def P(self, X_data, X_phys, lam=None):
        K11 = self.block(self.RBF(X_data, X_data, lam))
        K12 = self.block(self.Lx2RBF(X_data, X_phys, lam))
        K21 = self.block(self.Lx1RBF(X_phys, X_data, lam))
        K22 = self.block(self.Lx1Lx2RBF(X_phys, X_phys, lam))
        P = jnp.block([[K11, K12], [K21, K22]])
        return 0.5 * (P + P.T)
 
    def f_values(self, X_phys):
        Xp = _arr(X_phys)
        try:
            return jnp.asarray(jax.vmap(self.f)(Xp), dtype=jnp.float64)
        except Exception:
            return jnp.asarray(np.array([self.f(z) for z in np.asarray(X_phys)]),
                               dtype=jnp.float64)
 
    def q(self, y_data, X_phys, eps1=None, eps2=None):
        eps1 = self.eps1 if eps1 is None else eps1
        eps2 = self.eps2 if eps2 is None else eps2
        y = _arr(y_data)
        fX = self.f_values(X_phys)
        n, N = y.shape[0], fX.shape[0]
        return jnp.hstack([eps1 * jnp.ones(n) - y, eps1 * jnp.ones(n) + y,
                           eps2 * jnp.ones(N) - fX, eps2 * jnp.ones(N) + fX])
 
    def eq(self, n, N):
        row = jnp.hstack([jnp.ones(n), -jnp.ones(n),
                          self.L1 * jnp.ones(N), -self.L1 * jnp.ones(N)])
        return row.reshape(1, -1), jnp.zeros(1)
 
    def bounds(self, n, N, C1=None, C2=None):
        C1 = self.C1 if C1 is None else C1
        C2 = self.C2 if C2 is None else C2
        lb = jnp.zeros(2 * n + 2 * N)
        ub = jnp.hstack([C1 * jnp.ones(2 * n), C2 * jnp.ones(2 * N)])
        return lb, ub
 
    def ineq(self, n, N, C1=None, C2=None):
        lb, ub = self.bounds(n, N, C1, C2)
        return jnp.eye(2 * n + 2 * N), lb, ub
 
    def qp_data(self, X_data, y_data, X_phys, hp=None):
        h = self._hp(hp)
        n, N = len(y_data), len(X_phys)
        A_eq, b = self.eq(n, N)
        lb, ub = self.bounds(n, N, h["C1"], h["C2"])
        return dict(P=self.P(X_data, X_phys, h["lam"]),
                    q=self.q(y_data, X_phys, h["eps1"], h["eps2"]),
                    A_eq=A_eq, b=b, lb=lb, ub=ub)
 
    def fit(self, X_data, y_data, X_phys, hp=None):
        n, N = len(X_data), len(X_phys)
        d = {k: np.asarray(v) for k, v in self.qp_data(X_data, y_data, X_phys, hp).items()}
 
        x = Variable(2 * n + 2 * N)
        con = d["A_eq"] @ x == d["b"]
        prob = Problem(Minimize(0.5 * quad_form(x, psd_wrap(d["P"])) + d["q"] @ x),
                       [x >= d["lb"], x <= d["ub"], con])
        prob.solve(solver="CLARABEL")
        if x.value is None:
            raise RuntimeError(f"QP solve failed: status = {prob.status}")
 
        x = x.value
        self.alpha = x[:n]
        self.alpha_ = x[n:2 * n]
        self.beta = x[2 * n:2 * n + N]
        self.beta_ = x[2 * n + N:]
        self.X_data = np.asarray(X_data)
        self.X_phys = np.asarray(X_phys)
        self.hp_fit = self._hp(hp)
        self.b = float(np.ravel(con.dual_value)[0])
        return self
 
    def predict(self, X_test):
        lam = self.hp_fit["lam"]
        u = jnp.asarray(self.alpha - self.alpha_)
        v = jnp.asarray(self.beta - self.beta_)
        out = (self.RBF(X_test, self.X_data, lam) @ u
               + self.Lx2RBF(X_test, self.X_phys, lam) @ v + self.b)
        return np.asarray(out)