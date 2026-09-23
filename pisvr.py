import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from cvxpy import Problem, Minimize, Variable, quad_form, psd_wrap, Parameter, sum_squares
from cvxpylayers.jax import CvxpyLayer
from sklearn.model_selection import KFold
 
jax.config.update("jax_enable_x64", True)
 
 
def _arr(X):
    return jnp.asarray(X, dtype=jnp.float64)
 
 
class PISVR:
 
    def __init__(self, m, lam, eps1, C1, eps2, C2, L, f, pde_params=None):
        self.m = m
        self.lam = float(lam)
        self.eps1 = float(eps1)
        self.C1 = float(C1)
        self.eps2 = float(eps2)
        self.C2 = float(C2)
        self.L = L
        self.f = f
        self.pde_params = {} if pde_params is None else pde_params
 
        self.x1 = sp.Matrix(sp.symbols(f"x1_1:{m+1}", real=True))
        self.x2 = sp.Matrix(sp.symbols(f"x2_1:{m+1}", real=True))
        self.lam_sym = sp.Symbol("lam", positive=True)
        k = sp.exp(-self.lam_sym * (self.x1 - self.x2).dot(self.x1 - self.x2))
        args = (list(self.x1), list(self.x2), self.lam_sym)
 
        self._K = self._gram_fn(sp.lambdify(args, k, "jax"))
        self._Lx1K = self._gram_fn(sp.lambdify(args, L(k, self.x1), "jax"))
        self._Lx2K = self._gram_fn(sp.lambdify(args, L(k, self.x2), "jax"))
        self._Lx1Lx2K = self._gram_fn(sp.lambdify(args, L(L(k, self.x1), self.x2), "jax"))
        self.L1 = float(L(sp.Integer(1), self.x1))
    
    @staticmethod
    def _gram_fn(fun):
        row = jax.vmap(fun, in_axes=(None, 0, None))
        return jax.jit(jax.vmap(row, in_axes=(0, None, None)))
 
    def RBF(self, X1, X2, lam):
        return self._K(_arr(X1), _arr(X2), lam)
 
    def Lx1RBF(self, X1, X2, lam):
        return self._Lx1K(_arr(X1), _arr(X2), lam)
 
    def Lx2RBF(self, X1, X2, lam):
        return self._Lx2K(_arr(X1), _arr(X2), lam)
 
    def Lx1Lx2RBF(self, X1, X2, lam):
        return self._Lx1Lx2K(_arr(X1), _arr(X2), lam)
 
    @staticmethod
    def block(M):
        return jnp.block([[M, -M], [-M, M]])
 
    def P(self, X_data, X_phys, lam):
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
 
    def q(self, y_data, X_phys, eps1, eps2):
        y = _arr(y_data)
        fX = self.f_values(X_phys)
        n, N = y.shape[0], fX.shape[0]
        return jnp.hstack([eps1 * jnp.ones(n) - y, eps1 * jnp.ones(n) + y,
                           eps2 * jnp.ones(N) - fX, eps2 * jnp.ones(N) + fX])
 
    def eq(self, n_data, n_phys):
        row = jnp.hstack([jnp.ones(n_data), -jnp.ones(n_data),
                          self.L1 * jnp.ones(n_phys), -self.L1 * jnp.ones(n_phys)])
        return row.reshape(1, -1), jnp.zeros(1)
 
    def bounds(self, n_data, n_phys, C1, C2):
        lb = jnp.zeros(2 * n_data + 2 * n_phys)
        ub = jnp.hstack([C1 * jnp.ones(2 * n_data), C2 * jnp.ones(2 * n_phys)])
        return lb, ub
 
    def ineq(self, n, N, C1=None, C2=None):
        lb, ub = self.bounds(n, N, C1, C2)
        return jnp.eye(2 * n + 2 * N), lb, ub
 
    def qp_data(self, X_data, y_data, X_phys, lam, C1, C2, eps1, eps2):
        n_data, n_phys = len(y_data), len(X_phys)
        A_eq, b = self.eq(n_data, n_phys)
        lb, ub = self.bounds(n_data, n_phys, C1, C2)
        return dict(P=self.P(X_data, X_phys, lam),
                    q=self.q(y_data, X_phys, eps1, eps2),
                    A_eq=A_eq, b=b, lb=lb, ub=ub)

    def forward_pass(self, params, n_splits=10):
        lam, C1, C2, eps1, eps2 = params
        n_data, n_phys = len(self.X_data), len(self.X_phys)
        fold_mses = []
        for fold, (train_idx, val_idx) in enumerate(self.split):
            X_train, X_val = self.X_data[train_idx], self.X_data[val_idx]
            y_train, y_val = self.y_data[train_idx], self.y_data[val_idx]
            n_data_fold = len(y_train)
            P = self.P(X_train, self.X_phys, lam)
            Lt = jnp.linalg.cholesky(P + 1e-6 * jnp.eye(P.shape[0])).T
            q = self.q(y_train, self.X_phys, eps1, eps2)
            A_eq, b = self.eq(n_data_fold, n_phys)
            lb, ub = self.bounds(n_data_fold, n_phys, C1, C2)
            layer = self.define_cvxpylayer(n_data_fold, n_phys)
            x, mu = self.layers[fold](Lt, q, A_eq, b, lb, ub)
            alpha = x[:n_data_fold] - x[n_data_fold:2 * n_data_fold]
            beta = x[2 * n_data_fold:2 * n_data_fold + n_phys] - x[2 * n_data_fold + n_phys:]

            y_pred = (self.RBF(X_val, X_train, lam) @ alpha
                      + self.Lx2RBF(X_val, self.X_phys, lam) @ beta + mu[0])
            mean_squared_error = jnp.mean((y_val - y_pred) ** 2)
            #print(f"Fold {fold + 1}/{n_splits}: Mean Squared Error = {mean_squared_error}")
            fold_mses.append(mean_squared_error)
        print(f"Cross-Validation: Mean MSE = {jnp.mean(jnp.array(fold_mses))}")
        print(params)
        return jnp.mean(jnp.array(fold_mses))
        

    def fit_cvxpylayer(self, X_data, y_data, X_phys, max_iter=100, lr=1e-4):
        self.X_data = jnp.asarray(X_data)
        self.y_data = jnp.asarray(y_data)
        self.X_phys = jnp.asarray(X_phys)
        for i in range(max_iter):
            grad = jax.grad(self.forward_pass, argnums=(0, 1, 2, 3, 4))(self.lam, self.C1, self.C2, self.eps1, self.eps2)
            #print(f"Iteration {i}: grad = {grad}")
            # Update parameters using gradient descent
            self.lam = max(self.lam - lr * grad[0], 1e-6)  # Ensure lam stays positive
            self.C1 = max(self.C1 - lr * grad[1], 0.0)
            self.C2 = max(self.C2 - lr * grad[2], 0.0)
            self.eps1 = max(self.eps1 - lr * grad[3], 0.0)
            self.eps2 = max(self.eps2 - lr * grad[4], 0.0)
            print(f"Iteration {i}: lam={self.lam}, C1={self.C1}, C2={self.C2}, eps1={self.eps1}, eps2={self.eps2}")
        return self.fit(self.X_data, self.y_data, self.X_phys)

    def fit_cvxpylayer_bfgs(self, X_data, y_data, X_phys, n_splits=10):
            self.X_data = jnp.asarray(X_data)
            self.y_data = jnp.asarray(y_data)
            self.X_phys = jnp.asarray(X_phys)

            n_data, n_phys = len(self.X_data), len(self.X_phys)
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            self.split = list(kfold.split(self.X_data))
            self.layers = [self.define_cvxpylayer(len(train_idx), n_phys) for train_idx, _ in self.split]

            bounds = [(1e-6, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
            result = minimize(self.forward_pass, x0=jnp.array([self.lam, self.C1, self.C2, self.eps1, self.eps2]), method='L-BFGS-B', bounds=bounds, jac=jax.grad(self.forward_pass, argnums=(0)))
            x = result.x
            self.lam, self.C1, self.C2, self.eps1, self.eps2 = x[0], x[1], x[2], x[3], x[4]
            return self.fit(self.X_data, self.y_data, self.X_phys)

    def cost_fn(self, x, Lt, q):
        return 0.5 * x @ Lt @ Lt.T @ x + q @ x
 
    def fit(self, X_data, y_data, X_phys):
        n_data, n_phys = len(X_data), len(X_phys)

        d = {k: np.asarray(v) for k, v in self.qp_data(X_data, y_data, X_phys, self.lam, self.C1, self.C2, self.eps1, self.eps2).items()}
 
        x = Variable(2 * n_data + 2 * n_phys)
        con = d["A_eq"] @ x == d["b"]
        prob = Problem(Minimize(0.5 * quad_form(x, psd_wrap(d["P"])) + d["q"] @ x),
                       [x >= d["lb"], x <= d["ub"], con])
        prob.solve(solver="CLARABEL")
 
        x = x.value
        self.alpha = x[:n_data]
        self.alpha_ = x[n_data:2 * n_data]
        self.beta = x[2 * n_data:2 * n_data + n_phys]
        self.beta_ = x[2 * n_data + n_phys:]
        self.X_data = np.asarray(X_data)
        self.X_phys = np.asarray(X_phys)
        self.b = float(np.ravel(con.dual_value)[0])
        return self

    def define_cvxpylayer(self, n_data, n_phys):
        n = 2 * (n_data + n_phys)
        x = Variable(n)
        Lt = Parameter((n, n))
        q = Parameter(n)
        A_eq = Parameter((1, n))
        b = Parameter(1)
        lb = Parameter(n)
        ub = Parameter(n)
        eq_con = A_eq @ x == b          # keep a handle to the constraint
        prob = Problem(Minimize(0.5 * sum_squares(Lt @ x) + q @ x),
                    [x >= lb, x <= ub, eq_con])
        return CvxpyLayer(prob, parameters=[Lt, q, A_eq, b, lb, ub],
                        variables=[x, eq_con.dual_variables[0]])
        
 
    def predict(self, X_test):
        u = jnp.asarray(self.alpha - self.alpha_)
        v = jnp.asarray(self.beta - self.beta_)
        out = (self.RBF(X_test, self.X_data, self.lam) @ u
               + self.Lx2RBF(X_test, self.X_phys, self.lam) @ v + self.b)
        return np.asarray(out)