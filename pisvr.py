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


class PDE:
    def __init__(self, L, f, hyperparams={}):
        self.L = L
        self.f = f
        self.hyperparams = hyperparams

class PISVR:
 
    def __init__(self, m : int, lam : float, eps1 :float, C1 : float, eps2 : float, C2 : float, pde : PDE):
        self.m = m
        self.lam = float(lam)
        self.eps1 = float(eps1)
        self.C1 = float(C1)
        self.eps2 = float(eps2)
        self.C2 = float(C2)
        self.L = pde.L
        self.f = pde.f
        self.pde_hyperparams = pde.hyperparams

        self.pde_hyperparam_sym = list(self.pde_hyperparams.keys())
        self.pde_hyperparam_values = list(self.pde_hyperparams.values())

        self.x1 = sp.Matrix(sp.symbols(f"x1_1:{m+1}", real=True))
        self.x2 = sp.Matrix(sp.symbols(f"x2_1:{m+1}", real=True))
        self.lam_sym = sp.Symbol("lam", positive=True)
        k = sp.exp(-self.lam_sym * (self.x1 - self.x2).dot(self.x1 - self.x2))

        if self.pde_hyperparams:
            args = (list(self.x1), list(self.x2), self.lam_sym, list(self.pde_hyperparam_sym))
        else:
            args = (list(self.x1), list(self.x2), self.lam_sym, [])

        if self.pde_hyperparam_sym:
            Lx1_k = self.L(k, self.x1, self.pde_hyperparam_sym)
            Lx2_k = self.L(k, self.x2, self.pde_hyperparam_sym)
            Lx1Lx2_k = self.L(Lx1_k, self.x2, self.pde_hyperparam_sym)
            L1 = self.L(sp.Integer(1), self.x1, self.pde_hyperparam_sym)
        else:
            Lx1_k = self.L(k, self.x1)
            Lx2_k = self.L(k, self.x2)
            Lx1Lx2_k = self.L(Lx1_k, self.x2)
            L1 = self.L(sp.Integer(1), self.x1)

        self._K = self._gram_fn(sp.lambdify(args, k, "jax"))
        self._Lx1K = self._gram_fn(sp.lambdify(args, Lx1_k, "jax"))
        self._Lx2K = self._gram_fn(sp.lambdify(args, Lx2_k, "jax"))
        self._Lx1Lx2K = self._gram_fn(sp.lambdify(args, Lx1Lx2_k, "jax"))
        if self.pde_hyperparam_sym:
            L1 = L1.subs(dict(zip(self.pde_hyperparam_sym, self.pde_hyperparam_values)))
        self.L1 = float(L1)
    
    @staticmethod
    def _gram_fn(fun):
        row = jax.vmap(fun, in_axes=(None, 0, None, None))
        return jax.jit(jax.vmap(row, in_axes=(0, None, None, None)))
 
    def RBF(self, X1, X2, lam, pde_hyperparams=[]):
        return self._K(_arr(X1), _arr(X2), lam, pde_hyperparams)
 
    def Lx1RBF(self, X1, X2, lam, pde_hyperparams=[]):
        return self._Lx1K(_arr(X1), _arr(X2), lam, pde_hyperparams)
 
    def Lx2RBF(self, X1, X2, lam, pde_hyperparams=[]):
        return self._Lx2K(_arr(X1), _arr(X2), lam, pde_hyperparams)
 
    def Lx1Lx2RBF(self, X1, X2, lam, pde_hyperparams=[]):
        return self._Lx1Lx2K(_arr(X1), _arr(X2), lam, pde_hyperparams)
 
    @staticmethod
    def block(M):
        return jnp.block([[M, -M], [-M, M]])
 
    def P(self, X_data, X_phys, lam, pde_hyperparams=[]):
        K11 = self.block(self.RBF(X_data, X_data, lam, pde_hyperparams))
        K12 = self.block(self.Lx2RBF(X_data, X_phys, lam, pde_hyperparams))
        K21 = self.block(self.Lx1RBF(X_phys, X_data, lam, pde_hyperparams))
        K22 = self.block(self.Lx1Lx2RBF(X_phys, X_phys, lam, pde_hyperparams))
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
 
    def qp_data(self, X_data, y_data, X_phys, lam, C1, C2, eps1, eps2, pde_hyperparams=[]):
        n_data, n_phys = len(y_data), len(X_phys)
        A_eq, b = self.eq(n_data, n_phys)
        lb, ub = self.bounds(n_data, n_phys, C1, C2)
        return dict(P=self.P(X_data, X_phys, lam, pde_hyperparams),
                    q=self.q(y_data, X_phys, eps1, eps2),
                    A_eq=A_eq, b=b, lb=lb, ub=ub)

    def forward_pass_quad_loss(self, params):
        
        n_data, n_phys = len(self.X_data), len(self.X_phys)
        layer = self.define_cvxpylayer(n_data, n_phys)
        P = self.P(self.X_data, self.X_phys, self.lam, params)
        Lt = jnp.linalg.cholesky(P + 1e-6 * jnp.eye(P.shape[0])).T
        q = self.q(self.y_data, self.X_phys, self.eps1, self.eps2)
        A_eq, b = self.eq(n_data, n_phys)
        lb, ub = self.bounds(n_data, n_phys, self.C1, self.C2)
        x, mu = layer(Lt, q, A_eq, b, lb, ub)
        return self.cost_fn(x, Lt, q)
        

    def forward_pass(self, params):


        if self.optimize_pde_hyperparams:
            pde_hyperparams = params[:len(self.pde_hyperparams)]
        else:
            pde_hyperparams = self.pde_hyperparam_values

        if "lam" in self.svr_hyperparams:
            lam = params[self.lam_index]
        else:
            lam = self.lam
        if "C1" in self.svr_hyperparams:
            C1 = params[self.C1_index]
        else:
            C1 = self.C1
        if "C2" in self.svr_hyperparams:
            C2 = params[self.C2_index]
        else:
            C2 = self.C2
        n_data, n_phys = len(self.X_data), len(self.X_phys)
        fold_mses = []
        for fold, (train_idx, val_idx) in enumerate(self.split):
            X_train, X_val = self.X_data[train_idx], self.X_data[val_idx]
            y_train, y_val = self.y_data[train_idx], self.y_data[val_idx]
            n_data_fold = len(y_train)
            P = self.P(X_train, self.X_phys, lam, pde_hyperparams)
            Lt = jnp.linalg.cholesky(P + 1e-6 * jnp.eye(P.shape[0])).T
            q = self.q(y_train, self.X_phys, self.eps1, self.eps2)
            A_eq, b = self.eq(n_data_fold, n_phys)
            lb, ub = self.bounds(n_data_fold, n_phys, C1, C2)
            x, mu = self.layers[fold](Lt, q, A_eq, b, lb, ub)
            alpha = x[:n_data_fold] - x[n_data_fold:2 * n_data_fold]
            beta = x[2 * n_data_fold:2 * n_data_fold + n_phys] - x[2 * n_data_fold + n_phys:]

            y_pred = (self.RBF(X_val, X_train, lam, pde_hyperparams) @ alpha
                      + self.Lx2RBF(X_val, self.X_phys, lam, pde_hyperparams) @ beta + mu[0])
            mean_squared_error = jnp.mean((y_val - y_pred) ** 2)
            #print(f"Fold {fold + 1}/{n_splits}: Mean Squared Error = {mean_squared_error}")
            fold_mses.append(mean_squared_error)
        return jnp.mean(jnp.array(fold_mses))

    def forward_pass_collocation_mse(self, params):
        """Weighted data + PDE-residual objective: lambda_data=C1, lambda_phys=C2."""
        if self.optimize_pde_hyperparams:
            pde_hyperparams = params[:len(self.pde_hyperparams)]
        else:
            pde_hyperparams = self.pde_hyperparam_values

        if "lam" in self.svr_hyperparams:
            lam = params[self.lam_index]
        else:
            lam = self.lam
        if "C1" in self.svr_hyperparams:
            C1 = params[self.C1_index]
        else:
            C1 = self.C1
        if "C2" in self.svr_hyperparams:
            C2 = params[self.C2_index]
        else:
            C2 = self.C2

        n_data, n_phys = len(self.X_data), len(self.X_phys)
        layer_key = (n_data, n_phys)
        if not hasattr(self, "_collocation_layer") or self._collocation_layer is None:
            self._collocation_layer = {}
        if layer_key not in self._collocation_layer:
            self._collocation_layer[layer_key] = self.define_cvxpylayer(n_data, n_phys)

        P = self.P(self.X_data, self.X_phys, lam, pde_hyperparams)
        Lt = jnp.linalg.cholesky(P + 1e-6 * jnp.eye(P.shape[0])).T
        q = self.q(self.y_data, self.X_phys, self.eps1, self.eps2)
        A_eq, b = self.eq(n_data, n_phys)
        lb, ub = self.bounds(n_data, n_phys, C1, C2)
        x, mu = self._collocation_layer[layer_key](Lt, q, A_eq, b, lb, ub)
        alpha = x[:n_data] - x[n_data:2 * n_data]
        beta = x[2 * n_data:2 * n_data + n_phys] - x[2 * n_data + n_phys:]

        data_pred = (
            self.RBF(self.X_data, self.X_data, lam, pde_hyperparams) @ alpha
            + self.Lx2RBF(self.X_data, self.X_phys, lam, pde_hyperparams) @ beta
            + mu[0]
        )
        data_mse = jnp.mean((self.y_data - data_pred) ** 2)

        pde_err = (
            self.Lx1RBF(self.X_phys, self.X_data, lam, pde_hyperparams) @ alpha
            + self.Lx1Lx2RBF(self.X_phys, self.X_phys, lam, pde_hyperparams) @ beta
            + self.L1 * mu[0]
            - self.f_values(self.X_phys)
        )
        pde_mse = jnp.mean(pde_err ** 2)
        return 100.0 * data_mse + 1.0 * pde_mse

    def fit_cvxpylayer_bfgs_qp(self, X_data, y_data, X_phys):
        self.X_data = jnp.asarray(X_data)
        self.y_data = jnp.asarray(y_data)
        self.X_phys = jnp.asarray(X_phys)
        x0 = list(self.pde_hyperparams.values())
        result = minimize(
            self.forward_pass_quad_loss,
            x0=x0,
            method="L-BFGS-B",
            jac=jax.grad(self.forward_pass_quad_loss, argnums=(0,)),
        )
        x = result.x
        self.pde_hyperparam_values = list(map(float, x))
        self.pde_hyperparams = dict(zip(self.pde_hyperparam_sym, self.pde_hyperparam_values))
        return self.fit(self.X_data, self.y_data, self.X_phys)


    def fit_cvxpylayer_bfgs(
        self, X_data, y_data, X_phys, n_splits=10,
        optimize_pde_hyperparams=False, svr_hyperparams=None,
    ):
            self.optimize_pde_hyperparams = optimize_pde_hyperparams
            self.svr_hyperparams = svr_hyperparams
            
            if svr_hyperparams is None and not optimize_pde_hyperparams:
                raise ValueError("No hyperparameters provided for optimization.")
            
            self.X_data = jnp.asarray(X_data)
            self.y_data = jnp.asarray(y_data)
            self.X_phys = jnp.asarray(X_phys)

            x0 = []
            bounds = []
            if optimize_pde_hyperparams:
                if self.pde_hyperparams is None:
                    raise ValueError("PDE hyperparameters are not defined. Cannot optimize them.")
                else:
                    x0.extend(self.pde_hyperparams.values())
                    bounds.extend([(None, None)] * len(self.pde_hyperparams))
            self.lam_index = None
            self.C1_index = None
            self.C2_index = None
            if svr_hyperparams is not None:
                if 'lam' in svr_hyperparams:
                    self.lam_index = len(x0)
                    x0.append(self.lam)
                    bounds.append((1e-6, None))
                if 'C1' in svr_hyperparams:
                    self.C1_index = len(x0)
                    x0.append(self.C1)
                    bounds.append((0.0, None))
                if 'C2' in svr_hyperparams:
                    self.C2_index = len(x0)
                    x0.append(self.C2)
                    bounds.append((0.0, None))


            n_data, n_phys = len(self.X_data), len(self.X_phys)
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            self.split = list(kfold.split(self.X_data))
            self.layers = [self.define_cvxpylayer(len(train_idx), n_phys) for train_idx, _ in self.split]

            print(
                "Starting BFGS optimization from x0; "
                f"parameters={x0}, bounds={bounds}"
            )

            def report_bfgs_state(params):
                params = np.asarray(params, dtype=float)
                loss = float(self.forward_pass(params))
                print(
                    f"L-BFGS-B iter {len(self._bfgs_history):>2}: "
                    f"params={np.asarray(params).tolist()}, CV MSE={loss:.8g}"
                )
                self._bfgs_history.append((params.copy(), loss))

            self._bfgs_history = []
            result = minimize(
                self.forward_pass,
                x0=np.asarray(x0, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                jac=jax.grad(self.forward_pass, argnums=(0,)),
                callback=report_bfgs_state,
            )
            if not np.isfinite(result.fun):
                raise RuntimeError("BFGS optimization did not produce a finite objective value.")
            print(
                f"BFGS result: CV MSE={result.fun:.6g}, success={result.success}: "
                f"{result.message}"
            )
            x = result.x

            if self.lam_index is not None:
                self.lam = x[self.lam_index]
            if self.C1_index is not None:
                self.C1 = x[self.C1_index]
            if self.C2_index is not None:
                self.C2 = x[self.C2_index]
            if optimize_pde_hyperparams:
                optimized_values = list(map(float, x[:len(self.pde_hyperparams)]))
                self.pde_hyperparam_values = optimized_values
                self.pde_hyperparams = dict(zip(self.pde_hyperparam_sym, optimized_values))
            return self.fit(self.X_data, self.y_data, self.X_phys)

    def fit_cvxpylayer_bfgs_collocation(
        self, X_data, y_data, X_phys, optimize_pde_hyperparams=False, svr_hyperparams=None,
    ):
        """Optimize the collocation MSE objective with L-BFGS-B without cross-validation."""
        self.optimize_pde_hyperparams = optimize_pde_hyperparams
        self.svr_hyperparams = svr_hyperparams

        if svr_hyperparams is None and not optimize_pde_hyperparams:
            raise ValueError("No hyperparameters provided for optimization.")

        self.X_data = jnp.asarray(X_data)
        self.y_data = jnp.asarray(y_data)
        self.X_phys = jnp.asarray(X_phys)

        x0 = []
        bounds = []
        if optimize_pde_hyperparams:
            if self.pde_hyperparams is None:
                raise ValueError("PDE hyperparameters are not defined. Cannot optimize them.")
            x0.extend(self.pde_hyperparams.values())
            bounds.extend([(None, None)] * len(self.pde_hyperparams))

        self.lam_index = None
        self.C1_index = None
        self.C2_index = None
        if svr_hyperparams is not None:
            if "lam" in svr_hyperparams:
                self.lam_index = len(x0)
                x0.append(self.lam)
                bounds.append((1e-6, None))
            if "C1" in svr_hyperparams:
                self.C1_index = len(x0)
                x0.append(self.C1)
                bounds.append((0.0, None))
            if "C2" in svr_hyperparams:
                self.C2_index = len(x0)
                x0.append(self.C2)
                bounds.append((0.0, None))

        n_data, n_phys = len(self.X_data), len(self.X_phys)
        self.layers = [self.define_cvxpylayer(n_data, n_phys)]
        self._collocation_layer = {(n_data, n_phys): self.layers[0]}

        print(
            "Starting collocation-only BFGS optimization from x0; "
            f"parameters={x0}, bounds={bounds}"
        )
        result = minimize(
            self.forward_pass_collocation_mse,
            x0=np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            jac=jax.grad(self.forward_pass_collocation_mse, argnums=(0,)),
        )
        if not np.isfinite(result.fun):
            raise RuntimeError("Collocation-only BFGS optimization did not produce a finite objective value.")

        print(
            f"Collocation BFGS result: MSE={result.fun:.6g}, success={result.success}: "
            f"{result.message}"
        )

        x = result.x
        if self.lam_index is not None:
            self.lam = x[self.lam_index]
        if self.C1_index is not None:
            self.C1 = x[self.C1_index]
        if self.C2_index is not None:
            self.C2 = x[self.C2_index]
        if optimize_pde_hyperparams:
            optimized_values = list(map(float, x[:len(self.pde_hyperparams)]))
            self.pde_hyperparam_values = optimized_values
            self.pde_hyperparams = dict(zip(self.pde_hyperparam_sym, optimized_values))
        return self.fit(self.X_data, self.y_data, self.X_phys)

    def fit_cvxpylayer_gradient_descent(
        self, X_data, y_data, X_phys, n_splits=10,
        optimize_pde_hyperparams=False, svr_hyperparams=None,
        learning_rate=1e-3, max_iter=1000, tol=1e-6,
    ):
        """Optimize CV MSE with unbounded normalized-gradient descent.

        Returns ``(fitted_model, history)``, where each history item is a
        ``(parameter_vector, CV_MSE)`` pair, including the initial point.
        """
        if svr_hyperparams is None and not optimize_pde_hyperparams:
            raise ValueError("No hyperparameters provided for optimization.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.optimize_pde_hyperparams = optimize_pde_hyperparams
        self.svr_hyperparams = svr_hyperparams
        self.X_data = jnp.asarray(X_data)
        self.y_data = jnp.asarray(y_data)
        self.X_phys = jnp.asarray(X_phys)

        params = []
        if optimize_pde_hyperparams:
            if not self.pde_hyperparams:
                raise ValueError("PDE hyperparameters are not defined for optimization.")
            params.extend(self.pde_hyperparams.values())

        self.lam_index = None
        self.C1_index = None
        self.C2_index = None
        if svr_hyperparams is not None:
            if "lam" in svr_hyperparams:
                self.lam_index = len(params)
                params.append(self.lam)
            if "C1" in svr_hyperparams:
                self.C1_index = len(params)
                params.append(self.C1)
            if "C2" in svr_hyperparams:
                self.C2_index = len(params)
                params.append(self.C2)

        if not params:
            raise ValueError("No parameters were selected for optimization.")

        n_phys = len(self.X_phys)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.split = list(kfold.split(self.X_data))
        self.layers = [
            self.define_cvxpylayer(len(train_idx), n_phys)
            for train_idx, _ in self.split
        ]

        current = np.asarray(params, dtype=float)
        history = [(current.copy(), float(self.forward_pass(current)))]
        print(
            f"Gradient descent step 0: params={current.tolist()}, "
            f"CV MSE={history[-1][1]:.8g}"
        )
        gradient_fn = jax.grad(self.forward_pass, argnums=0)
        for step in range(1, max_iter + 1):
            gradient = np.asarray(gradient_fn(current), dtype=float)
            if not np.all(np.isfinite(gradient)):
                raise RuntimeError(
                    f"Gradient descent produced a non-finite gradient at step {step}; "
                    f"params={current.tolist()}, CV MSE={history[-1][1]:.8g}."
                )
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm <= tol:
                break

            candidate = current - learning_rate * (gradient / gradient_norm)
            try:
                loss = float(self.forward_pass(candidate))
            except Exception as exc:
                self.gradient_descent_history = history
                print(
                    f"Gradient descent step {step} failed: "
                    f"params={candidate.tolist()}, error={exc}"
                )
                raise RuntimeError(
                    f"Gradient descent objective failed at step {step}; "
                    f"params={candidate.tolist()}."
                ) from exc
            if not np.isfinite(loss):
                self.gradient_descent_history = history + [(candidate.copy(), loss)]
                print(
                    f"Gradient descent step {step}: params={candidate.tolist()}, "
                    f"CV MSE={loss} (non-finite)"
                )
                raise RuntimeError(
                    f"Gradient descent produced a non-finite CV objective at step {step}; "
                    f"params={candidate.tolist()}. Try a smaller learning_rate."
                )
            current = candidate
            history.append((current.copy(), loss))
            print(
                f"Gradient descent step {step}: params={current.tolist()}, "
                f"CV MSE={loss:.8g}"
            )

        if self.lam_index is not None:
            self.lam = float(current[self.lam_index])
        if self.C1_index is not None:
            self.C1 = float(current[self.C1_index])
        if self.C2_index is not None:
            self.C2 = float(current[self.C2_index])
        if optimize_pde_hyperparams:
            values = list(map(float, current[:len(self.pde_hyperparams)]))
            self.pde_hyperparam_values = values
            self.pde_hyperparams = dict(zip(self.pde_hyperparam_sym, values))

        self.gradient_descent_history = history
        fitted_model = self.fit(self.X_data, self.y_data, self.X_phys)
        return fitted_model, history

    def cost_fn(self, x, Lt, q):
        return 0.5 * x @ Lt @ Lt.T @ x + q @ x
 
    def fit(self, X_data, y_data, X_phys):
        n_data, n_phys = len(X_data), len(X_phys)

        d = {k: np.asarray(v) for k, v in self.qp_data(
            X_data, y_data, X_phys, self.lam, self.C1, self.C2, self.eps1, self.eps2,
            self.pde_hyperparam_values,
        ).items()}
 
        x = Variable(2 * n_data + 2 * n_phys)
        con = d["A_eq"] @ x == d["b"]
        prob = Problem(Minimize(0.5 * quad_form(x, psd_wrap(d["P"])) + d["q"] @ x),
                       [x >= d["lb"], x <= d["ub"], con])
        prob.solve(solver="CLARABEL")

        # The dual-form QP optimum is the negative of the primal PISVR loss.
        self.obj = -float(prob.value)
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
        out = (self.RBF(X_test, self.X_data, self.lam, self.pde_hyperparam_values) @ u
               + self.Lx2RBF(X_test, self.X_phys, self.lam, self.pde_hyperparam_values) @ v + self.b)
        return np.asarray(out)