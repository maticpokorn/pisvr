import numpy as np
import sympy as sp
from sklearn.model_selection import KFold

from pisvr import PDE, PISVR


def test_collocation_objective_uses_full_data_shape():
    diffusion_x = sp.Symbol("diffusion_x", real=True)
    diffusion_y = sp.Symbol("diffusion_y", real=True)

    def operator(expr, variables, coefficients):
        return coefficients[0] * sp.diff(expr, variables[0], 2) + coefficients[1] * sp.diff(expr, variables[1], 2)

    def exact_solution(points):
        x, y = points[:, 0], points[:, 1]
        return np.sin(x) * np.sin(y)

    pde = PDE(
        operator,
        lambda point: -(0.8 + 0.4) * np.sin(point[0]) * np.sin(point[1]),
        hyperparams={diffusion_x: 0.8, diffusion_y: 0.4},
    )
    model = PISVR(
        m=2,
        lam=0.7,
        eps1=0.02,
        C1=10.0,
        eps2=0.01,
        C2=10.0,
        pde=pde,
    )

    rng = np.random.default_rng(7)
    X_data = rng.uniform(0.0, np.pi, size=(22, 2))
    y_data = exact_solution(X_data)
    phys_axis = np.linspace(0.0, np.pi, 6)
    X_phys = np.column_stack([axis.ravel() for axis in np.meshgrid(phys_axis, phys_axis)])
    model.X_data = np.asarray(X_data)
    model.y_data = np.asarray(y_data)
    model.X_phys = np.asarray(X_phys)
    model.optimize_pde_hyperparams = True
    model.svr_hyperparams = []
    model.lam_index = None
    model.C1_index = None
    model.C2_index = None
    model.split = list(KFold(n_splits=5, shuffle=True, random_state=42).split(model.X_data))
    model.layers = [
        model.define_cvxpylayer(len(train_idx), len(model.X_phys))
        for train_idx, _ in model.split
    ]

    value = float(model.forward_pass_collocation_mse(np.asarray([0.8, 0.4], dtype=float)))
    assert np.isfinite(value)


def test_fixed_pde_parameter():
    coefficient = sp.Symbol("coefficient", real=True)

    def operator(expr, variables, coefficients=None):
        coefficient_value = 2.0 if coefficients is None else coefficients[0]
        return sp.diff(expr, variables[0]) + coefficient_value * expr

    pde = PDE(operator, lambda _x: 0.0, hyperparams={coefficient: 2.0})
    model = PISVR(
        m=1,
        lam=1.0,
        eps1=0.02,
        C1=10.0,
        eps2=0.0,
        C2=10.0,
        pde=pde,
    )

    X_data = np.linspace(0.0, 2.0, 8).reshape(-1, 1)
    y_data = np.exp(-2.0 * X_data[:, 0])
    X_phys = np.linspace(0.0, 2.0, 8).reshape(-1, 1)
    X_test = np.linspace(0.0, 2.0, 20).reshape(-1, 1)

    assert model.L1 == 2.0
    model.fit(X_data, y_data, X_phys)
    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((predictions - np.exp(-2.0 * X_test[:, 0])) ** 2))
    print(f"Fixed PDE coefficient: {coefficient} = {pde.hyperparams[coefficient]:.6g}")
    print(f"Fixed-parameter prediction RMSE: {rmse:.6g}")

    assert predictions.shape == (len(X_test),)
    assert np.all(np.isfinite(predictions))


def test_optimize_pde_parameter():
    coefficient = sp.Symbol("diffusion", real=True)
    true_coefficient = 0.8

    def operator(expr, variables, coefficients):
        return coefficients[0] * sp.diff(expr, variables[0], 2)

    pde = PDE(
        operator,
        lambda x: -true_coefficient * np.sin(x[0]),
        hyperparams={coefficient: 0.7},
    )
    model = PISVR(
        m=1,
        lam=0.8,
        eps1=0.02,
        C1=10.0,
        eps2=0.01,
        C2=10.0,
        pde=pde,
    )

    X_data = np.linspace(0.1, np.pi - 0.1, 12).reshape(-1, 1)
    y_data = np.sin(X_data[:, 0])
    X_phys = np.linspace(0.0, np.pi, 10).reshape(-1, 1)

    model.fit_cvxpylayer_bfgs_qp(
        X_data,
        y_data,
        X_phys,
    )

    estimate = model.pde_hyperparams[coefficient]
    print(f"Optimized PDE coefficient: {coefficient} = {estimate:.6g} "
          f"(synthetic target: {true_coefficient:.6g})")
    assert np.isfinite(estimate)
    assert np.isclose(model.pde_hyperparam_values[0], estimate)


def test_2d_anisotropic_poisson():
    diffusion_x = sp.Symbol("diffusion_x", real=True)
    diffusion_y = sp.Symbol("diffusion_y", real=True)
    value_x, value_y = 0.8, 0.4

    def operator(expr, variables, coefficients):
        return (
            coefficients[0] * sp.diff(expr, variables[0], 2)
            + coefficients[1] * sp.diff(expr, variables[1], 2)
        )

    def exact_solution(points):
        return np.sin(points[:, 0]) * np.sin(points[:, 1])

    pde = PDE(
        operator,
        lambda point: -(value_x + value_y) * np.sin(point[0]) * np.sin(point[1]),
        hyperparams={diffusion_x: value_x, diffusion_y: value_y},
    )
    model = PISVR(
        m=2,
        lam=0.7,
        eps1=0.02,
        C1=10.0,
        eps2=0.01,
        C2=10.0,
        pde=pde,
    )

    rng = np.random.default_rng(24)
    X_data = rng.uniform(0.0, np.pi, size=(16, 2))
    y_data = exact_solution(X_data)
    grid = np.linspace(0.0, np.pi, 5)
    X_phys = np.column_stack([axis.ravel() for axis in np.meshgrid(grid, grid)])
    test_grid = np.linspace(0.0, np.pi, 8)
    X_test = np.column_stack(
        [axis.ravel() for axis in np.meshgrid(test_grid, test_grid)]
    )

    model.fit(X_data, y_data, X_phys)
    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((predictions - exact_solution(X_test)) ** 2))
    print(
        "2D anisotropic Poisson coefficients: "
        f"diffusion_x={value_x:.3g}, diffusion_y={value_y:.3g}"
    )
    print(f"2D anisotropic Poisson prediction RMSE: {rmse:.6g}")

    assert predictions.shape == (len(X_test),)
    assert np.all(np.isfinite(predictions))


def test_optimize_2d_pde_coefficients():
    diffusion_x = sp.Symbol("diffusion_x", real=True)
    diffusion_y = sp.Symbol("diffusion_y", real=True)
    true_x, true_y = 0.8, 0.4

    def operator(expr, variables, coefficients):
        return (
            coefficients[0] * sp.diff(expr, variables[0], 2)
            + coefficients[1] * sp.diff(expr, variables[1], 2)
        )

    def exact_solution(points):
        x, y = points[:, 0], points[:, 1]
        return np.sin(x) * np.sin(y) + 0.35 * np.sin(2 * x) * np.sin(3 * y)

    def forcing(point):
        x, y = point[0], point[1]
        return (
            -(true_x + true_y) * np.sin(x) * np.sin(y)
            - 0.35 * (4 * true_x + 9 * true_y) * np.sin(2 * x) * np.sin(3 * y)
        )

    pde = PDE(
        operator,
        forcing,
        hyperparams={diffusion_x: 0.75, diffusion_y: 0.3},
    )
    model = PISVR(
        m=2,
        lam=0.7,
        eps1=0.02,
        C1=10.0,
        eps2=0.01,
        C2=10.0,
        pde=pde,
    )

    rng = np.random.default_rng(42)
    X_data = rng.uniform(0.0, np.pi, size=(20, 2))
    y_data = exact_solution(X_data)
    phys_axis = np.linspace(0.0, np.pi, 20)
    X_phys = np.column_stack(
        [axis.ravel() for axis in np.meshgrid(phys_axis, phys_axis)]
    )
    test_axis = np.linspace(0.0, np.pi, 8)
    X_test = np.column_stack(
        [axis.ravel() for axis in np.meshgrid(test_axis, test_axis)]
    )

    model.fit_cvxpylayer_bfgs_qp(
        X_data,
        y_data,
        X_phys,
    )

    predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((predictions - exact_solution(X_test)) ** 2))
    estimate_x = model.pde_hyperparams[diffusion_x]
    estimate_y = model.pde_hyperparams[diffusion_y]
    print(
        "Optimized 2D PDE coefficients: "
        f"diffusion_x={estimate_x:.6g} (target {true_x:.6g}), "
        f"diffusion_y={estimate_y:.6g} (target {true_y:.6g})"
    )
    print(f"Optimized 2D PDE prediction RMSE: {rmse:.6g}")

    assert np.isfinite(estimate_x)
    assert np.isfinite(estimate_y)
    assert predictions.shape == (len(X_test),)
    assert np.all(np.isfinite(predictions))


if __name__ == "__main__":
    test_fixed_pde_parameter()
    test_optimize_pde_parameter()
    test_2d_anisotropic_poisson()
    test_optimize_2d_pde_coefficients()
    print("All PDE hyperparameter tests passed.")
