import os

import numpy as np
import sympy as sp
import jax
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import KFold

from pisvr import PDE, PISVR


OUT_DIR = "diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)


def make_problem():
    diffusion_x = sp.Symbol("diffusion_x", real=True)
    diffusion_y = sp.Symbol("diffusion_y", real=True)
    true_params = np.asarray([0.8, 0.4], dtype=float)

    def operator(expr, variables, coefficients):
        return (
            coefficients[0] * sp.diff(expr, variables[0], 2)
            + coefficients[1] * sp.diff(expr, variables[1], 2)
        )

    def exact_solution(points):
        x = points[:, 0]
        y = points[:, 1]
        return np.sin(x) * np.sin(y) + 0.35 * np.sin(2 * x) * np.sin(3 * y)

    def forcing(point):
        x = point[0]
        y = point[1]
        return (
            -(true_params[0] + true_params[1]) * np.sin(x) * np.sin(y)
            - 0.35 * (4 * true_params[0] + 9 * true_params[1])
            * np.sin(2 * x) * np.sin(3 * y)
        )

    pde = PDE(
        operator,
        forcing,
        hyperparams={diffusion_x: float(true_params[0]), diffusion_y: float(true_params[1])},
    )

    rng = np.random.default_rng(42)
    X_data = rng.uniform(0.0, np.pi, size=(60, 2))
    y_data = exact_solution(X_data)

    # Enforce the homogeneous Dirichlet boundary condition by adding zero-valued
    # samples on the square boundary x=0, x=pi, y=0, and y=pi.
    boundary_axis = np.linspace(0.0, np.pi, 12)
    boundary_points = []
    for value in boundary_axis:
        boundary_points.append((0.0, value))
        boundary_points.append((np.pi, value))
        boundary_points.append((value, 0.0))
        boundary_points.append((value, np.pi))
    X_boundary = np.asarray(boundary_points, dtype=float)
    y_boundary = exact_solution(X_boundary)

    X_data = np.vstack([X_data, X_boundary])
    y_data = np.concatenate([y_data, y_boundary])

    phys_axis = np.linspace(0.0, np.pi, 10)
    X_phys = np.column_stack([axis.ravel() for axis in np.meshgrid(phys_axis, phys_axis)])

    model = PISVR(
        m=2,
        lam=0.3,
        eps1=0.01,
        C1=10.0,
        eps2=0.0,
        C2=10.0,
        pde=pde,
    )
    model.X_data = np.asarray(X_data)
    model.y_data = np.asarray(y_data)
    model.X_phys = np.asarray(X_phys)
    return model, (diffusion_x, diffusion_y), true_params, exact_solution


def make_objective(model):
    def objective(params):
        params = np.asarray(params, dtype=float)
        model.pde_hyperparam_values = [float(v) for v in params]
        model.pde_hyperparams = dict(zip(model.pde_hyperparam_sym, model.pde_hyperparam_values))
        model.fit(model.X_data, model.y_data, model.X_phys)
        return model.obj

    return objective


def plot_loss_landscapes():
    model, hyperparams, true_params, _exact_solution = make_problem()
    objective = make_objective(model)

    n_splits = 5
    model.optimize_pde_hyperparams = True
    model.svr_hyperparams = []
    model.lam_index = None
    model.C1_index = None
    model.C2_index = None
    model.split = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=42).split(model.X_data)
    )
    model.layers = [
        model.define_cvxpylayer(len(train_idx), len(model.X_phys))
        for train_idx, _ in model.split
    ]

    diffusion_x_values = np.linspace(0.2, 2.0, 12)
    diffusion_y_values = np.linspace(0.05, 1.0, 12)
    qp_objective = np.empty((len(diffusion_x_values), len(diffusion_y_values)))
    collocation_mse = np.empty_like(qp_objective)

    candidates = [
        (float(dx), float(dy))
        for dx in diffusion_x_values
        for dy in diffusion_y_values
    ]
    for dx, dy in tqdm(candidates, desc="Evaluating 2-parameter landscape"):
        i = int(np.argmin(np.abs(diffusion_x_values - dx)))
        j = int(np.argmin(np.abs(diffusion_y_values - dy)))
        params = np.asarray([dx, dy], dtype=float)
        model.pde_hyperparam_values = [float(dx), float(dy)]
        model.pde_hyperparams = dict(zip(model.pde_hyperparam_sym, model.pde_hyperparam_values))
        model.fit(model.X_data, model.y_data, model.X_phys)
        qp_objective[i, j] = model.obj
        collocation_mse[i, j] = float(model.forward_pass_collocation_mse(params))

    x_grid, y_grid = np.meshgrid(diffusion_x_values, diffusion_y_values, indexing="ij")

    for title, values, filename, show_traj in [
        ("PISVR quadratic objective", qp_objective, "loss_landscape_2param_qp.png", False),
        ("collocation MSE", collocation_mse, "loss_landscape_2param_collocation_mse.png", True),
    ]:
        minimum_index = np.unravel_index(np.argmin(values), values.shape)
        best_x = float(diffusion_x_values[minimum_index[0]])
        best_y = float(diffusion_y_values[minimum_index[1]])
        grid_min = np.asarray([best_x, best_y], dtype=float)

        if np.all(values > 0.0):
            color_values = values
            colorbar_label = f"{title} (log scale)"
        else:
            floor = max(float(np.ptp(values)) * 1e-8, np.finfo(float).eps)
            color_values = values - float(np.min(values)) + floor
            colorbar_label = f"log excess of {title} above its minimum"

        color_min = float(np.min(color_values))
        color_max = float(np.max(color_values))
        if np.isclose(color_min, color_max, rtol=1e-12, atol=0.0):
            norm_min, norm_max = color_min / 1.01, color_max * 1.01
        else:
            norm_min, norm_max = color_min, color_max

        fig, ax = plt.subplots(figsize=(8, 6))
        filled = ax.contourf(
            x_grid,
            y_grid,
            color_values,
            levels=np.geomspace(norm_min, norm_max, 30),
            norm=LogNorm(vmin=norm_min, vmax=norm_max),
            cmap="viridis",
        )
        ax.scatter(best_x, best_y, color="red", marker="x", s=80, linewidths=2,
                   label=f"grid minimum ({best_x:.3g}, {best_y:.3g})")

        if show_traj:
            optimization_starts = [
                np.asarray([0.5, 0.2]),
            ]
            parameter_bounds = [
                (float(diffusion_x_values[0]), float(diffusion_x_values[-1])),
                (float(diffusion_y_values[0]), float(diffusion_y_values[-1])),
            ]
            optimization_runs = []
            for start in optimization_starts:
                trajectory = [(start.copy(), float(model.forward_pass_collocation_mse(start)))]

                def record_step(params, path=trajectory):
                    params = np.asarray(params, dtype=float).copy()
                    loss = float(model.forward_pass_collocation_mse(params))
                    path.append((params, loss))

                result = minimize(
                    model.forward_pass_collocation_mse,
                    x0=np.asarray(start, dtype=float),
                    method="L-BFGS-B",
                    bounds=parameter_bounds,
                    jac=jax.grad(model.forward_pass_collocation_mse, argnums=0),
                    callback=record_step,
                    options={"maxls": 10},
                )
                path = np.asarray([p for p, _ in trajectory], dtype=float)
                optimization_runs.append((np.asarray(result.x), float(result.fun), path))

            colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(optimization_runs)))
            for run_index, ((final_params, final_loss, path), color) in enumerate(
                zip(optimization_runs, colors), start=1
            ):
                ax.plot(
                    path[:, 0], path[:, 1],
                    color=color,
                    linewidth=1.7,
                    marker="o",
                    markersize=3.5,
                    markeredgecolor="black",
                    label=f"run {run_index} (collocation MSE={final_loss:.3g})",
                    zorder=5,
                )
                ax.scatter(*path[0], color=color, edgecolor="black", marker="s", s=48, zorder=6)
                ax.scatter(*path[-1], color=color, edgecolor="black", marker="*", s=90, zorder=6)

        ax.axvline(true_params[0], color="white", linestyle="--", linewidth=1.2,
                   label=f"true diffusion_x {true_params[0]:g}")
        ax.axhline(true_params[1], color="white", linestyle="--", linewidth=1.2,
                   label=f"true diffusion_y {true_params[1]:g}")

        ax.set_title(f"2D PDE: {title}")
        ax.set_xlabel(r"diffusion coefficient in $x$")
        ax.set_ylabel(r"diffusion coefficient in $y$")
        fig.colorbar(filled, ax=ax, label=colorbar_label)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=180)
        plt.close(fig)

        print(f"{title} minimum: diffusion_x={best_x:.6g}, diffusion_y={best_y:.6g}")


def plot_solution_comparison():
    model, hyperparams, true_params, exact_solution = make_problem()
    n_splits = 5
    model.optimize_pde_hyperparams = True
    model.svr_hyperparams = []
    model.lam_index = None
    model.C1_index = None
    model.C2_index = None
    model.split = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=42).split(model.X_data)
    )
    model.layers = [
        model.define_cvxpylayer(len(train_idx), len(model.X_phys))
        for train_idx, _ in model.split
    ]

    diffusion_x_values = np.linspace(0.2, 2.0, 12)
    diffusion_y_values = np.linspace(0.05, 1.0, 12)
    collocation_mse = np.empty((len(diffusion_x_values), len(diffusion_y_values)))
    for i, dx in enumerate(diffusion_x_values):
        for j, dy in enumerate(diffusion_y_values):
            params = np.asarray([dx, dy], dtype=float)
            model.pde_hyperparam_values = [float(dx), float(dy)]
            model.pde_hyperparams = dict(zip(model.pde_hyperparam_sym, model.pde_hyperparam_values))
            collocation_mse[i, j] = float(model.forward_pass_collocation_mse(params))

    best_idx = np.unravel_index(np.argmin(collocation_mse), collocation_mse.shape)
    optimized_params = np.asarray([diffusion_x_values[best_idx[0]], diffusion_y_values[best_idx[1]]], dtype=float)

    solution_axis = np.linspace(0.0, np.pi, 60)
    X_grid, Y_grid = np.meshgrid(solution_axis, solution_axis)
    X_eval = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    exact_values = exact_solution(X_eval)

    model.pde_hyperparam_values = [float(v) for v in optimized_params]
    model.pde_hyperparams = dict(zip(model.pde_hyperparam_sym, model.pde_hyperparam_values))
    model.fit(model.X_data, model.y_data, model.X_phys)
    optimized_values = model.predict(X_eval)

    model.pde_hyperparam_values = [float(v) for v in true_params]
    model.pde_hyperparams = dict(zip(model.pde_hyperparam_sym, model.pde_hyperparam_values))
    model.fit(model.X_data, model.y_data, model.X_phys)
    true_values = model.predict(X_eval)

    optimized_rmse = float(np.sqrt(np.mean((optimized_values - exact_values) ** 2)))
    true_rmse = float(np.sqrt(np.mean((true_values - exact_values) ** 2)))
    exact_grid = exact_values.reshape(X_grid.shape)
    optimized_grid = optimized_values.reshape(X_grid.shape)
    true_grid = true_values.reshape(X_grid.shape)

    shared_min = min(float(exact_grid.min()), float(optimized_grid.min()), float(true_grid.min()))
    shared_max = max(float(exact_grid.max()), float(optimized_grid.max()), float(true_grid.max()))
    error_true = np.abs(true_grid - exact_grid)
    error_optimized = np.abs(optimized_grid - exact_grid)
    error_max = max(float(error_true.max()), float(error_optimized.max()), np.finfo(float).eps)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    field_levels = np.linspace(shared_min, shared_max, 25)
    field_panels = [
        ("Exact", exact_grid),
        (f"true PDE params ({true_params[0]:.3g}, {true_params[1]:.3g})", true_grid),
        (f"optimized params ({optimized_params[0]:.3g}, {optimized_params[1]:.3g})", optimized_grid),
    ]
    for axis, (title, values) in zip(axes[0], field_panels):
        image = axis.contourf(X_grid, Y_grid, values, levels=field_levels,
                              cmap="RdBu_r", extend="both")
        axis.scatter(model.X_data[:, 0], model.X_data[:, 1], c="black", s=10)
        axis.scatter(model.X_phys[:, 0], model.X_phys[:, 1], marker="+",
                     c="tab:green", s=20)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(image, ax=axes[0], shrink=0.85, label="u(x, y)")

    error_levels = np.linspace(0.0, error_max, 25)
    error_panels = [
        (f"true parameters absolute error (RMSE={true_rmse:.3f})", error_true),
        (f"optimized parameters absolute error (RMSE={optimized_rmse:.3f})", error_optimized),
    ]
    for axis, (title, values) in zip(axes[1, :2], error_panels):
        error_image = axis.contourf(X_grid, Y_grid, values, levels=error_levels,
                                    cmap="viridis", extend="max")
        axis.scatter(model.X_data[:, 0], model.X_data[:, 1], c="white", s=9,
                     edgecolors="black", linewidths=0.4)
        axis.scatter(model.X_phys[:, 0], model.X_phys[:, 1], marker="+",
                     c="tab:red", s=20)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(error_image, ax=axes[1, :2], shrink=0.85,
                 label="absolute error |u_pred - u_exact|")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.6,
        f"true params: ({true_params[0]:.4g}, {true_params[1]:.4g})\n"
        f"optimized params: ({optimized_params[0]:.4g}, {optimized_params[1]:.4g})\n\n"
        f"RMSE true params: {true_rmse:.4f}\n"
        f"RMSE optimized params: {optimized_rmse:.4f}",
        fontsize=11, family="monospace", va="center",
    )
    fig.savefig(os.path.join(OUT_DIR, "solution_comparison_2param.png"), dpi=150)
    plt.close(fig)

    print(
        f"True params: diffusion_x={true_params[0]:.6g}, diffusion_y={true_params[1]:.6g}\n"
        f"Optimized params: diffusion_x={optimized_params[0]:.6g}, diffusion_y={optimized_params[1]:.6g}\n"
        f"RMSE (true params): {true_rmse:.6g}\n"
        f"RMSE (optimized params): {optimized_rmse:.6g}"
    )


if __name__ == "__main__":
    plot_loss_landscapes()
    plot_solution_comparison()
    print(f"Saved plots in {OUT_DIR}")
