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


def make_pisvr_objective(model):
    """Refit PISVR at candidate parameters and return its quadratic loss."""
    def objective(params):
        model.pde_hyperparam_values = list(map(float, params))
        model.fit(model.X_data, model.y_data, model.X_phys)
        return model.obj

    return objective


def make_1d_problem():
    coeff = sp.Symbol("diffusion", real=True)
    true_coeff = 0.8

    def operator(expr, variables, coefficients):
        return coefficients[0] * sp.diff(expr, variables[0], 2)

    pde = PDE(
        operator,
        lambda x: -true_coeff * np.sin(x[0]),
        hyperparams={coeff: 0.7},
    )

    X_data = np.linspace(0.15, np.pi - 0.15, 10).reshape(-1, 1)
    y_data = np.sin(X_data[:, 0])
    X_phys = np.linspace(0.0, np.pi, 15).reshape(-1, 1)

    model = PISVR(
        m=1,
        lam=0.5,
        eps1=0.02,
        C1=10.0,
        eps2=0.0,
        C2=100.0,
        pde=pde,
    )
    model.X_data = np.asarray(X_data)
    model.y_data = np.asarray(y_data)
    model.X_phys = np.asarray(X_phys)
    return model, coeff, true_coeff


def plot_1d_landscape():
    model, coeff, true_coeff = make_1d_problem()
    objective = make_pisvr_objective(model)
    grid = np.linspace(0.6, 1.0, 120)
    losses = np.asarray([objective([float(val)]) for val in grid])

    best = grid[np.argmin(losses)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(grid, losses, color="tab:blue", lw=2, label="PISVR quadratic loss")
    ax.scatter([best], [losses.min()], color="tab:green", zorder=3)
    ax.axvline(true_coeff, color="tab:orange", ls="--", label=f"true {true_coeff}")
    ax.axvline(best, color="tab:green", ls=":", label=f"best {best:.3f}")
    ax.set_title("1D PDE coefficient QP objective (lower is better)")
    ax.set_xlabel(r"diffusion coefficient $a$")
    ax.set_ylabel("PISVR quadratic loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_landscape_1d.png"), dpi=180)
    plt.close(fig)
    print(f"1D best coefficient = {best:.6g}; true = {true_coeff}")


def make_2d_problem():
    diffusion = sp.Symbol("diffusion", real=True)
    true_diffusion = 2.0

    def operator(expr, variables, coefficients):
        return coefficients[0] * sum(
            sp.diff(expr, variable, 2) for variable in variables
        )

    def exact_solution(points):
        return np.sin(points[:, 0]) * np.cos(points[:, 1])

    def forcing(point):
        return -2 * true_diffusion * np.sin(point[0]) * np.cos(point[1])

    pde = PDE(
        operator,
        forcing,
        hyperparams={diffusion: 1.5},
    )

    rng = np.random.default_rng(0)
    X_data = rng.uniform(0.0, 2 * np.pi, size=(40, 2))
    y_data = exact_solution(X_data) + 0.1 * rng.standard_normal(len(X_data))
    phys_axis = np.linspace(0.0, 2 * np.pi, 10)
    X_phys = np.column_stack([axis.ravel() for axis in np.meshgrid(phys_axis, phys_axis)])

    model = PISVR(
        m=2,
        lam=0.1,
        eps1=0.1,
        C1=10.0,
        eps2=0.1,
        C2=100.0,
        pde=pde,
    )
    model.X_data = np.asarray(X_data)
    model.y_data = np.asarray(y_data)
    model.X_phys = np.asarray(X_phys)
    return model, diffusion, true_diffusion, exact_solution


def plot_2d_diffusion_landscape():
    model, diffusion, true_diffusion, _exact_solution = make_2d_problem()
    objective = make_pisvr_objective(model)

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

    diffusion_values = np.linspace(0.1, 10.0, 50)
    qp_losses = np.empty_like(diffusion_values)
    cv_mses = np.empty_like(diffusion_values)
    for index, value in enumerate(tqdm(diffusion_values)):
        params = [float(value)]
        qp_losses[index] = objective(params)
        cv_mses[index] = float(model.forward_pass(np.asarray(params, dtype=float)))

    best_qp_idx = int(np.argmin(qp_losses))
    best_diffusion = float(diffusion_values[best_qp_idx])
    best_cv_idx = int(np.argmin(cv_mses))
    best_cv_diffusion = float(diffusion_values[best_cv_idx])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    qp_line, = ax.plot(diffusion_values, qp_losses, color="tab:blue", lw=2,
                       label="PISVR quadratic objective")
    ax.scatter([best_diffusion], [qp_losses[best_qp_idx]], color="tab:blue",
               marker="o", zorder=3, label=f"QP min {best_diffusion:.3g}")
    ax.axvline(true_diffusion, color="tab:green", ls="--",
               label=f"true diffusion {true_diffusion:g}")
    ax.set_yscale("log")
    ax.set_ylabel("PISVR quadratic objective (log scale)")

    cv_ax = ax.twinx()
    cv_line, = cv_ax.plot(diffusion_values, cv_mses, color="tab:orange", lw=2,
                          ls="-.", label=f"{n_splits}-fold CV MSE")
    cv_ax.scatter([best_cv_diffusion], [cv_mses[best_cv_idx]], color="tab:orange",
                  marker="s", zorder=3,
                  label=f"CV min {best_cv_diffusion:.3g}")
    cv_ax.set_yscale("log")
    cv_ax.set_ylabel(f"{n_splits}-fold CV MSE (log scale)")

    ax.set_title("2D Poisson: loss versus diffusion coefficient")
    ax.set_xlabel("diffusion coefficient")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    cv_handles, cv_labels = cv_ax.get_legend_handles_labels()
    ax.legend(handles + cv_handles, labels + cv_labels)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_landscape_2d_diffusion.png"), dpi=180)
    plt.close(fig)

    solution_axis = np.linspace(0.0, 2 * np.pi, 60)
    X_grid, Y_grid = np.meshgrid(solution_axis, solution_axis)
    X_eval = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    exact_values = _exact_solution(X_eval)

    solution_diffusion = best_cv_diffusion
    model.pde_hyperparam_values = [solution_diffusion]
    model.fit(model.X_data, model.y_data, model.X_phys)
    optimal_values = model.predict(X_eval)

    model.pde_hyperparam_values = [true_diffusion]
    model.fit(model.X_data, model.y_data, model.X_phys)
    true_values = model.predict(X_eval)

    optimal_rmse = float(np.sqrt(np.mean((optimal_values - exact_values) ** 2)))
    true_rmse = float(np.sqrt(np.mean((true_values - exact_values) ** 2)))
    exact_grid = exact_values.reshape(X_grid.shape)
    optimal_grid = optimal_values.reshape(X_grid.shape)
    true_grid = true_values.reshape(X_grid.shape)
    shared_min = min(float(exact_grid.min()), float(optimal_grid.min()),
                     float(true_grid.min()))
    shared_max = max(float(exact_grid.max()), float(optimal_grid.max()),
                     float(true_grid.max()))
    error_true = np.abs(true_grid - exact_grid)
    error_optimal = np.abs(optimal_grid - exact_grid)
    error_max = max(float(error_true.max()), float(error_optimal.max()),
                    np.finfo(float).eps)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    field_levels = np.linspace(shared_min, shared_max, 25)
    field_panels = [
        ("Exact", exact_grid),
        (f"PISVR, true diffusion ({true_diffusion:g})", true_grid),
        (f"PISVR, CV optimum ({solution_diffusion:.3g})", optimal_grid),
    ]
    for axis, (title, values) in zip(axes[0], field_panels):
        image = axis.contourf(X_grid, Y_grid, values, levels=field_levels,
                              cmap="RdBu_r", extend="both")
        axis.scatter(model.X_data[:, 0], model.X_data[:, 1], c="black", s=10,
                     label="measurements")
        axis.scatter(model.X_phys[:, 0], model.X_phys[:, 1], marker="+",
                     c="tab:green", s=22, label="collocation points")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(image, ax=axes[0], shrink=0.85, label="u(x, y)")

    error_levels = np.linspace(0.0, error_max, 25)
    error_panels = [
        (f"True diffusion absolute error (RMSE={true_rmse:.3f})", error_true),
        (f"CV-optimal absolute error (RMSE={optimal_rmse:.3f})", error_optimal),
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
        f"2D Poisson, fixed lambda={model.lam:g}\n"
        f"true diffusion: {true_diffusion:g}\n"
        f"CV-optimal diffusion: {solution_diffusion:.4g}\n\n"
        f"RMSE at true diffusion: {true_rmse:.4f}\n"
        f"RMSE at CV optimum: {optimal_rmse:.4f}",
        fontsize=11, family="monospace", va="center",
    )
    fig.savefig(os.path.join(OUT_DIR, "solution_comparison_2d_diffusion.png"), dpi=150)
    plt.close(fig)

    print(
        f"2D best diffusion (QP) = {best_diffusion:.6g}; true diffusion = {true_diffusion} "
        f"(fixed lambda={model.lam:g}); {n_splits}-fold CV minimum="
        f"{best_cv_diffusion:.6g}"
    )
    print(f"2D solution RMSE: true diffusion={true_rmse:.6g}, "
          f"CV-optimal diffusion={optimal_rmse:.6g}")


def plot_2d_hyperparameter_landscapes():
    model, diffusion, true_diffusion, _exact_solution = make_2d_problem()
    n_splits = 5
    initial_params = np.asarray(
        [model.pde_hyperparam_values[0], model.lam], dtype=float
    )
    model.optimize_pde_hyperparams = True
    model.svr_hyperparams = ["lam"]
    model.lam_index = 1
    model.C1_index = None
    model.C2_index = None
    model.split = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=42).split(model.X_data)
    )
    model.layers = [
        model.define_cvxpylayer(len(train_idx), len(model.X_phys))
        for train_idx, _ in model.split
    ]

    diffusion_values = np.linspace(0.1, 5.0, 10)
    lambda_values = np.linspace(0.05, 0.5, 10)
    optimization_starts = [
        initial_params,
        np.asarray([1.0, 0.05]),
        np.asarray([1.0, 0.5]),
        np.asarray([4.5, 0.05]),
        np.asarray([4.5, 0.5]),
    ]
    parameter_bounds = [
        (float(diffusion_values[0]), float(diffusion_values[-1])),
        (float(lambda_values[0]), float(lambda_values[-1])),
    ]
    optimization_runs = []
    for start in optimization_starts:
        trajectory = [(start.copy(), float(model.forward_pass(start)))]

        def record_bfgs_step(params, path=trajectory):
            params = np.asarray(params, dtype=float).copy()
            loss = float(model.forward_pass(params))
            path.append((params, loss))
            print(
                f"Landscape run start={start.tolist()} | iter={len(path)-1} | "
                f"params={params.tolist()} | CV MSE={loss:.8g}"
            )

        result = minimize(
            model.forward_pass,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=parameter_bounds,
            jac=jax.grad(model.forward_pass, argnums=0),
            callback=record_bfgs_step,
            options={"maxls": 10},
        )
        if not np.isfinite(result.fun):
            raise RuntimeError("CV-based L-BFGS-B returned a non-finite objective.")
        path = np.asarray([params for params, _loss in trajectory])
        optimization_runs.append((np.asarray(result.x), float(result.fun), path))

    qp_objectives = np.empty((len(diffusion_values), len(lambda_values)))
    cv_mses = np.empty_like(qp_objectives)
    candidates = [
        (float(a), float(lam))
        for a in diffusion_values
        for lam in lambda_values
    ]
    for diffusion_value, lambda_value in tqdm(candidates):
        i = int(np.argmin(np.abs(diffusion_values - diffusion_value)))
        j = int(np.argmin(np.abs(lambda_values - lambda_value)))
        model.lam = lambda_value
        model.pde_hyperparam_values = [diffusion_value]
        model.pde_hyperparams = dict(
            zip(model.pde_hyperparam_sym, model.pde_hyperparam_values)
        )
        model.fit(model.X_data, model.y_data, model.X_phys)
        qp_objectives[i, j] = model.obj
        params = np.asarray([diffusion_value, lambda_value], dtype=float)
        cv_mses[i, j] = float(model.forward_pass(params))

    diffusion_grid, lambda_grid = np.meshgrid(
        diffusion_values, lambda_values, indexing="ij"
    )
    surfaces = [
        (qp_objectives, "PISVR quadratic objective", "loss_landscape_2d_qp.png", False),
        (cv_mses, f"{n_splits}-fold CV MSE", "loss_landscape_2d_cv_mse.png", True),
    ]
    for values, title, filename, show_trajectory in surfaces:
        minimum_index = np.unravel_index(np.argmin(values), values.shape)
        best_diffusion = float(diffusion_values[minimum_index[0]])
        best_lambda = float(lambda_values[minimum_index[1]])

        if np.all(values > 0.0):
            color_values = values
            colorbar_label = f"{title} (log scale)"
        else:
            # A standard logarithm cannot display signed QP objective values.
            # Log the excess above the minimum to show the landscape's relative rise.
            floor = max(float(np.ptp(values)) * 1e-8, np.finfo(float).eps)
            color_values = values - float(np.min(values)) + floor
            colorbar_label = f"log excess of {title} above its minimum"

        color_min = float(np.min(color_values))
        color_max = float(np.max(color_values))
        if np.isclose(color_min, color_max, rtol=1e-12, atol=0.0):
            norm_min, norm_max = color_min / 1.01, color_max * 1.01
        else:
            norm_min, norm_max = color_min, color_max
        contour_levels = np.geomspace(norm_min, norm_max, 30)
        fig, ax = plt.subplots(figsize=(8, 6))
        filled = ax.contourf(
            diffusion_grid,
            lambda_grid,
            color_values,
            levels=contour_levels,
            norm=LogNorm(vmin=norm_min, vmax=norm_max),
            cmap="viridis",
        )
        lines = ax.contour(
            diffusion_grid,
            lambda_grid,
            color_values,
            levels=10,
            colors="white",
            linewidths=0.6,
            alpha=0.7,
        )
        ax.clabel(lines, inline=True, fontsize=8, fmt="%.3g")
        ax.scatter(best_diffusion, best_lambda, color="red", marker="x", s=80,
                   linewidths=2, label=f"grid minimum ({best_diffusion:.3g}, {best_lambda:.3g})")
        if show_trajectory:
            colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(optimization_runs)))
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
                    label=f"run {run_index} (CV MSE={final_loss:.3g})",
                    zorder=5,
                )
                ax.scatter(*path[0], color=color, edgecolor="black", marker="s",
                           s=48, zorder=6)
                ax.scatter(*path[-1], color=color, edgecolor="black", marker="*",
                           s=90, zorder=6)
        ax.axvline(true_diffusion, color="white", linestyle="--", linewidth=1.2,
                   label=f"true diffusion {true_diffusion:g}")
        ax.set_title(f"2D Poisson: {title}")
        ax.set_xlabel("diffusion coefficient")
        ax.set_ylabel(r"RBF parameter $\lambda$")
        fig.colorbar(filled, ax=ax, label=colorbar_label)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=180)
        plt.close(fig)
        print(
            f"{title} minimum: diffusion={best_diffusion:.6g}, "
            f"lambda={best_lambda:.6g}"
        )
    for run_index, (final_params, final_loss, _path) in enumerate(optimization_runs, start=1):
        print(
            f"CV L-BFGS-B run {run_index}: diffusion={final_params[0]:.6g}, "
            f"lambda={final_params[1]:.6g}, CV MSE={final_loss:.6g}"
        )


if __name__ == "__main__":
    # plot_1d_landscape()
    plot_2d_hyperparameter_landscapes()
    plot_2d_diffusion_landscape()
    print(f"Saved plots in {OUT_DIR}")
