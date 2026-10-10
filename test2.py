import warnings

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None


def tensor_to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        try:
            return np.asarray(x.detach().cpu().tolist())
        except Exception:
            return None

from pisvr import PDE, PISVR

# 1. Define the linear differential operator for your PISVR class
# Equation: u'(x) + 2u(x) = 0
def L_op(expr, x):
    return sp.diff(expr, x[0]) + 2 * expr

# 2. Define the right-hand side of the equation
def f_rhs(x):
    return 0.0

pde = PDE(L_op, f_rhs)

# 3. Generate the extreme sparse data (5 points)
X_data = np.linspace(0, 2, 10).reshape(-1, 1)
y_data = np.exp(-2 * X_data).flatten()

# Inject a gross outlier at x=1.0 (sensor failure)
y_data[5] += 0.75

# 4. Generate Physics Collocation Points
X_phys = np.linspace(0, 2, 40).reshape(-1, 1)


def train_pinn(X_data, y_data, X_phys, epochs=2000, lr=1e-3):
    if torch is None or nn is None or optim is None:
        warnings.warn("PyTorch is unavailable; skipping PINN.")
        return None

    try:
        class ODEPINN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.net(x)

        # Construct from lists to avoid Torch's NumPy bridge when incompatible
        # NumPy/Torch binaries are installed in the active environment.
        X_data_t = torch.tensor(X_data.tolist(), dtype=torch.float32)
        y_data_t = torch.tensor(y_data.reshape(-1, 1).tolist(), dtype=torch.float32)
        X_phys_t = torch.tensor(X_phys.tolist(), dtype=torch.float32, requires_grad=True)

        model = ODEPINN()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            y_hat = model(X_data_t)
            data_loss = torch.mean((y_hat - y_data_t) ** 2)

            u_phys = model(X_phys_t)
            du_phys = torch.autograd.grad(u_phys.sum(), X_phys_t, create_graph=True)[0]
            phys_loss = torch.mean((du_phys + 2.0 * u_phys) ** 2)

            loss = data_loss + 0.1 * phys_loss
            loss.backward()
            optimizer.step()

        return model
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"PyTorch/PINN could not be used in this environment: {exc}. Skipping PINN.")
        return None


# 5. Instantiate PISVR
# The hyperparameters C1 and eps1 are the keys to outlier resistance.
# C1 = 1.0 caps the influence of the outlier.
pisvr = PISVR(
    m=1,
    lam=2.0,      # RBF width
    eps1=0.01,    # Epsilon tube for data (ignores small noise)
    C1=10.0,       # Box constraint bounding outlier influence
    eps2=0.0,     # Epsilon tube for physics compliance
    C2=10.0,      # High penalty for violating the ODE
    pde=pde,
)

# 6. Fit the model
print("Fitting PISVR...")
pisvr.fit(X_data, y_data, X_phys)

# 7. Fit a plain SVR baseline for comparison (same data, no physics penalty)
base = PISVR(
    m=1,
    lam=5.0,
    eps1=0.1,
    C1=10.0,
    eps2=0.0,
    C2=0.0,
    pde=pde,
)
base.fit(X_data, y_data, X_phys[:1])

# 8. Fit the PINN baseline for comparison
print("Fitting PINN...")
pinn = train_pinn(X_data, y_data, X_phys, epochs=2000, lr=1e-3)

# 9. Predict and Evaluate
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_true = np.exp(-2 * X_test).flatten()
y_pred = pisvr.predict(X_test)
y_base = base.predict(X_test)
if pinn is not None:
    try:
        y_pinn = tensor_to_numpy(pinn(torch.tensor(X_test.tolist(), dtype=torch.float32)))
        if y_pinn is not None:
            y_pinn = y_pinn.reshape(-1)
        else:
            y_pinn = None
    except Exception as exc:
        warnings.warn(f"PINN prediction could not be converted to NumPy in this environment: {exc}. Skipping PINN plot.")
        y_pinn = None
else:
    y_pinn = None

rmse_pisvr = np.sqrt(np.mean((y_pred - y_true) ** 2))
rmse_base = np.sqrt(np.mean((y_base - y_true) ** 2))
rmse_pinn = np.sqrt(np.mean((y_pinn - y_true) ** 2)) if y_pinn is not None else np.nan

print(f"PISVR RMSE: {rmse_pisvr:.6f}")
print(f"Plain SVR RMSE: {rmse_base:.6f}")
if y_pinn is not None:
    print(f"PINN RMSE: {rmse_pinn:.6f}")

# 10. Plot the results to visualize the outlier rejection
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'k--', label="Exact Solution", linewidth=2)
plt.plot(X_test, y_pred, 'b-', label=f"PISVR Prediction (RMSE={rmse_pisvr:.4f})", linewidth=2)
plt.plot(X_test, y_base, 'g--', label=f"Plain SVR (RMSE={rmse_base:.4f})", linewidth=2)
if y_pinn is not None:
    plt.plot(X_test, y_pinn, color='tab:purple', linestyle=':',
             label=f"PINN output (RMSE={rmse_pinn:.4f})", linewidth=2)
plt.scatter(X_data, y_data, c='red', s=100, label=f"Sparse Data (with Outlier at x={X_data[5][0]:.1f})", zorder=5)
plt.title("PISVR and PINN Results on ODE", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("u(x)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()