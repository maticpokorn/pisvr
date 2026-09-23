import numpy as np
import sympy as sp
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pisvr import PISVR

rng = np.random.default_rng(0)
rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))

# Usage: model.fit(X_data, y_data, X_phys)
#   X_data / y_data : the (few, possibly noisy) measurements
#   X_phys          : collocation points where the operator residual L[u] - f is enforced
# The plain-SVR baseline uses C2 = 0, which forces every beta to 0, so the
# collocation points cannot matter; a single dummy point avoids building large,
# unused physics blocks.


# ======================================================================
# Example 1 (1-D):   L[u] = u',   f(x) = cos x
#                    exact solution  u(x) = sin x   (data + physics agree)
# L[1] = 0 here, so the physics term does not enter the equality constraint.
# ======================================================================
L1d = lambda e, v: sp.diff(e, v[0])
f1d = lambda x: np.cos(x[0])
u1d = lambda x: np.sin(x)

n, N = 10, 10
X = np.sort(rng.uniform(0, 2 * np.pi, n))[:, None]
y = u1d(X[:, 0]) + 0.05 * rng.standard_normal(n)
Xp = np.linspace(0, 2 * np.pi, N)[:, None]                 # collocation points
Xt = np.linspace(0, 2 * np.pi, 400)[:, None]

pi1 = PISVR(1, 0.1, 0.1, 10, 0.1, 100, L1d, f1d).fit_cvxpylayer_bfgs(X, y, Xp, n_splits=3)
base = PISVR(1, 0.1, 0.1, 10, 0.0, 0.0, L1d, f1d).fit_cvxpylayer_bfgs(X, y, Xp[:1], n_splits=3)   # C2=0 -> plain SVR

print(f"[1-D]  RMSE")
print(f"  plain SVR: {rmse(base.predict(Xt), u1d(Xt[:, 0])):.4f}")
print(f"  PISVR BFGS: {rmse(pi1.predict(Xt), u1d(Xt[:, 0])):.4f}")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(Xt[:, 0], u1d(Xt[:, 0]), "k--", label="exact  sin(x)")
ax.plot(Xt[:, 0], base.predict(Xt), color="tab:orange", label="plain SVR (C2=0)")
ax.plot(Xt[:, 0], pi1.predict(Xt), color="tab:blue", lw=2, label="PISVR")
ax.scatter(X[:, 0], y, c="k", s=25, zorder=3, label="data")
ax.scatter(Xp[:, 0], np.full(N, ax.get_ylim()[0]), marker="|", c="tab:green", s=60,
           label="collocation points")
ax.set(title="1-D:  u' = cos x", xlabel="x", ylabel="u")
ax.legend()
fig.tight_layout()
fig.savefig("pisvr_1d.png", dpi=130)


# ======================================================================
# Example 2 (2-D Poisson):  L[u] = Laplacian(u),  f = -2 sin(x1) cos(x2)
#                           exact solution  u = sin(x1) cos(x2)
# ======================================================================
lap = lambda e, v: sum(sp.diff(e, vi, 2) for vi in v)
f2d = lambda x: -2 * np.sin(x[0]) * np.cos(x[1])
u2d = lambda X: np.sin(X[:, 0]) * np.cos(X[:, 1])

n, N = 5, 5
X2 = rng.uniform(0, 2 * np.pi, (n, 2))
y2 = u2d(X2) + 0.1 * rng.standard_normal(n)

gp = np.linspace(0, 2 * np.pi, N)                          # 10 x 10 collocation grid
Xp2 = np.column_stack([a.ravel() for a in np.meshgrid(gp, gp)])

g = np.linspace(0, 2 * np.pi, 60)
G1, G2 = np.meshgrid(g, g)
Xg = np.column_stack([G1.ravel(), G2.ravel()])
U = u2d(Xg)

pi21 = PISVR(2, 0.1, 0.1, 10, 0.1, 100, lap, f2d).fit(X2, y2, Xp2)
pi22 = PISVR(2, 0.1, 0.1, 10, 0.1, 100, lap, f2d).fit_cvxpylayer_bfgs(X2, y2, Xp2, n_splits=5)
base21 = PISVR(2, 0.1, 0.1, 10, 0.0, 0.0, lap, f2d).fit(X2, y2, Xp2[:1])
base22 = PISVR(2, 0.1, 0.1, 10, 0.0, 0.0, lap, f2d).fit_cvxpylayer_bfgs(X2, y2, Xp2[:1], n_splits=5)
Up, Ub = pi22.predict(Xg), base22.predict(Xg)

print(f"[2-D]  RMSE")
print(f"PISVR: {rmse(pi21.predict(Xg), U)}")
print(f"PISVR (HPO): {rmse(pi22.predict(Xg), U)}")
print(f"SVR: {rmse(base21.predict(Xg), U)}")
print(f"SVR (HPO): {rmse(base22.predict(Xg), U)}")

fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
lv = np.linspace(-1.1, 1.1, 23)
for a, (t, Z) in zip(axs[0], [("exact  sin(x1) cos(x2)", U), ("PISVR", Up), ("plain SVR (C2=0)", Ub)]):
    c = a.contourf(G1, G2, Z.reshape(G1.shape), levels=lv, cmap="RdBu_r", extend="both")  # clip wild extrapolation
    a.scatter(X2[:, 0], X2[:, 1], c="k", s=10)
    a.set_title(t)
axs[0][1].scatter(Xp2[:, 0], Xp2[:, 1], marker="+", c="tab:green", s=22)   # collocation grid
fig.colorbar(c, ax=axs[0], shrink=0.85, label="u")

le = np.linspace(0, 0.6, 25)
for a, (t, Z) in zip(axs[1][:2], [("|PISVR - exact|", Up), ("|plain SVR - exact|", Ub)]):
    ce = a.contourf(G1, G2, np.abs(Z - U).reshape(G1.shape), levels=le, cmap="viridis", extend="max")
    a.scatter(X2[:, 0], X2[:, 1], c="w", s=8, edgecolors="k", linewidths=0.4)
    a.set_title(t)
fig.colorbar(ce, ax=axs[1][:2], shrink=0.85, label="abs. error (clipped at 0.6)")
axs[1][2].axis("off")
axs[1][2].text(0.0, 0.6, f"2-D Poisson,  n = {n} data points,  N = {len(Xp2)} collocation pts\n\n"
                         f"RMSE plain SVR: {rmse(Ub, U):.3f}\nRMSE PISVR:      {rmse(Up, U):.3f}",
               fontsize=11, family="monospace", va="center")
fig.savefig("pisvr_2d.png", dpi=130)


# ======================================================================
# Example 3 (time-dependent, 2 space dims + time -> m = 3):
#   forced heat equation   u_t - alpha (u_xx + u_yy) = f(x, y, t)
#   f = sin x sin y (2 alpha cos t - sin t)
#   exact solution         u = sin x sin y cos t     on [0, pi]^2 x [0, pi]
# L[1] = 0, but f depends on time, so the physics term carries real information.
# Data live only in t in [0, T/4]; collocation points cover the WHOLE space-time
# domain, so the physics is the only source of information for t > T/4.
# ======================================================================
alpha, T = 0.3, np.pi
Lheat = lambda e, v: sp.diff(e, v[2]) - alpha * (sp.diff(e, v[0], 2) + sp.diff(e, v[1], 2))
fheat = lambda p: np.sin(p[0]) * np.sin(p[1]) * (2 * alpha * np.cos(p[2]) - np.sin(p[2]))
uheat = lambda P: np.sin(P[:, 0]) * np.sin(P[:, 1]) * np.cos(P[:, 2])

def heat_data(seed, n=40):                       # scattered noisy space-time samples
    r = np.random.default_rng(100 + seed)
    X_xy = r.uniform(0, T, (n, 2))
    X_t = r.uniform(0, T / 4, (n, 1))           # <-- NO samples after T/4
    X = np.hstack([X_xy, X_t])
    return X, uheat(X) + 0.1 * r.standard_normal(n)

N_phys = 150                                     # collocation points over the full space-time box
Xp3 = np.random.default_rng(7).uniform(0, T, (N_phys, 3))   # fixed across data sets

heat_args = (3, 0.1, 0.1, 100, 0.001)            # m, lambda_, eps1, C1, eps2   (C2 added below)
Pt = np.random.default_rng(99).uniform(0, T, (4000, 3))   # random space-time test points

# A single random draw can go either way, so first compare over several data sets.
n_seeds = 12
res = []
for s in tqdm(range(n_seeds)):
    Xs, ys = heat_data(s)
    e_pi = rmse(PISVR(*heat_args, 100, Lheat, fheat).fit(Xs, ys, Xp3).predict(Pt), uheat(Pt))
    e_base = rmse(PISVR(*heat_args, 0.0, Lheat, fheat).fit(Xs, ys, Xp3[:1]).predict(Pt), uheat(Pt))
    res.append((e_pi, e_base))
res = np.array(res)
print(f"[heat] over {n_seeds} data sets:  PISVR {res[:, 0].mean():.3f} +/- {res[:, 0].std():.3f}   "
      f"plain SVR {res[:, 1].mean():.3f} +/- {res[:, 1].std():.3f}   "
      f"PISVR better in {(res[:, 0] < res[:, 1]).sum()}/{n_seeds}")

# The figures below use the FIRST data set of that study (seed 0), not one chosen for its outcome.
X3, y3 = heat_data(0)
pi3 = PISVR(*heat_args, 100, Lheat, fheat).fit(X3, y3, Xp3)
base3 = PISVR(*heat_args, 0.0, Lheat, fheat).fit(X3, y3, Xp3[:1])
print(f"[heat] plotted data set (seed 0):  plain SVR {res[0, 1]:.3f}   PISVR {res[0, 0]:.3f}")

# evaluation helpers
gx = np.linspace(0, T, 40)
GX, GY = np.meshgrid(gx, gx)
def slice_at(t):
    return np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, t)])

# --- Figure A: spatial snapshots at several times --------------------------------
times = [0.0, T / 3, 2 * T / 3, T]
fig, axs = plt.subplots(3, len(times), figsize=(15, 10.5), constrained_layout=True)
rows = [("exact", None), ("PISVR", pi3), ("plain SVR (C2=0)", base3)]
for j, t in enumerate(times):
    P = slice_at(t)
    near = np.abs(X3[:, 2] - t) < 0.35                      # data samples close to this time
    for i, (name, model) in enumerate(rows):
        Z = uheat(P) if model is None else model.predict(P)
        a = axs[i, j]
        c = a.contourf(GX, GY, Z.reshape(GX.shape), levels=lv, cmap="RdBu_r", extend="both")
        a.scatter(X3[near, 0], X3[near, 1], c="k", s=14)
        a.set_aspect("equal")
        a.set_title(f"{name},  t = {t:.2f}")
        if j == 0:
            a.set_ylabel("y")
        if i == 2:
            a.set_xlabel("x")
fig.colorbar(c, ax=axs, shrink=0.6, label="u")
fig.suptitle("Forced 2-D heat equation:  snapshots  (dots = data with |t_i - t| < 0.35)")
fig.savefig("pisvr_heat_snapshots.png", dpi=120)

# --- Figure B: error vs. time, and a time trace at the domain centre --------------
tt = np.linspace(0, T, 25)
err_pi, err_base = [], []
for t in tt:
    P = slice_at(t)
    err_pi.append(rmse(pi3.predict(P), uheat(P)))
    err_base.append(rmse(base3.predict(P), uheat(P)))

tc = np.linspace(0, T, 200)
Pc = np.column_stack([np.full(tc.size, T / 2), np.full(tc.size, T / 2), tc])

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
axs[0].plot(tt, err_base, "o-", color="tab:orange", label="plain SVR (C2=0)")
axs[0].plot(tt, err_pi, "o-", color="tab:blue", label="PISVR")
axs[0].set(title="Spatial RMSE at each time", xlabel="t", ylabel="RMSE over (x, y)")
axs[0].legend()
axs[1].plot(tc, uheat(Pc), "k--", label="exact")
axs[1].plot(tc, base3.predict(Pc), color="tab:orange", label="plain SVR (C2=0)")
axs[1].plot(tc, pi3.predict(Pc), color="tab:blue", lw=2, label="PISVR")
axs[1].set(title="Time trace at the centre  (x = y = pi/2)", xlabel="t", ylabel="u")
axs[1].legend()
fig.savefig("pisvr_heat_error.png", dpi=130)