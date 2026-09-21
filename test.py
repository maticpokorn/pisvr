import numpy as np
import matplotlib.pyplot as plt
from svr import SVR

rng = np.random.default_rng(0)

# Fit on a noisy sine wave and check predictions are within tolerance
X_train = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
y_train = np.sin(X_train.ravel()) + rng.normal(0, 0.1, 30)

model = SVR(lambda_=1.0, eps=0.2, C=10.0)
model.fit(X_train, y_train)

X_test = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
y_pred = model.predict(X_train, X_test)
y_true = np.sin(X_test.ravel())

rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
print(f"RMSE: {rmse:.4f}")
assert rmse < 0.3, f"RMSE too high: {rmse:.4f}"
print("Test passed.")

X_dense = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_dense = model.predict(X_train, X_dense)

plt.figure(figsize=(8, 4))
plt.scatter(X_train.ravel(), y_train, s=20, color="steelblue", label="Training data")
plt.plot(X_dense.ravel(), np.sin(X_dense.ravel()), color="gray", linestyle="--", label="True sine")
plt.plot(X_dense.ravel(), y_dense, color="tomato", label="SVR prediction")
plt.legend()
plt.tight_layout()
plt.show()
