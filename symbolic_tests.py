import sympy as sp

lambda_ = 1.0
n = 3

# Symbolic vector 1: [x1_1, x1_2, x1_3]
x1s = sp.symbols(f"x1_1:{n+1}", real=True)
x1 = sp.Matrix(x1s)

# Symbolic vector 2: [x2_1, x2_2, x2_3]
x2s = sp.symbols(f"x2_1:{n+1}", real=True)
x2 = sp.Matrix(x2s)

rbf = sp.exp(- lambda_ * (x1-x2).dot(x1-x2))

def custom_linear_operator(expr, x):
    return sp.diff(expr, x[0]) + 2 * sp.diff(expr, x[1]) - sp.diff(expr, x[2])

def diff_operator(expr, x1, x2):
    return sp.diff(sp.diff(expr, x1), x2)

print(custom_linear_operator(custom_linear_operator(rbf, x1), x2))