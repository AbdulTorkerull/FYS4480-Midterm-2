import numpy as np

# --------------------------------------------------
# 1. Hamiltonian matrix
# --------------------------------------------------

def H_exact(g):
    return np.array([
        [2-g,   -g/2,   -g/2,   -g/2,   -g/2,    0],
        [-g/2,   4-g,   -g/2,   -g/2,    0,     -g/2],
        [-g/2,  -g/2,    6-g,    0,     -g/2,   -g/2],
        [-g/2,  -g/2,     0,     6-g,   -g/2,   -g/2],
        [-g/2,   0,     -g/2,   -g/2,    8-g,   -g/2],
        [0,     -g/2,   -g/2,   -g/2,   -g/2,    10-g]
    ], dtype=float)

def H0_matrix():
    return np.diag([2, 4, 6, 6, 8, 10])

def V_matrix(g):
    return H_exact(g) - H0_matrix()



# --------------------------------------------------
# 2. Vector matrix multiplciation (recursion relation)
# --------------------------------------------------

def E2_E3_vector_matrix(g):
    H0 = H0_matrix()
    V = H_exact(g) - H0
    w = V[1:, 0]
    W = V[1:, 1:] - V[0, 0] * np.eye(5)
    R = np.diag(1.0 / (H0[0, 0] - np.diag(H0)[1:]))

    E2 = w @ R @ w
    E3 = w @ R @ W @ R @ w

    return E2, E3



# --------------------------------------------------
# 3. Diagram sum
# --------------------------------------------------

def E2_E3_diagrams_sum(g, hf=False):

    eps_h = np.array([0.0, 1.0])
    eps_p = np.array([2.0, 3.0])

    if hf:
        eps_h = eps_h - g/2

    V_hhpp = V_pppp = V_hhhh = -g/2 * np.ones((2, 2))
    T = -g/2 / (2 * (eps_h[None, :] - eps_p[:, None]))

    D1 = np.trace(V_hhpp @ T)
    D4 = np.trace(T.T @ V_pppp @ T)
    D5 = np.trace(T @ V_hhhh @ T.T)
    D8 = 0.0

    if not hf:
        D8 = g**3/16 * np.sum(1.0 / (eps_h[None, :] - eps_p[:, None])**2)

    E2 = D1
    E3 = D4 + D5 + D8

    return E2, E3

if __name__ == "__main__":
    g = -0.5
    E2_an, E3_an = -7*g**2/24, -g**3/12
    print(f"\ng = {g}:")

    print("Analytical solution     ", (E2_an, E3_an))
    print("recursion solution ", E2_E3_vector_matrix(g))
    print("diagrams sum solution   ", E2_E3_diagrams_sum(g))