import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Hamiltonian matrix and exact energy
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

def E_exact(g):
    return np.linalg.eigvalsh(H_exact(g))[0]



# --------------------------------------------------
# 2. Energy through second order, normal and Hartree-Fock
# --------------------------------------------------

def E_ref(g):
    return 2 - g                                        

def E2(g):
    return (-7*g**2)/(24)

def E2_hf(g):
    return -(g**2/4) * (1/(2+g) + 2/(4+g) + 1/(6+g))      

def E_second_order(g):
    return E_ref(g) + E2(g)

def E_second_order_hf(g):
    return E_ref(g) + E2_hf(g)



# --------------------------------------------------
# 3. Plot: absolute deviation, normal against Hartree-Fock
# --------------------------------------------------

if __name__ == "__main__":
    g = np.linspace(-1, 1, 401)
    g = g[g != 0]                                         

    exact = np.array([E_exact(x) for x in g])

    plt.figure(figsize=(8, 5))
    plt.semilogy(g, np.abs(E_second_order(g) - exact), "-", lw=2, label="normal, 2nd order")
    plt.semilogy(g, np.abs(E_second_order_hf(g) - exact), "--", lw=2, label="HF, 2nd order")
    plt.axhline(1e-3, color="gray", lw=1, label=r"$10^{-3}$")
    plt.xlabel(r"$g$")
    plt.ylabel(r"$|E-E_{FCI}|$")
    plt.title("Absolute deviation through second order, normal against Hartree-Fock")
    plt.grid(alpha=.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()