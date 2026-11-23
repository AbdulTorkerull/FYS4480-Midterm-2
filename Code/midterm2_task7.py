import numpy as np
import matplotlib.pyplot as plt

from dmrg import pairing_terms, dmrg_ground_state         
from midterm2_task6iii import (E2, E2_hf, E_exact, E_ref, E_second_order, E_second_order_hf)

# --------------------------------------------------
# 1. DMRG energy: 4 levels, 4 particles, bond dimension chi
# --------------------------------------------------

def E_dmrg(g, chi):
    return dmrg_ground_state(pairing_terms(4, g), 4, 4, chi_max=chi).energy



# --------------------------------------------------
# 2. Plot: absolute deviation from the exact energy
# --------------------------------------------------

def E_third_order(g):
    return E_ref(g) + E2(g) - g**3/12                   

def E3_hf(g):
    eps_h = np.array([0.0, 1.0]) - g/2                    
    eps_p = np.array([2.0, 3.0])                          

    V = -g/2 * np.ones((2, 2))
    T = -g/2 / (2 * (eps_h[None, :] - eps_p[:, None]))

    return np.trace(T.T @ V @ T) + np.trace(T @ V @ T.T)  

def E_third_order_hf(g):
    return E_ref(g) + E2_hf(g) + E3_hf(g)

if __name__ == "__main__":
    g = np.linspace(-1, 1, 81)
    g = g[g != 0]                                          

    exact = np.array([E_exact(x) for x in g])
    dmrg1 = np.array([E_dmrg(x, 1) for x in g])
    dmrg2 = np.array([E_dmrg(x, 2) for x in g])
    third_hf = np.array([E_third_order_hf(x) for x in g])

    plt.figure(figsize=(8, 5))
    plt.plot(g, exact, "k", lw=2.5, label="Exact (FCI)")
    plt.plot(g, E_ref(g), "-", lw=2, label=r"HF = first order ($2-g$)")
    plt.plot(g, dmrg1, ":", lw=2, label=r"DMRG, $\chi=1$")
    plt.plot(g, E_second_order(g), "-", lw=2, label="normal, 2nd order")
    plt.plot(g, E_second_order_hf(g), "--", lw=2, label="HF, 2nd order")
    plt.plot(g, E_third_order(g), "-", lw=2, label="normal, 3rd order")
    plt.plot(g, third_hf, "--", lw=2, label="HF, 3rd order")
    plt.plot(g, dmrg2, "-.", lw=2, label=r"DMRG, $\chi=2$")
    plt.xlabel(r"$g$")
    plt.ylabel("Ground-state energy")
    plt.title("Ground-state energy against $g$")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.semilogy(g, np.abs(E_ref(g) - exact), "-", lw=2, label=r"HF = first order ($2-g$)")
    plt.semilogy(g, np.abs(dmrg1 - exact), ":", lw=2, label=r"DMRG, $\chi=1$")
    plt.semilogy(g, np.abs(E_second_order(g) - exact), "-", lw=2, label="normal, 2nd order")
    plt.semilogy(g, np.abs(E_second_order_hf(g) - exact), "--", lw=2, label="HF, 2nd order")
    plt.semilogy(g, np.abs(E_third_order(g) - exact), "-", lw=2, label="normal, 3rd order")
    plt.semilogy(g, np.abs(third_hf - exact), "--", lw=2, label="HF, 3rd order")
    plt.semilogy(g, np.abs(dmrg2 - exact), "-.", lw=2, label=r"DMRG, $\chi=2$")
    plt.xlabel(r"$g$")
    plt.ylabel(r"$|E-E_{FCI}|$")
    plt.title("Absolute deviation from the exact energy")
    plt.grid(alpha=.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()