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
# 2. Energies at order up to third
# --------------------------------------------------

def E_ref(g):  return 2 - g
def E2(g):     return (-7*g**2)/(24)
def E3(g):     return (-g**3)/(12)

def E_first_order(g):
    return E_ref(g)

def E_second_order(g):
    return E_ref(g) + E2(g)

def E_third_order(g):
    return E_ref(g) + E2(g) + E3(g)



# --------------------------------------------------
# 3. Plot: energies as functions of g
# --------------------------------------------------

g = np.linspace(-1, 1, 201)
exact = np.array([E_exact(x) for x in g])

plt.figure(figsize=(8, 5))
plt.plot(g, exact, "k", lw=2.2, label="Exact")
plt.plot(g, E_first_order(g), ":", lw=2, label=r"$E_{ref}$")
plt.plot(g, E_second_order(g), "--", lw=2, label=r"$E_{ref}+E^{(2)}$")
plt.plot(g, E_third_order(g), "-.", lw=2, label=r"$E_{ref}+E^{(2)}+E^{(3)}$")
plt.xlabel(r"$g$")
plt.ylabel("Ground-state energy")
plt.title("Ground-state energy versus $g$")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.show()



# --------------------------------------------------
# 4. Plot: absolute deviation from the exact energy
# --------------------------------------------------

g = np.linspace(-1, 1, 401)
g = g[g != 0]                                  
exact = np.array([E_exact(x) for x in g])

plt.figure(figsize=(8, 5))
plt.semilogy(g, np.abs(E_first_order(g) - exact), ":", lw=2, label=r"$E_{ref}$")
plt.semilogy(g, np.abs(E_second_order(g) - exact), "--", lw=2, label=r"$E_{ref}+E^{(2)}$")
plt.semilogy(g, np.abs(E_third_order(g) - exact), "-.", lw=2, label=r"$E_{ref}+E^{(2)}+E^{(3)}$")
plt.axhline(1e-3, color="gray", lw=1, label=r"$10^{-3}$")
plt.xlabel(r"$g$")
plt.ylabel(r"$|E-E_{FCI}|$")
plt.title("Absolute deviation from the exact energy")
plt.grid(alpha=.3, which="both")
plt.legend()
plt.tight_layout()
plt.show()