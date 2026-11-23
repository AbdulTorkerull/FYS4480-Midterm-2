import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Hamiltonian matrices (your code, unchanged)
# --------------------------------------------------

def H_exact(g):
    return np.array([
        [2-g,   -g/2,     -g/2,     -g/2,     -g/2,      0],
        [-g/2,     4-g,   -g/2,     -g/2,      0,       -g/2],
        [-g/2,     -g/2,     6-g,    0,       -g/2,     -g/2],
        [-g/2,     -g/2,      0,       6-g,   -g/2,     -g/2],
        [-g/2,      0,       -g/2,     -g/2,     8-g,   -g/2],
        [0,        -g/2,     -g/2,     -g/2,     -g/2,    10-g]
    ], dtype=float)

def H_approx(g):
    return H_exact(g)[:5, :5]

def diagonalize(H):
    return np.linalg.eigh(H)



# --------------------------------------------------
# 2. Ground states, intermediate normalization
# --------------------------------------------------

def intermediate(vec):
    return vec / vec[0]

def t_first(g):
    return np.array([g/4, g/8, g/8, g/12])

def E_ref(g):  return 2 - g
def E2(g):     return -7*g**2/24
def E3(g):     return -g**3/12

def E_var(g):
    H = H_exact(g)
    psi = np.zeros(6)
    psi[0] = 1.0
    psi[1:5] = t_first(g)
    return psi @ H @ psi / (psi @ psi)

def norm_psi1(g):
    return np.sum(t_first(g)**2)



# --------------------------------------------------
# 3. Collect results
# --------------------------------------------------

def calculate_results(g_values):
    out = {k: [] for k in ["E_exact","E_ci","E_var","E_pt3","E_formula",
                           "c_exact","c_ci","t1","t_0011_exact","t_0011_2nd"]}
    for g in g_values:
        e, v = diagonalize(H_exact(g))
        out["E_exact"].append(e[0])
        out["c_exact"].append(intermediate(v[:, 0]))
        ea, va = diagonalize(H_approx(g))
        out["E_ci"].append(ea[0])
        out["c_ci"].append(intermediate(va[:, 0]))
        out["E_var"].append(E_var(g))
        out["E_pt3"].append(E_ref(g) + E2(g) + E3(g))
        out["E_formula"].append(E_ref(g)
                                + (E2(g) + E3(g))/(1 + norm_psi1(g)))
        out["t1"].append(t_first(g))
        t = t_first(g)
        out["t_0011_2nd"].append(t[2]*t[1] + t[3]*t[0])
        out["t_0011_exact"].append(out["c_exact"][-1][5])
    res = {k: np.array(v) for k, v in out.items()}
    res["g"] = np.array(g_values)
    return res



# --------------------------------------------------
# 4. Plots
# --------------------------------------------------

def plot_energies(r):
    g = r["g"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(g, r["E_exact"], "k", lw=2.2, label="Exact (6x6)")
    ax[0].plot(g, r["E_ci"], "--", lw=2, label="CI, at most 2p2h (5x5)")
    ax[0].plot(g, r["E_var"], "-.", lw=2, label=r"$\langle H\rangle$ in $|\Phi_0\rangle+\hat\Omega^{(1)}|\Phi_0\rangle$")
    ax[0].plot(g, r["E_pt3"], ":", lw=2.2, label=r"$E_{ref}+E^{(2)}+E^{(3)}$")
    ax[0].set_xlabel(r"$g$"); ax[0].set_ylabel("Ground-state energy")
    ax[0].set_title("Ground-state energy"); ax[0].grid(alpha=.3); ax[0].legend()

    for k, lab in [("E_ci", "CI 2p2h"), ("E_var", r"$\langle H\rangle$ with $\Omega^{(1)}$"),
                   ("E_pt3", r"$E_{ref}+E^{(2)}+E^{(3)}$")]:
        ax[1].plot(g, r[k] - r["E_exact"], label=lab, lw=2)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel(r"$g$"); ax[1].set_ylabel(r"$E-E_{exact}$")
    ax[1].set_title("Difference from exact energy"); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig("plots/energies.png", dpi=200); plt.close()

def plot_amplitudes(r):
    g = r["g"]
    names = [r"$|13\rangle$  ($t_2^3$)", r"$|14\rangle$  ($t_2^4$)",
             r"$|23\rangle$  ($t_1^3$)", r"$|24\rangle$  ($t_1^4$)"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for m, a in enumerate(ax.flat):
        a.plot(g, r["c_exact"][:, m+1], "k", lw=2.2, label="Exact")
        a.plot(g, r["c_ci"][:, m+1], "--", lw=2, label="CI 2p2h")
        a.plot(g, r["t1"][:, m], ":", lw=2.4, label=r"$\hat\Omega^{(1)}$")
        a.set_title(names[m]); a.grid(alpha=.3)
    ax[1, 0].set_xlabel(r"$g$"); ax[1, 1].set_xlabel(r"$g$")
    ax[0, 0].set_ylabel("amplitude"); ax[1, 0].set_ylabel("amplitude")
    ax[0, 0].legend()
    plt.suptitle("Amplitudes in intermediate normalization")
    plt.tight_layout(); plt.savefig("plots/amplitudes.png", dpi=200); plt.close()

def plot_0011(r):
    g = r["g"]
    plt.figure(figsize=(8, 5))
    plt.plot(g, r["t_0011_exact"], "k", lw=2.2, label="Exact")
    plt.plot(g, r["t_0011_2nd"], ":", lw=2.4,
             label=r"$\hat\Omega^{(2)}$: $t_1^3t_2^4+t_1^4t_2^3=\frac{7}{192}g^2$")
    plt.xlabel(r"$g$"); plt.ylabel(r"amplitude of $|0011\rangle$")
    plt.title(r"Amplitude of $|0011\rangle=|\Phi_{12}^{34}\rangle$")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig("plots/amplitude_0011.png", dpi=200); plt.close()

def plot_error_scaling():
    g = np.logspace(-2, -0.3, 40)
    ex = np.array([diagonalize(H_exact(x))[0][0] for x in g])
    pt3 = np.array([E_ref(x) + E2(x) + E3(x) for x in g])
    var = np.array([E_var(x) for x in g])
    plt.figure(figsize=(7, 5))
    plt.loglog(g, np.abs(pt3 - ex), "o-", label=r"$|E_{ref}+E^{(2)}+E^{(3)}-E_{exact}|$")
    plt.loglog(g, np.abs(var - ex), "s-", label=r"$|\langle H\rangle-E_{exact}|$")
    plt.loglog(g, g**4, "k--", lw=1, label=r"$g^4$")
    plt.loglog(g, g**3, "k:", lw=1, label=r"$g^3$")
    plt.xlabel(r"$g$"); plt.ylabel("energy error"); plt.grid(alpha=.3, which="both")
    plt.legend(); plt.tight_layout(); plt.savefig("plots/error_scaling.png", dpi=200); plt.close()



# --------------------------------------------------
# 5. Run
# --------------------------------------------------

if __name__ == "__main__":
    g_values = np.linspace(-1, 1, 201)
    r = calculate_results(g_values)

    plot_energies(r); plot_amplitudes(r); plot_0011(r); plot_error_scaling()
    
    print("max |<H> - formula| :", np.max(np.abs(r["E_var"] - r["E_formula"])))
    print("variational >= exact everywhere:", np.all(r["E_var"] >= r["E_exact"] - 1e-12))
    print("CI >= exact everywhere:", np.all(r["E_ci"] >= r["E_exact"] - 1e-12))
    for gt in [-1.0, -0.5, 0.5, 1.0]:
        i = np.argmin(abs(r["g"] - gt))
        print(f"\ng={gt}")
        print(" exact amps |13>,|14>,|23>,|24>,|34>:", np.round(r["c_exact"][i][1:], 4))
        print(" CI amps                              :", np.round(r["c_ci"][i][1:], 4))
        print(" Omega1 amps                          :", np.round(r["t1"][i], 4), " 0")
        print(" E exact, CI, var, PT3:", np.round([r["E_exact"][i], r["E_ci"][i], r["E_var"][i], r["E_pt3"][i]], 5))
