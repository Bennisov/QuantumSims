import numpy
import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy

m = 0.067
energy_scale = 27211.6
len_scale = 0.05292
nx = 100
w = 10 / energy_scale


@numba.njit
def phi(j, x, L):
    return np.sqrt(2 / L) * np.sin(j * np.pi * x / L)


@numba.njit
def e_calc(j, L):
    return (np.pi ** 2 * j ** 2) / (2 * m * L ** 2)


@numba.njit
def v_calc(L, i, j):
    if i == j:
        return L ** 2 * (i ** 2 * np.pi ** 2 - 6) * m * w ** 2 / (24 * i ** 2 * np.pi ** 2)
    else:
        return 2 * L ** 2 * i * j * m * w ** 2 * ((-1) ** (i + j) + 1) / (np.pi ** 2 * (-i + j) ** 2 * (i + j) ** 2)


@numba.njit
def psi_calc(c, N, L):
    xs = numpy.linspace(0, L, nx)
    psi = numpy.zeros(xs.shape)
    for j in range(xs.shape[0]):
        for i in range(1, N+1):
            psi[j] += c[i-1] * phi(i, xs[j], L)
    return psi


fig_it = 0
Ls = numpy.array([100, 20, 200]) / len_scale
for L in Ls:
    all_energies = numpy.full((18, 6), np.nan)
    psis1 = numpy.zeros((18, nx))
    psis2 = numpy.zeros((18, nx))
    psis3 = numpy.zeros((18, nx))
    it = 0
    for n in range(3, 21):
        V = numpy.zeros((n, n))
        E = numpy.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                V[i - 1, j - 1] = v_calc(L, i, j)
                if i == j:
                    E[i - 1, j - 1] = e_calc(i, L)
        H = E + V
        energies, c = scipy.linalg.eigh(H)
        len_energies = energies.size
        if len_energies <= 6:
            all_energies[it, :len_energies] = energies
        else:
            all_energies[it] = energies[:6]
        print(f"e shape: {energies.shape}, c shape: {c.shape}")
        psis1[it] = psi_calc(c[:, 0], n, L)
        psis2[it] = psi_calc(c[:, 1], n, L)
        psis3[it] = psi_calc(c[:, 2], n, L)
        it += 1
    plt.figure()
    ns = numpy.linspace(3, 21, 18, dtype=int)
    for i in range(6):
        plt.plot(ns, all_energies[:, i]*energy_scale, label=f"{i}. energy level")
    plt.legend(loc="best")
    plt.xlabel("N")
    plt.ylabel("Energy [meV]")
    plt.savefig(f"3XII_{fig_it}.png")
    plt.show()
    fig_it += 1
    plt.figure()
    xs = numpy.linspace(0, L, nx)
    colors = plt.cm.get_cmap('autumn_r')(np.linspace(0, 1, psis1.shape[0]))
    for i in range(psis1.shape[0]):
        plt.plot(xs * len_scale, psis1[i, :], color=colors[i])
    plt.xlabel("x [nm]")
    plt.ylabel("$\Psi_1(x)$")
    plt.savefig(f"3XII_{fig_it}.png")
    plt.show()
    fig_it += 1
    plt.figure()
    for i in range(psis2.shape[0]):
        plt.plot(xs * len_scale, psis2[i, :], color=colors[i])
    plt.xlabel("x [nm]")
    plt.ylabel("$\Psi_2(x)$")
    plt.savefig(f"3XII_{fig_it}.png")
    plt.show()
    fig_it += 1
    plt.figure()
    for i in range(psis3.shape[0]):
        plt.plot(xs * len_scale, psis3[i, :], color=colors[i])
    plt.xlabel("x [nm]")
    plt.ylabel("$\Psi_3(x)$")
    plt.savefig(f"3XII_{fig_it}.png")
    plt.show()
    fig_it += 1
