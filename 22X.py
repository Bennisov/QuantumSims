import random

import numba
import numpy as np
import numpy
import matplotlib.pyplot as plt
import math

L = 100 / .05292
m = 0.067
energy_scale = 27211.6
N = 300
x_scale = 0.05292
dx = L / N


@numba.njit
def ham(alpha, phi, dx, V, N, m=m):
    temp = numpy.zeros(N + 1, dtype=float)
    for i in range(1, N):
        temp[i] = - (phi[i + 1] + phi[i - 1] - 2 * phi[i]) / (2 * m * dx ** 2) + V[i] * phi[i]
    for i in range(1, N):
        temp[i] = phi[i] - alpha * temp[i]
    return temp


@numba.njit
def norm(delx, phi, N):
    sum = 0.
    for i in range(N + 1):
        sum += phi[i] ** 2
    sum *= delx
    phi /= math.sqrt(sum)
    return phi


@numba.njit
def energy(phi, dx, V, N, m=m):
    e = 0
    for i in range(1, N):
        temp = - (phi[i + 1] + phi[i - 1] - 2 * phi[i]) / (2 * m * dx ** 2) + V[i] * phi[i]
        e += phi[i] * dx * temp
    return e


@numba.njit
def ortn(phi, dx, phi2):
    c1 = 0.
    for i in range(N+1):
        c1 += phi[i] * phi2[i] * dx
    for i in range(N+1):
        phi2[i] = phi2[i] - c1 * phi[i]
    return phi2


def sch(V, delx, N, alpha, tol=1e-6):
    phi = numpy.random.uniform(-1, 1, N + 1)
    it = 0
    eold = 99999
    es = numpy.array([])
    while 1:
        phi[0] = 0.
        phi[-1] = 0.
        phi = ham(alpha=alpha, phi=phi, dx=delx, V=V, N=N)
        phi = norm(delx, phi, N)
        enew = energy(phi=phi, dx=delx, V=V, N=N)
        if not (it % 1000):
            print(it)
            print(enew * energy_scale)
        if (abs(enew - eold) * energy_scale) < tol:
            break
        eold = enew
        it += 1
        es = numpy.append(es, enew)
        if it > 1e9:
            break
    return phi, es


def zad2(V, delx, N, m=m, tol=1e-6):
    alpha = 0.95 * m * delx ** 2
    phi, _ = sch(V=V, delx=delx, N=N, alpha=alpha, tol=tol)
    phi2 = numpy.random.uniform(-1, 1, N + 1)
    it = 0
    eold = 99999
    es = numpy.array([])
    while 1:
        phi2[0] = 0.
        phi2[-1] = 0.
        phi2 = ham(alpha=alpha, phi=phi2, dx=delx, V=V, N=N)
        phi2 = ortn(phi, delx, phi2)
        phi2 = norm(delx, phi2, N)
        enew = energy(phi=phi2, dx=delx, V=V, N=N)
        if not (it % 1000):
            print(it)
            print(enew * energy_scale)
        if (abs(enew - eold) * energy_scale) < tol:
            break
        eold = enew
        it += 1
        es = numpy.append(es, enew)
        if it > 1e9:
            break
    return phi, phi2, es


V = numpy.zeros(N + 1, dtype=float)
plt.figure()
a = numpy.array([.9, .95, .97, .99, 1.])
for alphas in a:
    alpha = alphas * m * dx * dx
    phi, es = sch(V=V, delx=dx, N=N, alpha=alphas)
    es = numpy.array(es)
    es *= energy_scale
    plt.plot(numpy.arange(0, len(es), 1), es, label=str(alphas))
plt.ylabel('Energy [meV]')
plt.xlabel('Iteracja')
plt.xscale('log')
plt.ylim(numpy.min(es), 1000)
plt.legend()
plt.savefig('2_1.png')
#plt.show()
phi, phi2, es = zad2(V=V, delx=dx, N=N)
xs = numpy.arange(start=0., stop=L+dx, step=dx, dtype=float)
xs *= x_scale
plt.figure()
plt.plot(xs, phi, label='stan podstawowy')
plt.plot(xs, phi2, label='1. stan wzbudzony')
plt.xlabel("x [nm]")
plt.ylabel("$\Psi(x)$")
plt.legend()
plt.savefig('2_2.png')
# plt.show()
plt.figure()
es *= energy_scale
plt.plot(numpy.arange(0, len(es), 1), es)
plt.xscale('log')
plt.xlabel("Iteracja")
plt.ylabel("Energia [meV]")
plt.savefig('2_3.png')
# plt.show()
V[int(N/2)] = -500./energy_scale
phi, phi2, es = zad2(V=V, delx=dx, N=N, tol=1e-9)
plt.figure()
plt.plot(xs, phi, label='stan podstawowy')
plt.plot(xs, phi2, label='1. stan wzbudzony')
plt.xlabel("x [nm]")
plt.ylabel("$\Psi(x)$")
plt.legend()
plt.savefig('2_4.png')
plt.show()
plt.figure()
es *= energy_scale
plt.plot(numpy.arange(0, len(es), 1), es)
plt.xscale('log')
plt.xlabel("Iteracja")
plt.ylabel("Energia [meV]")
plt.savefig('2_5.png')
#plt.show()
phi_s = numpy.loadtxt('phi0.txt')
phi_s1 = numpy.loadtxt('phi1.txt')
plt.figure()
plt.plot(xs, abs(phi_s)-abs(phi), label='Stan podstawowy')
plt.plot(xs, abs(phi_s1)-abs(phi2), label='1 Stan wzbudzony')
plt.xlabel("x [nm]")
plt.ylabel("$\Delta \Psi(x)$")
plt.legend()
plt.savefig('2_6.png')
plt.show()
