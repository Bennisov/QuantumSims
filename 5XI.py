import numba
import numpy as np
import numpy
import matplotlib.pyplot as plt
import math
import cmath

m = 0.067
energy_scale = 27211.6
x_scale = 0.05292
size = 340

V = numpy.zeros(size, dtype=float)
phi = numpy.zeros(size, dtype=np.complex128)
dx = 0.5 / x_scale

V[100:121] = 10
V[220:241] = 10

V = V / energy_scale
phi[-1] = 1


#@numba.njit
def schr(E, phi=phi, V=V, dx=dx, size=size):
    q = math.sqrt(2 * m * E)
    phi[size-2] = cmath.exp(-1j * q * dx)
    for i in range(size-3, -1, -1):
        phi[i] = - 2 * m * (E - V[i+1]) * dx**2 * phi[i+1] - phi[i+2] + 2 * phi[i+1]
    return phi


E1 = 7/energy_scale
phi = schr(E=E1)
x1 = 10 * dx
x2 = 11 * dx
phi1 = phi[10]
phi2 = phi[11]
q = math.sqrt(2 * m * E1)
A = phi1*cmath.exp(1j * q * x1) - phi2*cmath.exp(1j * q * x2)
A /= (cmath.exp(1j*q*x1))**2 - (cmath.exp(1j*q*x2))**2
B = -(-phi2*cmath.exp(1j*q*x1) + phi1*cmath.exp(1j*q*x2)) * cmath.exp((1j * q * x1)+(1j*q*x2))
B /= (cmath.exp(1j*q*x1))**2 - (cmath.exp(1j*q*x2))**2
R = (cmath.polar(B)[0])**2 / (cmath.polar(A)[0]**2)
T = 1 / (cmath.polar(A)[0]**2)
print(T,R)
phi_modsq = numpy.zeros(size, dtype=np.complex128)
for i in range(size):
    phi_modsq[i] = cmath.polar(phi[i])[0] ** 2
phi_modsq = numpy.array(phi_modsq.real, dtype=float)
phi_teo = numpy.zeros(size, dtype=float)
for i in range(size):
    x = i * dx
    var = A * cmath.exp(1j*q*x) + B * cmath.exp(-1j*q*x)
    phi_teo[i] = cmath.polar(var)[0] ** 2
xs = numpy.arange(0, dx*size, dx)
plt.figure()
plt.plot(xs*x_scale, V*energy_scale, label="V[meV]")
plt.plot(xs*x_scale, phi_modsq, label="$|\phi|^2$")
plt.plot(xs*x_scale, phi_teo, label="$|\phi_<|^2$")
plt.legend(loc="best")
plt.xlabel("x[nm]")
plt.savefig("5XI_1.png")
plt.show()
dE = 50 / (10000 * energy_scale)
Es = numpy.arange(0 + dE, dE * 10000 + dE, dE)
Ts = numpy.zeros(10000)
Rs = numpy.zeros(10000)
for i in range(10000):
    phi = numpy.zeros(size, dtype=np.complex128)
    phi[-1] = 1
    phi = schr(E=Es[i], phi=phi)
    x1, x2, phi1, phi2, q, A, B, R, T = 0, 0, 0, 0, 0, 0, 0, 0, 0
    x1 = 10 * dx
    x2 = 11 * dx
    phi1 = phi[10]
    phi2 = phi[11]
    q = math.sqrt(2 * m * Es[i])
    A = phi1 * cmath.exp(1j * q * x1) - phi2 * cmath.exp(1j * q * x2)
    A /= (cmath.exp(1j * q * x1)) ** 2 - (cmath.exp(1j * q * x2)) ** 2
    B = -(-phi2 * cmath.exp(1j * q * x1) + phi1 * cmath.exp(1j * q * x2)) * cmath.exp((1j * q * x1)+(1j*q*x2))
    B /= (cmath.exp(1j * q * x1)) ** 2 - (cmath.exp(1j * q * x2)) ** 2
    R = (cmath.polar(B)[0]) ** 2 / (cmath.polar(A)[0] ** 2)
    T = 1 / (cmath.polar(A)[0] ** 2)
    Ts[i] = T
    Rs[i] = R
plt.figure()
plt.plot(Es*energy_scale,Ts,label="T")
plt.plot(Es*energy_scale,Rs,label="R")
plt.legend(loc='best')
plt.xlabel("E[meV]")
plt.savefig("5XI_2.png")
plt.show()
eps = 1e-3
ind = np.where(Ts > 1. - eps)
ind = ind[0][:]
for i in range(len(ind) - 2, -1, -1):
    if (ind[i + 1] - ind[i]) == 1:
        ind = np.delete(ind, i + 1)
Es1 = Es[ind[:4]]
plt.figure()
plt.plot(xs*x_scale, V*energy_scale, label="V")
for i in range(4):
    phi = schr(E=Es1[i])
    phi_modsq = numpy.zeros(size, dtype=np.complex128)
    for j in range(size):
        phi_modsq[j] = cmath.polar(phi[j])[0] ** 2
    phi_modsq = numpy.array(phi_modsq.real, dtype=float)
    plt.plot(xs*x_scale,phi_modsq,label=f"E={Es1[i]*energy_scale:.1f}[meV]")
plt.legend(loc='best')
plt.xlabel("x[nm]")
plt.savefig("5XI_3.png")
plt.show()