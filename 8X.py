import numba
import numpy as np
import numpy
import matplotlib.pyplot as plt
import math
from matplotlib import cm

L = 100/.05292
m = 0.067
energy_scale = 27211.6
N1 = 100
x_scale = 0.05292


@numba.njit
def norm(delx, phi, N):
    sum = 0.
    for i in range(N+1):
        sum += phi[i]**2
    sum *= delx
    phi /= math.sqrt(sum)
    return phi


@numba.njit
def sch(E, V, delx, N, m=m):
    phi = numpy.zeros(N+1, dtype=float)
    phi[0] = 0.
    phi[1] = 1.
    for i in range(2,N+1):
        phi[i] = - 2 * m * (E - V[i]) * delx * delx * phi[i-1] - phi[i-2] + 2 * phi[i-1]
    return phi


@numba.njit
def sch_faster(E, V, delx, N, m=m):
    phi = numpy.zeros(N+1, dtype=float)
    phi[0] = 0.
    phi[1] = 1.
    for i in range(2,N+1):
        phi[i] = - 2 * m * (E - V[i]) * delx * delx * phi[i-1] - phi[i-2] + 2 * phi[i-1]
    return phi[-1]


#zadanie 1
Es = numpy.arange(start=0., stop=35, step=35/1000.)
Es /= energy_scale
V = numpy.zeros(N1+1)
delx = L/N1
xs = numpy.arange(start=0., stop=L+delx, step=delx, dtype=float)
xs *= x_scale
phins = numpy.zeros(len(Es))
for i in range(len(Es)):
    phins[i] = sch_faster(m=m, E=Es[i], V=V, delx=delx, N=N1)
Es *= energy_scale
plt.figure(dpi=500)
plt.plot(Es, phins)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('E[meV]')
plt.ylabel('$\psi_{N}$')
plt.savefig("1_1.png")
plt.show()
phi1 = sch(m=m, E=2.3/energy_scale, V=V, delx=delx, N=N1)
phi2 = sch(m=m, E=2.3*1.05/energy_scale, V=V, delx=delx, N=N1)
phi3 = sch(m=m, E=2.3*0.95/energy_scale, V=V, delx=delx, N=N1)
phi1 = norm(delx, phi1, N1)
phi2 = norm(delx, phi2, N1)
phi3 = norm(delx, phi3, N1)
plt.figure(dpi=500)
plt.axhline(y=0, color='r', linestyle='--')
plt.plot(xs, phi1, label="E = 2.3 meV")
plt.plot(xs, phi2, label="E = 105% * 2.3 meV")
plt.plot(xs, phi3, label="E = 95% * 2.3 meV")
plt.legend(loc='best')
plt.xlabel('x [nm]')
plt.ylabel('$\psi(x)$')
plt.savefig('1_2.png')
plt.show()

#zad2
@numba.njit
def gola(x1, x2, V, delx, N):
    fx1 = sch_faster(E=x1, V=V, delx=delx, N=N)
    fx2 = sch_faster(E=x2, V=V, delx=delx, N=N)
    if fx1 * fx2 > 0.:
        print("WTF")
        return -9999, -9999
    x3 = (x1 + x2) / 2
    fx3 = sch_faster(E=x3, V=V, delx=delx, N=N)
    if fx1 * fx3 > 0.:
        x1 = x3
    if fx2 * fx3 > 0.:
        x2 = x3
    return x1, x2


Es /= energy_scale
x1 = Es[0]
x2 = Es[1]
zeros = numpy.zeros(7)
ind = 0
for j in range(7):
    fx1 = sch_faster(E=x1, V=V, delx=delx, N=N1)
    for k in range(ind+1, len(Es)):
        fx2 = sch_faster(E=Es[k], V=V, delx=delx, N=N1)
        if fx1*fx2 < 0.:
            x2 = Es[k]
            ind = k
            break
    while 1:
        if x2 - x1 <= (0.000001 / energy_scale):
            break
        x1, x2 = gola(x1=x1, x2=x2, V=V, delx=delx, N=N1)
    zeros[j] = (x1+x2)/2.
    x1 = Es[ind]
    x2 = Es[ind]
x_ana = numpy.zeros(7)
for i in range(7):
    x_ana[i] = (i+1)**2 * math.pi ** 2 /(2 * m * L**2)

N2 = 300
delx2 = L/N2
V = numpy.zeros(N2+1)

x1 = Es[0]
x2 = Es[1]
zeros1 = numpy.zeros(7)
ind = 0
for j in range(7):
    fx1 = sch_faster(E=x1, V=V, delx=delx2, N=N2)
    for k in range(ind+1, len(Es)):
        fx2 = sch_faster(E=Es[k], V=V, delx=delx2, N=N2)
        if fx1*fx2 < 0.:
            x2 = Es[k]
            ind = k
            break
    while 1:
        if x2 - x1 <= (0.000001 / energy_scale):
            break
        x1, x2 = gola(x1=x1, x2=x2, V=V, delx=delx2, N=N2)
    zeros1[j] = (x1+x2)/2.
    x1 = Es[ind]
    x2 = Es[ind]
for i in range(7):
    zeros[i] = f"{zeros[i]:.4g}"
    zeros1[i] = f"{zeros1[i]:.4g}"
    x_ana[i] = f"{x_ana[i]:.4g}"
for i in range(7):
    print(str(i+1) + ". N=100: " + str(zeros[i]) + " N=300: " + str(zeros1[i]) + " ana: " + str(x_ana[i]))

Ws = numpy.arange(start=0., stop=1000., step=1000./100.)
Ws /= energy_scale
Es = numpy.arange(start=-50., stop=35., step=1/10.)
Es /= energy_scale
zerooos = [None] * len(Ws)
for i in range(len(Ws)):
    V[int(N2/2)] = -Ws[i]
    x1 = Es[0]
    x2 = Es[1]
    zeros = numpy.zeros(7)
    ind = 0
    for j in range(7):
        fx1 = sch_faster(E=x1, V=V, delx=delx2, N=N2)
        for k in range(ind+1, len(Es)):
            fx2 = sch_faster(E=Es[k], V=V, delx=delx2, N=N2)
            if fx1*fx2 < 0.:
                x2 = Es[k]
                ind = k
                break
        while 1:
            if x2 - x1 <= (0.000001 / energy_scale):
                break
            x1, x2 = gola(x1=x1, x2=x2, V=V, delx=delx2, N=N2)
        zeros[j] = (x1+x2)/2.
        x1 = Es[ind]
        x2 = Es[ind]
    zerooos[i] = zeros

plt.figure(dpi=500)
zerooos = numpy.array(zerooos)
zerooos *= energy_scale
Ws *= energy_scale
for i in range(7):
    plt.scatter(Ws, zerooos[:, i], label=(str(i+1)+' miejsce zerowe'), s=5)
plt.ylim(-50, 35)
plt.xlabel("W [meV]")
plt.ylabel("$E_{N}$ [meV]")
plt.legend(loc='best')
plt.savefig('1_4.png')
plt.show()
ind = np.where(Ws == 500.)
ind = ind[0][0]
zerooos /= energy_scale
Ws /= energy_scale
plt.figure(dpi=500)
xs = numpy.arange(start=0., stop=L+delx2, step=delx2, dtype=float)
xs *= x_scale
V[int(N2/2)] = -500./energy_scale
for i in range(2):
    E = zerooos[ind, i]
    phi = sch(E=E, V=V, delx=delx2, N=N2)
    phi = norm(delx=delx2, phi=phi, N=N2)
    numpy.savetxt('phi'+str(i)+'.txt', phi)
    plt.plot(xs, phi, label=str(i+1)+' miejsce zerowe')
plt.legend(loc='best')
plt.xlabel('x [nm]')
plt.ylabel('$\psi(x)$')
plt.savefig('1_5.png')
plt.show()
# WORK IN PROGRESS
plt.figure(dpi=500)
for i in range(2):
    for j in range(0, len(Ws), int(len(Ws)/5)):
        V[int(N2 / 2)] = -Ws[j]
        E = zerooos[j, i]
        phi = sch(E=E, V=V, delx=delx2, N=N2)
        phi = norm(delx=delx2, phi=phi, N=N2)
        Ws *= energy_scale
        color = 'red'
        if i==1:
            color = 'blue'
        plt.plot(xs, phi, label="MZ: "+str(i+1)+",W="+str(f"{Ws[j]/1000:.2g}"), color=color)
        Ws /= energy_scale
plt.legend(loc='best', ncol=2)
plt.xlabel("x [nm]")
plt.ylabel("$\Psi$")
plt.savefig("1_6.png")
plt.show()
