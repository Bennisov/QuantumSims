import numpy
import matplotlib.pyplot as plt
import numba


@numba.njit
def exp_x(phi, x):
    return dx * numpy.sum(numpy.abs(phi) ** 2 * x)


@numba.njit
def norm(phi, dx):
    const = numpy.sum(numpy.abs(phi) ** 2) * dx
    if const <= 0:
        raise ValueError("C < 0")
    return phi / numpy.sqrt(const)


@numba.njit
def ham(phi, bul=True):
    n = phi.shape[0]
    var = numpy.zeros(n, dtype=numpy.complex128)
    for j in range(1, n - 1):
        if bul:
            var[j] = (
                    -(phi[j + 1] + phi[j - 1] - 2 * phi[j]) / (2 * m * dx ** 2)
                    + (m * w ** 2 * x[j] ** 2) / 2 * phi[j]
            )
        else:
            var[j] = -(phi[j + 1] + phi[j - 1] - 2 * phi[j]) / (2 * m * dx ** 2)
    return var


@numba.njit
def time_evo(phi0, w, nt, bul=True):
    n = phi0.shape[0]
    num_snapshots = nt // 500
    phis = numpy.zeros((num_snapshots + 1, n), dtype=numpy.complex128)
    phi1 = phi0
    phi2 = phi0 * numpy.exp(-1j * w * dt / 2)
    phi2 = norm(phi2, dx)

    j = 0
    for i in range(nt):
        phi3 = phi1 + 2 * dt / 1j * ham(phi2, bul)
        phi1 = phi2
        phi2 = phi3

        # Save snapshots at intervals of 500 time steps
        if i % 500 == 0:
            phis[j, :] = phi3
            j += 1

    return phis


# Parameters
dt = 1.0
m = 0.067
len_scale = 0.05292
e_scale = 27211.6
w = 5.0 / e_scale

# Grid setup
N = 201
x = numpy.linspace(-100, 100, N, dtype=numpy.float64)
x = x / len_scale
dx = x[1] - x[0]
t_scale = 2.42 * 10 ** (-5)

phi0 = numpy.zeros(N, dtype=numpy.complex128)
x0 = 30.0 / len_scale
for i in range(1, N - 1):
    phi0[i] = numpy.exp(-m * w * (x[i] - x0) ** 2 / 2)

phi0 = norm(phi0, dx)

T = 2 * numpy.pi / w
nt = int(10 * T)

plt.figure()
plt.plot(x, numpy.abs(phi0) ** 2, label="|phi0|**2")
plt.xlabel("x [nm]")
plt.ylabel("|phi0|**2")
plt.legend()
plt.show()

phis = time_evo(phi0, w, nt)

plt.figure()
plt.imshow(
    numpy.abs(phis),
    extent=[x[0] * len_scale, x[-1] * len_scale, 0, nt * dt * t_scale],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
plt.colorbar(label="|phi|")
plt.xlabel("x [nm]")
plt.ylabel("Time [ps]")
plt.show()
ts = numpy.linspace(0, nt * dt, phis.shape[0])
ex = numpy.zeros(phis.shape[0])
ex_teo = x0 * numpy.cos(w * ts)
for i in range(phis.shape[0]):
    ex[i] = exp_x(phis[i, :], x)
plt.figure()
plt.plot(ts * t_scale, ex*len_scale, label="calc exp x")
plt.plot(ts * t_scale, ex_teo*len_scale, label="teo exp x")
plt.xlabel("time [ns]")
plt.ylabel("x [nm]")
plt.legend(loc="best")
plt.show()

x0 = 0.
for i in range(1, N - 1):
    phi0[i] = numpy.exp(-m * w * (x[i] - x0) ** 2 / 2)

phi0 = norm(phi0, dx)

phis = time_evo(phi0, w, nt)

plt.figure()
plt.imshow(
    numpy.abs(phis),
    extent=[x[0] * len_scale, x[-1] * len_scale, 0, nt * dt * t_scale],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
plt.colorbar(label="|phi|")
plt.xlabel("x [nm]")
plt.ylabel("Time [ps]")
plt.show()

phis = time_evo(phi0, w, nt, bul=False)

plt.figure()
plt.imshow(
    numpy.abs(phis),
    extent=[x[0] * len_scale, x[-1] * len_scale, 0, nt * dt * t_scale],
    origin="lower",
    aspect="auto",
    cmap="viridis"
)
plt.colorbar(label="|phi|")
plt.xlabel("x [nm]")
plt.ylabel("Time [ps]")
plt.show()


