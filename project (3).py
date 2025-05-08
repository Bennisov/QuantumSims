from ast import Constant
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as scp
import imageio.v2 as imageio  # For creating GIFs

N_init = 2
L_init = 100

len_scale = 0.05292
e_scale = 27211.6
t_scale = 0.02418  # fs

fps = 10
time_evo_freq = 4
h = 1
m = 0.067

frames = []

path = "./wyniki"
temp_path = "./temp"
gif_path = f"{path}/Time_evolution.gif"


def proj(L, N, show_arrays_for_energies=False, do_time_evo=False):
    dt = 100  # j.a
    w = 10.0 / e_scale  # meV
    n_ksi = 10
    L = L / len_scale
    a = L / (2 * N)
    d_ksi = a / n_ksi
    output = {
        "dt": dt,
        "a_val": a,
        "nodes": [],
        "elements": [],
        "reshaped_array": [],
        "reshaped_arrays": [],
        "H_glo": [],
        "S_glo": [],
        "S": [],
        "T": [],
        "V": [],
        "eigval": [],
        "eigvec": [],
        "xt": [],
        "time_evo_arrays": [],
    }

    def generate_mesh(L, N):
        num_nodes = 2 * N + 1
        x_vals = np.linspace(-L / 2, L / 2, num_nodes)
        y_vals = np.linspace(-L / 2, L / 2, num_nodes)
        nodes = [(x, y) for y in y_vals for x in x_vals]
        elements = []
        for i in range(1, 2 * N + 1):
            for j in range(1, 2 * N + 1):
                n1 = (i - 1) * num_nodes + j
                n2 = n1 + 1
                n3 = n1 + num_nodes
                n4 = n3 + 1
                elements.append([n1, n2, n3, n4])
        return np.array(nodes), np.array(elements)

    def ind_tran(i, j):
        return i * (2 * N + 1) + j

    def inv_tran(n):
        i = int(n / (2 * N + 1))
        j = n % (2 * N + 1)
        return i, j

    def f(i, ksi):
        return (1 + (-1) ** i * ksi) / 2

    def g(i, ksix, ksiy):
        match i:
            case 0:
                return f(1, ksix) * f(1, ksiy)
            case 1:
                return f(2, ksix) * f(1, ksiy)
            case 2:
                return f(1, ksix) * f(2, ksiy)
            case 3:
                return f(2, ksix) * f(2, ksiy)

    def w_ind(i):
        match i:
            case 0:
                return 5 / 9
            case 1:
                return 8 / 9
            case 2:
                return 5 / 9

    def p_ind(i):
        match i:
            case 0:
                return -math.sqrt(3 / 5)
            case 1:
                return 0
            case 2:
                return math.sqrt(3 / 5)

    def derx(i, l, n):
        return (g(i, p_ind(l) + d_ksi, p_ind(n)) - g(i, p_ind(l) - d_ksi, p_ind(n))) / (
            2 * d_ksi
        )

    def dery(i, l, n):
        return (g(i, p_ind(l), p_ind(n) + d_ksi) - g(i, p_ind(l), p_ind(n) - d_ksi)) / (
            2 * d_ksi
        )

    def calculate_S_matrix(i, j):
        s_ji = 0
        for l in range(3):
            for n in range(3):
                s_ji += (
                    w_ind(l)
                    * w_ind(n)
                    * g(j, p_ind(l), p_ind(n))
                    * g(i, p_ind(l), p_ind(n))
                )
        return a**2 / 4 * s_ji

    def calculate_T_matrix(i, j):
        t_ji = 0
        for l in range(3):
            for n in range(3):
                t_ji += (
                    w_ind(l)
                    * w_ind(n)
                    * (derx(j, l, n) * derx(i, l, n) + dery(j, l, n) * dery(i, l, n))
                )
        return h**2 / (2 * m) * t_ji

    def calculate_V_matrix(k, i, j):
        v_ji = 0
        x_nlg1 = nodes[elements[k, 0] - 1, 0]
        x_ngl2 = nodes[elements[k, 1] - 1, 0]

        y_nlg1 = nodes[elements[k, 0] - 1, 1]
        y_ngl3 = nodes[elements[k, 2] - 1, 1]
        for l in range(3):
            for n in range(3):
                x = (1 - p_ind(l)) / 2 * x_nlg1 + (1 + p_ind(l)) / 2 * x_ngl2
                y = (1 - p_ind(n)) / 2 * y_nlg1 + (1 + p_ind(n)) / 2 * y_ngl3
                v_ji += (
                    w_ind(l)
                    * w_ind(n)
                    * g(j, p_ind(l), p_ind(n))
                    * g(i, p_ind(l), p_ind(n))
                    * (x**2 + y**2)
                )
        return a**2 * m * w**2 / 8 * v_ji

    def merge_mat(S, T, V):
        h_gl = np.zeros(((2 * N + 1) ** 2, (2 * N + 1) ** 2))
        s_gl = np.zeros(((2 * N + 1) ** 2, (2 * N + 1) ** 2))
        for k in range(elements.shape[0]):
            for i1 in range(4):
                for i2 in range(4):
                    s_gl[elements[k, i1] - 1, elements[k, i2] - 1] += S[i1, i2]
                    h_gl[elements[k, i1] - 1, elements[k, i2] - 1] += (
                        T[i1, i2] + V[k, i1, i2]
                    )
        for i in range(s_gl.shape[0]):
            var1 = int(i / (2 * N + 1))
            var2 = i % (2 * N + 1)
            if (var1 * var2 == 0) or (var1 == 2 * N) or (var2 == 2 * N):
                s_gl[i, :] = 0.0
                h_gl[i, :] = 0.0
                s_gl[:, i] = 0.0
                h_gl[:, i] = 0.0
                s_gl[i, i] = 1.0
                h_gl[i, i] = -1410.0
        return h_gl, s_gl

    def generate_wave_func(node_values=[]):
        psi_node = np.zeros((2 * N + 1, 2 * N + 1))
        for i in range(psi_node.shape[0]):
            for j in range(psi_node.shape[1]):
                ind = ind_tran(i, j)
                psi_node[i, j] = math.exp(
                    -m * w / 2 * (nodes[ind, 0] ** 2 + nodes[ind, 1] ** 2)
                )
                if len(node_values) != 0:
                    psi_node[i, j] = node_values[ind]

        psi_element = np.zeros((elements.shape[0], n_ksi, n_ksi))
        for k in range(elements.shape[0]):
            ksixs = np.linspace(-1.0, 1.0, n_ksi)
            ksiys = np.linspace(-1.0, 1.0, n_ksi)
            for i in range(len(ksixs)):
                for j in range(len(ksiys)):
                    ksix = ksixs[i]
                    ksiy = ksiys[j]
                    for l in range(4):
                        ind = elements[k, l] - 1
                        indi, indj = inv_tran(ind)
                        psi_element[k, i, j] += psi_node[indi, indj] * g(l, ksix, ksiy)
        reshaped_array = np.zeros(((2 * N) * n_ksi, (2 * N) * n_ksi))
        for idx in range(elements.shape[0]):
            col_start = (idx // (2 * N)) * n_ksi
            row_start = (idx % (2 * N)) * n_ksi
            reshaped_array[
                row_start : row_start + n_ksi, col_start : col_start + n_ksi
            ] = psi_element[idx]
        return reshaped_array

    def generate_mat_and_solve_own_eq(elements):
        S = np.zeros((4, 4))
        T = np.zeros((4, 4))
        V = np.zeros((4 * N**2, 4, 4))
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                S[i, j] = calculate_S_matrix(i, j)
                T[i, j] = calculate_T_matrix(i, j)
                for k in range(elements.shape[0]):
                    V[k, i, j] = calculate_V_matrix(k, i, j)
        H_glo, S_glo = merge_mat(S, T, V)
        eigval, eigvec = scp.linalg.eigh(H_glo, S_glo)
        indices = np.where(eigval > 0)[0]
        eigval = eigval[indices]
        eigvec = eigvec[:, indices]
        return eigval, eigvec, H_glo, S_glo, S, T, V

    def calculate_X_matrix(k, i, j):
        x_ji = 0
        x_nlg1 = nodes[elements[k, 0] - 1, 0]
        x_ngl2 = nodes[elements[k, 1] - 1, 0]
        for l in range(3):
            for n in range(3):
                x = (1 - p_ind(l)) / 2 * x_nlg1 + (1 + p_ind(l)) / 2 * x_ngl2
                x_ji += (
                    w_ind(l)
                    * w_ind(n)
                    * g(j, p_ind(l), p_ind(n))
                    * g(i, p_ind(l), p_ind(n))
                    * x
                )
        return a**2 / 4 * x_ji

    def merge_xt(d):
        X_glo = np.zeros(((2 * N + 1) ** 2, (2 * N + 1) ** 2))
        X = np.zeros((4 * N**2, 4, 4))
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                for k in range(X.shape[0]):
                    X[k, i, j] = calculate_X_matrix(k, i, j)
        for k in range(X.shape[0]):
            for i1 in range(4):
                for i2 in range(4):
                    X_glo[elements[k, i1] - 1, elements[k, i2] - 1] += X[k, i1, i2]
        d_conj = d.conj()
        print(X_glo[0])
        xt = np.dot(np.dot(d_conj.T, X_glo), d)
        return xt

    def time_evo(E1, E2, c1, c2, S_glo, H_glo):
        dE = E2 - E1
        T = 2 * math.pi / dE
        nt = int(T / dt)
        xt = np.zeros(nt)
        A = S_glo - dt * H_glo / (2 * 1j)
        B = S_glo + dt * H_glo / (2 * 1j)
        d0 = c1 + c2
        time_evo_mat = np.zeros((nt, (2 * N + 1) ** 2))
        for i in range(nt):
            P = np.dot(B, d0)
            d1 = scp.linalg.solve(A, P)
            d0 = d1
            xt[i] = merge_xt(d1).real
            time_evo_mat[i] = (d1**2).real
        return time_evo_mat, xt

    nodes, elements = generate_mesh(L, N)
    reshaped_array = generate_wave_func()
    eigval, eigvec, H_glo, S_glo, S, T, V = generate_mat_and_solve_own_eq(elements)
    output["nodes"] = nodes
    output["elements"] = elements
    output["reshaped_array"] = reshaped_array
    output["eigval"] = eigval
    output["eigvec"] = eigvec
    output["H_glo"] = H_glo
    output["S_glo"] = S_glo
    output["S"] = S
    output["T"] = T
    output["V"] = V

    if show_arrays_for_energies:
        reshaped_arrays = []
        for i in range(eigvec.shape[1]):
            reshaped_arrays.append(generate_wave_func(eigvec[:, i]))
        output["reshaped_arrays"] = np.array(reshaped_arrays)
    if do_time_evo:
        time_evo_arrays = []
        time_evo_matrixes, xt = time_evo(
            eigval[0], eigval[1], eigvec[:, 0], eigvec[:, 1], S_glo, H_glo
        )
        for i in range(time_evo_matrixes.shape[0]):
            time_evo_arrays.append(generate_wave_func(time_evo_matrixes[i, :]))
        output["xt"] = xt
        output["time_evo_arrays"] = np.array(time_evo_arrays)

    return output


plt.style.use("dark_background")

# zadanie 1, 2
output = proj(N=2, L=100)
elements = output["elements"]
nodes = output["nodes"]
reshaped_array = output["reshaped_array"]
for i in range(elements.shape[0]):
    for j in range(elements.shape[1]):
        element_num = i + 1
        node_num = j + 1
        global_node = elements[i, j]
        node_x = nodes[global_node - 1][0]
        node_y = nodes[global_node - 1][1]

        print(
            f"Element {element_num}, Local_Node {node_num}: Global_Node {global_node}, "
            f"Coordinates: ({node_x}, {node_y})"
        )

fig_arr, ax_arr = plt.subplots(figsize=(19.2, 10.8))
X = np.linspace(-L_init / 2, L_init / 2, reshaped_array.shape[0])
Y = np.linspace(-L_init / 2, L_init / 2, reshaped_array.shape[1])
X, Y = np.meshgrid(X, Y)
contour = ax_arr.contourf(X, Y, reshaped_array, cmap="plasma", levels=100)
ax_arr.set_xlabel("x [nm]")
ax_arr.set_ylabel("y [nm]")
fig_arr.colorbar(contour, ax=ax_arr)
fig_arr.suptitle(f"$\Psi(x, y) $ for N={N_init} and L={L_init}")
fig_arr.savefig(f"{path}/Psi(xy)_N={N_init}_L={L_init}.png")

# zadanie 3, 4, 4a

output = proj(N=2, L=100)
S = output["S"]
T = output["T"]
V = output["V"]
H_glo = output["H_glo"]
S_glo = output["S_glo"]
a = output["a_val"]
print("S:")
print(" a^2 / 36 * [")
for i in range(S.shape[0]):
    print("[ ", end=" ")
    for j in range(S.shape[1]):
        print(f"{(S[i, j] * 9 * 4 / (a ** 2)):.0f}", end=" ")
    print("]")
print("]")

print("T:")
print(" h^2/12m [")
for i in range(T.shape[0]):
    print("[ ", end=" ")
    for j in range(T.shape[1]):
        print(f"{(T[i, j] * 2 * 6 * m / (h ** 2)):.0f}", end=" ")
    print("]")
print("]")

print("V^11: ")
print("[")
for i in range(V.shape[1]):
    print("[ ", end=" ")
    for j in range(V.shape[2]):
        print(f"{(V[10, i, j]):.2f}", end=" ")
    print("]")
print("]")


print("H: ")
print("[")
for i in range(H_glo.shape[0]):
    print("[ ", end=" ")
    for j in range(H_glo.shape[1]):
        print(f"{(H_glo[i, j]):.2f}", end=" ")
    print("]")
print("]")
print("S: ")
print("[")
for i in range(S_glo.shape[0]):
    print("[ ", end=" ")
    for j in range(S_glo.shape[1]):
        print(f"{(S_glo[i, j]):.2f}", end=" ")
    print("]")
print("]")

# zadanie 5, 6
L_min = 10
L_max = 500
L_num = 500
num_of_energies = 15
Ls = np.linspace(L_min, L_max, num=L_num)
energies_dep_L = np.zeros((L_num, num_of_energies))
for i in range(L_num):
    L = Ls[i]
    output = proj(L=L, N=N_init)
    eigval = output["eigval"]
    eigvec = output["eigvec"]
    if eigval.shape[0] < num_of_energies:
        energies_dep_L[i, :] = np.concatenate(
            (
                eigval,
                np.full(num_of_energies - eigval.size, np.nan),
            )
        )
    else:
        energies_dep_L[i, :] = eigval[:num_of_energies]

array_dep_L = np.argmin(energies_dep_L, axis=0)
sum = float(np.average(np.delete(array_dep_L, np.where(array_dep_L <= 0.0))) + L_min)
print(f"Optimal value of L: {sum:.4f} nm")

Ns = np.arange(1, 10, 1)
energies_dep_N = np.zeros((Ns.size, num_of_energies))
for i in range(Ns.size):
    N = Ns[i]
    output = proj(L=sum, N=N)
    eigval = output["eigval"]
    eigvec = output["eigvec"]
    if eigval.shape[0] < num_of_energies:
        energies_dep_N[i, :] = np.concatenate(
            (eigval, np.full(num_of_energies - eigval.size, np.nan))
        )
    else:
        energies_dep_N[i, :] = eigval[:num_of_energies]


L_opt = sum
N_opt = 6
output = proj(L=L_opt, N=N_opt, show_arrays_for_energies=True)
eigval_opt = np.round(output["eigval"], 8)
eigvec_opt = output["eigvec"]
reshaped_arrays_opt = output["reshaped_arrays"]

num_of_wave_plots = 6
fig_Ls, ax_Ls = plt.subplots(figsize=(19.2, 10.8))
fig_Ns, ax_Ns = plt.subplots(figsize=(19.2, 10.8))
figs_axis = [plt.subplots(figsize=(19.2, 10.8)) for _ in range(num_of_wave_plots)]

for i in range(num_of_energies):
    ax_Ls.plot(
        Ls,
        energies_dep_L[:, i] * e_scale,
        label=f"energy state {i+1}={eigval[i]*e_scale:.4f}",
    )
    ax_Ns.plot(
        Ns,
        energies_dep_N[:, i] * e_scale,
        label=f"energy state {i+1}={eigval[i]*e_scale:.4f}",
    )

for i in range(num_of_wave_plots):
    fig, ax = figs_axis[i]
    X = np.linspace(-L_opt / 2, L_opt / 2, reshaped_arrays_opt.shape[1]) * len_scale
    Y = np.linspace(-L_opt / 2, L_opt / 2, reshaped_arrays_opt.shape[2]) * len_scale
    X, Y = np.meshgrid(X, Y)
    contour = ax.contourf(X, Y, reshaped_arrays_opt[i, :, :], cmap="plasma", levels=100)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    fig.colorbar(contour, ax=ax)

    fig.suptitle(
        f"$\Psi(x, y)$ for energy = {eigval_opt[i]*e_scale:.4f} meV for N={N_opt} and L={L_opt:.4f} nm"
    )
    fig.savefig(f"{path}/Psi(xy)_for_E_{eigval_opt[i]*e_scale:.4f}_{i}.png")

ax_Ls.set_yscale("log")
ax_Ns.set_yscale("log")
ax_Ls.set_ylabel("Energy [meV]")
ax_Ns.set_ylabel("Energy [meV]")
ax_Ls.set_xlabel("L [nm]")
ax_Ns.set_xlabel("N")
ax_Ls.legend(loc="lower right")
ax_Ns.legend(loc="lower right")

fig_Ls.suptitle(f"E(L)")
fig_Ls.savefig(f"{path}/E(L).png")

fig_Ns.suptitle(f"E(N)")
fig_Ns.savefig(f"{path}/E(N).png")

output = proj(N=N_opt, L=L_opt, do_time_evo=True)
time_evo_arrays = output["time_evo_arrays"]
xt = output["xt"]
dt = output["dt"]

time = np.linspace(0, (xt.shape[0] - 1) * dt * t_scale, num=xt.shape[0])
fig_xt, ax_xt = plt.subplots(figsize=(19.2, 10.8))
ax_xt.plot(
    time,
    xt * len_scale,
)

for i in range(time_evo_arrays.shape[0]):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    X = np.linspace(-L_opt / 2, L_opt / 2, time_evo_arrays.shape[1])
    Y = np.linspace(-L_opt / 2, L_opt / 2, time_evo_arrays.shape[2])
    X, Y = np.meshgrid(X, Y)
    contour = ax.contourf(X, Y, time_evo_arrays[i, :, :], cmap="plasma", levels=100)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    fig.colorbar(contour, ax=ax)
    fig.suptitle(
        f"$\Psi (x,y,t)$ for N={N_opt} and L={L_opt:.4f} slowed down to {t_scale * dt * fps} fs per second"
    )
    fig.savefig(f"{temp_path}/frame_{i}.png")
    frames.append(f"{temp_path}/frame_{i}.png")
    plt.close(fig)
    if i % int(time_evo_arrays.shape[0] / time_evo_freq) == 0:
        fig1, ax1 = plt.subplots(figsize=(19.2, 10.8))
        contour = ax1.contourf(
            X, Y, time_evo_arrays[i, :, :], cmap="plasma", levels=100
        )
        fig1.colorbar(contour, ax=ax1)
        ax1.set_ylabel("y [nm]")
        ax1.set_xlabel("x [nm]")
        fig1.suptitle(
            f"$\Psi(x, y)$ for time = {i*dt*t_scale:.4f} for N={N_opt} and L={L_opt:.4f}"
        )
        fig1.savefig(f"{path}/Time_evo_t={i*dt*t_scale:.4f}fs.png")


ax_xt.set_ylabel("<x> [nm]")
ax_xt.set_xlabel("t [fs]")
fig_xt.suptitle(f"<x>(t) for N={N_opt} L={L_opt:.4f} dt={dt}")
fig_xt.savefig(f"{path}/x(t)_N={N_opt}_L={L_opt:.4f}_dt={dt}.png")
plt.show()

# Create GIF from saved frames
images = [imageio.imread(frame) for frame in frames]
imageio.mimsave(gif_path, images, fps=fps)

# Optionally, delete temporary frames after GIF creation
import os

for frame in frames:
    os.remove(frame)
