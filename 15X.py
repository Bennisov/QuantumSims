import numpy as np
import numpy
import matplotlib.pyplot as plt
import math
import scipy as scp

len_scale = 0.05292
e_scale = 27211.6


def proj(L,N):
    len_scale = 0.05292
    w = 10.0 / 27211.6  # meV
    e_scale = 27211.6
    m = 0.067
    h = 1
    n_ksi = 10
    L /= len_scale
    a = L / (2 * N)
    d_ksi = a / n_ksi
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
                print(g(j, p_ind(l), p_ind(n)),  g(i, p_ind(l), p_ind(n)))
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
                        * (x ** 2 + y ** 2)
                )
        return a ** 2 * m * w ** 2 / 8 * v_ji


    def merge_mat(S, T, V):
        h_gl = numpy.zeros(((2 * N + 1) ** 2, (2 * N + 1) ** 2))
        s_gl = numpy.zeros(((2 * N + 1) ** 2, (2 * N + 1) ** 2))
        for k in range(elements.shape[0]):
            for i1 in range(4):
                for i2 in range(4):
                    s_gl[elements[k, i1]-1, elements[k, i2]-1] += S[i1, i2]
                    h_gl[elements[k, i1]-1, elements[k, i2]-1] += T[i1, i2] + V[k, i1, i2]
        for i in range(s_gl.shape[0]):
            var1 = int(i/(2*N+1))
            var2 = i%(2*N+1)
            if (var1 * var2 == 0) or (var1 == 2 * N) or (var2 == 2 * N):
                s_gl[i, :] = 0.
                h_gl[i, :] = 0.
                s_gl[:, i] = 0.
                h_gl[:, i] = 0.
                s_gl[i, i] = 1.0
                h_gl[i, i] = -1410.0
        return h_gl, s_gl


    def z2():
        psi_node = numpy.zeros((2 * N + 1, 2 * N + 1))
        for i in range(psi_node.shape[0]):
            for j in range(psi_node.shape[1]):
                ind = ind_tran(i, j)
                psi_node[i, j] = math.exp(
                    -m * w / 2 * (nodes[ind, 0] ** 2 + nodes[ind, 1] ** 2)
                )
        psi_element = numpy.zeros((elements.shape[0], n_ksi, n_ksi))
        for k in range(elements.shape[0]):
            ksixs = np.linspace(-1.0, 1.0, n_ksi)
            ksiys = np.linspace(-1.0, 1.0, n_ksi)
            for i in range(len(ksixs)):
                for j in range(len(ksiys)):
                    ksix = ksixs[i]
                    ksiy = ksiys[j]
                    # x = (1 - ksix) / 2 * elements[k, 1] + (1 + ksix) / 2 * elements[k, 2]
                    # y = (1 - ksiy) / 2 * elements[k, 1] + (1 + ksiy) / 2 * elements[k, 3]
                    for l in range(4):
                        ind = elements[k, l] - 1
                        indi, indj = inv_tran(ind)
                        psi_element[k, i, j] += psi_node[indi, indj] * g(l, ksix, ksiy)
        reshaped_array = np.zeros(((2 * N) * n_ksi, (2 * N) * n_ksi))
        for idx in range(elements.shape[0]):
            col_start = (idx // (2 * N)) * n_ksi
            row_start = (idx % (2 * N)) * n_ksi
            reshaped_array[row_start: row_start + n_ksi, col_start: col_start + n_ksi] = (
                psi_element[idx]
            )
        return reshaped_array

    def z3(elements):
        S = np.zeros((4, 4))
        T = np.zeros((4, 4))
        V = np.zeros((4 * N ** 2, 4, 4))
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                S[i, j] = calculate_S_matrix(i, j)
                T[i, j] = calculate_T_matrix(i, j)
                for k in range(elements.shape[0]):
                    V[k, i, j] = calculate_V_matrix(k, i, j)
        H_glo, S_glo = merge_mat(S, T, V)
        eigval, eigvec = scp.linalg.eigh(H_glo, S_glo)
        indx = np.where(eigval < 0.)
        eigval = np.delete(eigval, indx)
        eigvec = np.delete(eigvec, indx, axis=0)
        return eigval, eigvec, H_glo, S_glo, S, T, V
    nodes, elements = generate_mesh(L, N)
    reshaped_array = z2()
    eigval, eigvec, H_glo, S_glo, S, T, V = z3(elements)
    return eigval


Ls = numpy.linspace(10, 500, num=500)
energies = numpy.zeros((500, 9)) #9 do poprawy
for i in range(500):
    L = Ls[i]
    eigval = proj(L=L, N=2)
    energies[i, :] = eigval
sum=0.
for i in range(9):
    sum += numpy.argmin(energies[:, i])
sum /= 9
print(sum)
plt.plot(Ls, energies*e_scale)
plt.yscale('log')
plt.show()
Ns = numpy.array([2, 3, 4, 5, 6, 7, 8])
energies2 = numpy.zeros((7, 9))
for i in range(7):
    N = Ns[i]
    eigval = proj(L=sum, N=N)
    energies2[i, :] = eigval[:9]
plt.plot(Ns, energies2*e_scale)
plt.yscale('log')
plt.show()
L = sum
N = 5




# for i in range(elements.shape[0]):
#     for j in range(elements.shape[1]):
#         element_num = i + 1
#         node_num = j + 1
#         global_node = elements[i, j]
#         node_x = nodes[global_node - 1][0]
#         node_y = nodes[global_node - 1][1]
#
#         print(
#             f"Element {element_num}, Local_Node {node_num}: Global_Node {global_node}, "
#             f"Coordinates: ({node_x}, {node_y})"
#         )

# X = np.linspace(-L / 2, L / 2, reshaped_array.shape[0]) * len_scale
# Y = np.linspace(-L / 2, L / 2, reshaped_array.shape[1]) * len_scale
# X, Y = np.meshgrid(X, Y)
# plt.figure(dpi=100)
# plt.contourf(X, Y, reshaped_array, cmap="plasma", levels=int(L * len_scale))
# plt.colorbar()
# plt.show()

# zadanie 3, 4, 4a


# print("S:")
# print(" a^2 / 36 * [")
# for i in range(S.shape[0]):
#     print("[ ", end=" ")
#     for j in range(S.shape[1]):
#         print(f"{(S[i, j] * 9 * 4 / (a ** 2)):.3f}", end=" ")
#     print("]")
# print("]")
#
# print("T:")
# print(" h^2/12m [")
# for i in range(T.shape[0]):
#     print("[ ", end=" ")
#     for j in range(T.shape[1]):
#         print(f"{(T[i, j] * 2 * 6 * m / (h ** 2)):.2f}", end=" ")
#     print("]")
# print("]")
#
# print("V^11: ")
# print("[")
# for i in range(V.shape[1]):
#     print("[ ", end=" ")
#     for j in range(V.shape[2]):
#         print(f"{(V[10, i, j]):.2f}", end=" ")
#     print("]")
# print("]")
#
#
# print("H: ")
# print("[")
# for i in range(H_glo.shape[0]):
#     print("[ ", end=" ")
#     for j in range(H_glo.shape[1]):
#         print(f"{(H_glo[i, j]):.4f}", end=" ")
#     print("]")
# print("]")
# print("S: ")
# print("[")
# for i in range(S_glo.shape[0]):
#     print("[ ", end=" ")
#     for j in range(S_glo.shape[1]):
#         print(f"{(S_glo[i, j]):.4f}", end=" ")
#     print("]")
# print("]")


