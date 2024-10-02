import numpy as np
from itertools import product
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve, LinearOperator, cg
import os 

PI = np.pi

class gmsfem_preconditioner:
    @staticmethod
    def get_mats(kappa: np.ndarray):
        size_y, size_x = kappa.shape
        h_y, h_x = 1.0 / size_y, 1.0 / size_x
        I, J, V = (
            np.zeros((size_y * size_x * 8,), dtype=np.int32),
            np.zeros((size_y * size_x * 8,), dtype=np.int32),
            np.zeros((size_y * size_x * 8,)),
        )
        marker = 0
        kappa_e = 0.0
        for y_ind, x_ind in product(range(size_y), range(size_x)):
            if x_ind >= 1:
                kappa_e = (
                    2.0
                    / (1.0 / kappa[y_ind, x_ind - 1] + 1.0 / kappa[y_ind, x_ind])
                    / h_x**2
                )
                I[marker] = y_ind * size_x + x_ind - 1
                J[marker] = y_ind * size_x + x_ind - 1
                V[marker] = kappa_e
                marker += 1
                I[marker] = y_ind * size_x + x_ind
                J[marker] = y_ind * size_x + x_ind
                V[marker] = kappa_e
                marker += 1
                I[marker] = y_ind * size_x + x_ind - 1
                J[marker] = y_ind * size_x + x_ind
                V[marker] = -kappa_e
                marker += 1
                I[marker] = y_ind * size_x + x_ind
                J[marker] = y_ind * size_x + x_ind - 1
                V[marker] = -kappa_e
                marker += 1
            if y_ind >= 1:
                kappa_e = (
                    2.0
                    / (1.0 / kappa[y_ind - 1, x_ind] + 1.0 / kappa[y_ind, x_ind])
                    / h_y**2
                )
                I[marker] = (y_ind - 1) * size_x + x_ind
                J[marker] = (y_ind - 1) * size_x + x_ind
                V[marker] = kappa_e
                marker += 1
                I[marker] = y_ind * size_x + x_ind
                J[marker] = y_ind * size_x + x_ind
                V[marker] = kappa_e
                marker += 1
                I[marker] = (y_ind - 1) * size_x + x_ind
                J[marker] = y_ind * size_x + x_ind
                V[marker] = -kappa_e
                marker += 1
                I[marker] = y_ind * size_x + x_ind
                J[marker] = (y_ind - 1) * size_x + x_ind
                V[marker] = -kappa_e
                marker += 1
        stiffness_mat = csr_matrix(
            (V[:marker], (I[:marker], J[:marker])),
            shape=(size_x * size_y, size_x * size_y),
        )
        return stiffness_mat

    @staticmethod
    def get_csr_submat(csr_mat, row_indx, col_indx):
        row_slice_mat = csr_mat[row_indx, :]
        row_slice_mat_csc = row_slice_mat.tocsc()
        sub_mat_csc = row_slice_mat_csc[:, col_indx]
        return sub_mat_csc.tocsr()

    def __init__(self, kappa: np.ndarray, partition: int) -> None:
        self.kappa = kappa
        self.size_y, self.size_x = self.kappa.shape[0], self.kappa.shape[1]
        self.h_y, self.h_x = 1.0 / self.size_y, 1.0 / self.size_x
        self.partition = partition
        self.elem_start_x = np.zeros((self.partition + 1,), dtype=int)
        self.elem_start_y = np.zeros((self.partition + 1,), dtype=int)
        for i in range(self.partition):
            self.elem_start_x[i + 1] = (
                self.elem_start_x[i] + self.size_x // self.partition
            )
            if i < self.size_x % self.partition:
                self.elem_start_x[i + 1] += 1
            self.elem_start_y[i + 1] = (
                self.elem_start_y[i] + self.size_y // self.partition
            )
            if i < self.size_y % self.partition:
                self.elem_start_y[i + 1] += 1
        self.A_mat = self.get_mats(self.kappa)

        # Create local2global_ind_list.
        # local2global_ind_list[coarse_y][coarse_x][i] is the global index of the i-th fine element in the (coarse_y, coarse_x) coarse element.
        # Should be useful in creating the prolongation matrix.
        # Check https://stackoverflow.com/questions/2397141/how-to-initialize-a-two-dimensional-array-list-of-lists-if-not-using-numpy-in
        # Do not write something like below.
        # self.local2global_ind_list = [[None] * self.partition] * self.partition

        self.local2global_ind_list = [
            [None for _ in range(self.partition)] for _ in range(self.partition)
        ]
        for coarse_y, coarse_x in product(range(self.partition), range(self.partition)):
            entries_num = (
                self.elem_start_y[coarse_y + 1] - self.elem_start_y[coarse_y]
            ) * (self.elem_start_x[coarse_x + 1] - self.elem_start_x[coarse_x])
            self.local2global_ind_list[coarse_y][coarse_x] = np.zeros(
                (entries_num,), dtype=int
            )
            for fine_y, fine_x in product(
                range(self.elem_start_y[coarse_y], self.elem_start_y[coarse_y + 1]),
                range(self.elem_start_x[coarse_x], self.elem_start_x[coarse_x + 1]),
            ):
                (self.local2global_ind_list[coarse_y][coarse_x])[
                    (fine_y - self.elem_start_y[coarse_y])
                    * (self.elem_start_x[coarse_x + 1] - self.elem_start_x[coarse_x])
                    + (fine_x - self.elem_start_x[coarse_x])
                ] = (fine_y * self.size_x + fine_x)

        # Create the block-jacobi lists.

        self.block_jacobi_list = [
            [None for _ in range(self.partition)] for _ in range(self.partition)
        ]
        for coarse_y, coarse_x in product(range(self.partition), range(self.partition)):
            self.block_jacobi_list[coarse_y][coarse_x] = self.get_csr_submat(
                self.A_mat,
                self.local2global_ind_list[coarse_y][coarse_x],
                self.local2global_ind_list[coarse_y][coarse_x],
            )

        # If this is None, fall back to the one-level preconditioner.

        self.R_mat = None
        self.Ac_mat = None

    def gmsfem_coarse_space(self, eigen_num: int):

        # Create the coarse space list.
        # coarse_space_list[coarse_y][coarse_x][i] is the i-th eigenfunction of the (coarse_y, coarse_x) coarse element.

        coarse_space_list = [
            [[None for _ in range(eigen_num)] for _ in range(self.partition)]
            for _ in range(self.partition)
        ]
        for coarse_y, coarse_x in product(range(self.partition), range(self.partition)):
            sub_kappa = self.kappa[
                self.elem_start_y[coarse_y] : self.elem_start_y[coarse_y + 1],
                self.elem_start_x[coarse_x] : self.elem_start_x[coarse_x + 1],
            ]
            sub_A_mat = self.get_mats(sub_kappa) / (self.partition**2)

            # Scaling the S matrix such that eigenvalues are 0, 1, 1, 2, 4, 4,..., if kappa is constant and H_x = H_y.

            sub_S_mat = diags(sub_kappa.flatten()).tocsr()
            # sub_S_mat = diags(sub_kappa.flatten() * PI **2).tocsr()
            eigen_vals, eigen_vecs = eigsh(sub_A_mat, k=eigen_num, M=sub_S_mat, sigma=-1.0)
            # print ("Eigenvalues:", eigen_vals)
            # print ("Eigenvectors:", eigen_vecs)
            for eigen_ind in range(eigen_num):
                coarse_space_list[coarse_y][coarse_x][eigen_ind] = eigen_vecs[
                    :, eigen_ind
                ]

            # print(
            #     f"Coarse space ({coarse_y}, {coarse_x}) done with eigen_vals={eigen_vals}."
            # )

        # A fancy way to create the prolongation matrix.
        # The shape of the prolongation matrix is (partition**2 * eigen_num, size_y * size_x).

        max_data_len = self.size_y * self.size_x * eigen_num
        R_data = np.zeros((max_data_len,))
        R_indices = np.zeros((max_data_len,), dtype=int)
        R_indptr = np.zeros((self.partition**2 * eigen_num + 1,), dtype=int)
        marker = 0
        for coarse_y, coarse_x in product(range(self.partition), range(self.partition)):
            for eigen_ind in range(eigen_num):

                # One row in the prolongation matrix.

                row_ind = (coarse_y * self.partition + coarse_x) * eigen_num + eigen_ind
                nnz = (
                    self.elem_start_y[coarse_y + 1] - self.elem_start_y[coarse_y]
                ) * (self.elem_start_x[coarse_x + 1] - self.elem_start_x[coarse_x])
                R_indptr[row_ind + 1] = nnz
                R_indices[marker : marker + nnz] = self.local2global_ind_list[coarse_y][
                    coarse_x
                ]
                R_data[marker : marker + nnz] = coarse_space_list[coarse_y][coarse_x][
                    eigen_ind
                ]
                marker += nnz
        R_indptr = np.cumsum(R_indptr)
        self.R_mat = csr_matrix(
            (R_data, R_indices, R_indptr),
            shape=(self.partition**2 * eigen_num, self.size_y * self.size_x),
        )
        self.Ac_mat = self.R_mat @ self.A_mat @ self.R_mat.T

    def apply_smoother(self, r: np.ndarray):
        u = np.zeros_like(r)
        for coarse_y, coarse_x in product(range(self.partition), range(self.partition)):
            r_on_coarse = r[self.local2global_ind_list[coarse_y][coarse_x]]
            u_on_coarse = spsolve(
                self.block_jacobi_list[coarse_y][coarse_x], r_on_coarse
            )
            u[self.local2global_ind_list[coarse_y][coarse_x]] = u_on_coarse
        return u


    def apply_preconditioner(self, rhs: np.ndarray):
        r = np.zeros_like(rhs)
        # For copy.
        r[:] = rhs[:]
        u = self.apply_smoother(r)
        if self.R_mat is not None:
            r = rhs - self.A_mat @ u
            r_coarse = self.R_mat @ r
            u_coarse = spsolve(self.Ac_mat, r_coarse)
            # u_coarse = cg(self.Ac_mat, r_coarse)[0]
            u += self.R_mat.T @ u_coarse
        r = rhs - self.A_mat @ u
        u += self.apply_smoother(r)
        return u

    def get_preconditioner_LO(self):
        return LinearOperator(self.A_mat.shape, matvec=self.apply_preconditioner)

def load_kappa_data(file_path):
    data = np.load(file_path) 
    first_key = data.files[0]
    kappas = data[first_key]
    if kappas.ndim != 3 or kappas.shape[1:] != (512, 512):
        raise ValueError(f"wrong: {kappas.shape}")
    return np.flip(kappas[0], axis=0)

N = 512
file_path = os.path.expanduser('~/Desktop/Data_set_64x/dataset(whole kappa).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 10000x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 1000x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 100x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 10x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 01x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 001x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 0001x).npz')
# file_path = os.path.expanduser('~/Desktop/Data set(Random Ball)/dataset(whole kappa 00001x).npz')

# kappa = np.ones((N, N))

kappa = load_kappa_data(file_path)

precond_ctx = gmsfem_preconditioner(kappa, partition=16)
precond_ctx.gmsfem_coarse_space(eigen_num=5)

# Construct the preconditioner.

# M = LinearOperator(precond_ctx.A_mat.shape, matvec=precond_ctx.apply_preconditioner)



# RHS vector.

b_vec = np.zeros((N, N))
b_vec[0, 0] = 1
b_vec[-1, -1] = 1
b_vec[0, -1] = -1
b_vec[-1, 0] = -1
b_vec = b_vec.flatten()

iter = 0
# M = precond_ctx.get_preconditioner_LO()
# x, info = cg(precond_ctx.A_mat, b_vec, maxiter=1000, callback=precond_ctx.iter_callback )

residuals = []
iter_count = 0

def iter_callback(Xi):
    global iter_count
    residual = np.linalg.norm(b_vec - precond_ctx.A_mat @ Xi)
    residuals.append(residual)
    print(f"At iterate {iter_count}, residual: {residual}")
    iter_count += 1

x, info = cg(
    precond_ctx.A_mat,
    b_vec,
    maxiter=100,
    M=precond_ctx.get_preconditioner_LO(),
    callback=iter_callback,
)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
filesave_path = os.path.join(desktop_path, ".csv")
np.savetxt(filesave_path, residuals, delimiter=",")

print(f"info: {info}")



