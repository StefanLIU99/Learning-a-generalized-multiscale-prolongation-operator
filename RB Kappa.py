import numpy as np
import os

dim = 512
num_matrices = 200
sub_dim = 32

A = np.zeros((dim * dim, dim * dim), dtype=float)
for i in range(dim * dim):
    y_i = i // dim
    x_i = i % dim
    for j in range(dim * dim):
        y_j = j // dim
        x_j = j % dim
        A[i, j] = 2 * np.exp(-np.sqrt(((x_i - x_j) / dim)**2 + ((y_i - y_j) / dim)**2))

mean = np.zeros(dim * dim)
kappas = []
sub_kappas = []
for k in range(num_matrices):
    sample = np.random.multivariate_normal(mean, A, 1)
    kappa = np.reshape(sample, (dim, dim))
    kappas.append(kappa)
    for m in range(0, dim, sub_dim):
        for n in range(0, dim, sub_dim):
            sub_kappa = kappa[m:m + sub_dim, n:n + sub_dim]
            sub_kappas.append(sub_kappa)

test = sub_kappas[:5000]
train = sub_kappas[5000:]
