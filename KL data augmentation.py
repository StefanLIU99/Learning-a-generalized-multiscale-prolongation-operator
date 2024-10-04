import numpy as np
import math
import os

save_folder = os.path.expanduser('')
file_path = os.path.join(save_folder, '')
data = np.load(file_path)
train_data = data['train']
test_data = data['test']
random_array = np.random.randn(_, 25)


for i in [0.5, 0.1, 0.02, 0.01]:
    M = int(math.ceil(_ * i))
    train = []
    test = []
    for data in test_data:
        test.append(data)
    for data in train_data:
        train.append(data)
    train = train[:M]
    vectors = np.reshape(train, (M, -1))
    log_vectors = np.log(vectors)
    average = np.mean(log_vectors, axis = 0)
    cov_matrix = np.cov(log_vectors, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]  
    for N in [5, 15, 25]:
        A_N = random_array[:, :N]
        train = []
        test = []
        for data in test_data:
            test.append(data)
        for data in train_data:
            train.append(data)
        train = train[:M]
        top_N_eigenvalues = eigenvalues[sorted_indices[:N]]
        top_N_eigenvectors = eigenvectors[:, sorted_indices[:N]]
        sqrt_eigenvalues = np.sqrt(top_N_eigenvalues)
        transformed_vectors = np.dot(top_N_eigenvectors, np.diag(sqrt_eigenvalues))
        num_iterations = 9216 - M
        for s in range(num_iterations):
            logkappa_vector = np.dot(transformed_vectors, A_N[s])
            kappa_vector = np.exp(logkappa_vector)
            kappa = np.reshape(kappa_vector, (32, 32))
            train.append(kappa)
        test_data = test
        train_data = train
        save_folder = os.path.expanduser('')
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f'')
        np.savez(save_path, test=test_data, train=train_data)
