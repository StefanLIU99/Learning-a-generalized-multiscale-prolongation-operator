import numpy as np
import os

file_path = os.path.expanduser('')
data = np.load(file_path)
train_data = data['train']
test_data = data['test']
train = []
test = []
for data in test_data:
    test.append(data)
for data in train_data:
    train.append(data)
temp_train = []
for data in train:
    x = np.flip(data, axis=1)
    temp_train.append(x)
    y = np.flip(data, axis=0)
    temp_train.append(y)
    z = np.transpose(data)
    temp_train.append(z)
    m = np.fliplr(np.rot90(data))
    temp_train.append(m)

