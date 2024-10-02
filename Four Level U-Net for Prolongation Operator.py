import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time


# Unet model
class Unet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Unet, self).__init__()

        # Encoder 
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv13 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        # Classifier
        self.conv15 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x1 = self.bn1(x)
        x = self.pool1(x1)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x2 = self.bn2(x)
        x = self.pool2(x2)

        x = torch.relu(self.conv5(x2))
        x = torch.relu(self.conv6(x))
        x3 = self.bn3(x)
        x = self.pool3(x3)

        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x))
        x = self.bn4(x)

        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.conv9(torch.cat([x, x3], dim=1)))
        x = torch.relu(self.conv10(x))
        x = self.bn5(x)

        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.conv11(torch.cat([x, x2], dim=1)))
        x = torch.relu(self.conv12(x))
        x = self.bn6(x)

        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.conv13(torch.cat([x, x1], dim=1)))
        x = torch.relu(self.conv14(x))
        x = self.bn7(x)

        x = self.conv15(x)

        return x
        


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, coefficients, predictions, targets):
        predictions = predictions.view(batch_size, num_classes, -1)
        sum_loss = 0
        for i in range(batch_size):
            orthonormal_vectors_i = []
            v_batch = predictions[i]
            A = torch.diag(h**2 * coefficients[i].flatten())
            for j in range(num_classes):
                v = v_batch[j].clone()
                for e in orthonormal_vectors_i:
                    v = v - torch.dot(v, torch.matmul(A, e)) * e
                if torch.dot(v, torch.matmul(A, v)) > 0:
                    k = torch.sqrt(torch.dot(v, torch.matmul(A, v)))
                    v = v / k
                    orthonormal_vectors_i.append(v)
                else:
                    orthonormal_vectors_i.append(v)
            C_i = torch.stack(orthonormal_vectors_i)
            sum_loss = sum_loss + (4 - torch.sum((torch.matmul(torch.matmul(C_i, A), targets[i])) * (torch.matmul(torch.matmul(C_i, A), targets[i])))) / 4
        loss = sum_loss / batch_size
        return loss

custom_loss = CustomLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_paths = [
   
]

label_paths = [
   
]

for file_path, label_path in zip(file_paths, label_paths):
    start_time = time.time()
    data = np.load(file_path)

    train_data = data['train']
    test_data = data['test']

    label_data = np.load(label_path)

    train_label = label_data['train']
    test_label = label_data['test']

    class UnetDataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = torch.from_numpy(data).unsqueeze(1).float()
            self.labels = torch.from_numpy(labels).float()
            self.transform = transform

        def __getitem__(self, index):
            image = self.data[index]
            label = self.labels[index]

            if self.transform:
                image = self.transform(image)

            return image, label

        def __len__(self):
            return len(self.data)

    train_dataset = UnetDataset(train_data, train_label)
    trainloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_dataset = UnetDataset(test_data, test_label)
    testloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    model = Unet(input_channels = 1, num_classes = 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = custom_loss

    N = 32
    h = 1.0 / float(N)
    batch_size = 20
    num_epochs = 30
    num_classes = 4
    train_epoch = []
    train_step = []
    train_losses = [] 
    test_epoch = [] 
    test_losses = []  
    t = []
    
    model.eval()
    running_loss = 0.0
    num = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(inputs, outputs, labels)
            running_loss += loss.item()
            num += 1

    accuracy = 1 - running_loss / num
    print(f'Epoch [{0}/{num_epochs}], Test Loss: {1 - accuracy:.4f}')
    test_epoch.append(0)  
    test_losses.append(f"{running_loss / num:.4f}")  

    for epoch in range(num_epochs):
        running_loss = 0.0
    
        model.train()  
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(inputs, outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 110 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/110:.4f}')
                train_epoch.append(epoch)
                train_step.append(i+1)
                train_losses.append(f"{running_loss/110:.4f}")
                running_loss = 0.0
    
        model.eval()  
        running_loss = 0.0
        num = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(inputs, outputs, labels)
                running_loss += loss.item()
                num += 1
        accuracy = 1 - running_loss / num
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {1 - accuracy:.4f}')
        test_epoch.append(epoch + 1)
        test_losses.append(f"{running_loss / num:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minutes = round(elapsed_time / 60, 2) 
    t.append(elapsed_time_minutes)


    # save model and loss data
    base_name = os.path.basename(file_path)
    save_file_name = base_name.replace('dataset', 'models')
    save_dir = os.path.join(os.path.expanduser("~"), "Desktop", save_file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    losses_path = os.path.join(save_dir, "losses")
    np.savez(losses_path, train_epoch=train_epoch, train_step=train_step, train_losses=train_losses, test_epoch=test_epoch, test_losses=test_losses, time=t)