import numpy as np

import network

from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# Transform to normalized Tensors 
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('../data/MNIST/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('../data/MNIST/', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

train_dataset_array = next(iter(train_loader))[0].numpy().reshape(60000, 28 * 28, 1)
train_lables = next(iter(train_loader))[1].numpy()

test_dataset_array = next(iter(test_loader))[0].numpy().reshape(10000, 28 * 28, 1)
test_lables = next(iter(test_loader))[1].numpy()


vectorized_train_lables = []
for t in train_lables:
    v = np.zeros((10, 1))
    v[t][0] = 1
    vectorized_train_lables.append(v)


net = network.Network([784, 30, 10])
net.SGD(
    training_data=list(zip(train_dataset_array, vectorized_train_lables)),
    epochs=30,
    batch_size=10,
    learning_rate=3.0,
    test_data=list(zip(test_dataset_array, test_lables)))