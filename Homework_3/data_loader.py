import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten() / 255.0,  # Normalize the data
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


# Convert labels to one-hot encoding
def to_categorical(y, num_classes):
    y = np.array(y, dtype='int')
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_Y = to_categorical(train_Y, 10)
test_Y = to_categorical(test_Y, 10)