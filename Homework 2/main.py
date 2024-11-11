import numpy as np
from torchvision.datasets import MNIST

# Funcție pentru descărcarea datelor MNIST
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True, train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

# Descărcarea datelor de antrenament și testare
train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# Normalizarea datelor
train_X = train_X / 255.0
test_X = test_X / 255.0

# Funcția softmax pentru calcularea probabilităților de clasă
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # Scăderea valorii maxime pentru stabilitate numerică
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Funcție pentru codificarea one-hot a etichetelor
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Parametrii modelului
input_size = 784
output_size = 10
learning_rate = 0.01
epochs = 100  #este în intervalul 50-500
batch_size = 50

# Setarea seed-ului pentru reproducibilitate
np.random.seed(41)

# Inițializarea greutăților și bias-urilor
W = np.random.randn(input_size, output_size)
b = np.random.randn(output_size)

# Codificarea one-hot a etichetelor de antrenament
train_Y_oh = one_hot_encode(train_Y)

# Calcularea acurateței inițiale pe setul de testare
z_test_initial = np.dot(test_X, W) + b
test_predictions_initial = softmax(z_test_initial)
correct_test_predictions_initial = np.argmax(test_predictions_initial, axis=1) == test_Y
test_accuracy_initial = np.mean(correct_test_predictions_initial)
print(f"Initial Test Accuracy = {test_accuracy_initial * 100:.2f}%")

# Antrenament
for epoch in range(epochs):
    # Amestecarea datelor
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_Y_oh = train_Y_oh[indices]

    epoch_loss = 0

    for start in range(0, train_X.shape[0], batch_size):
        end = start + batch_size
        batch_X = train_X[start:end]
        batch_Y_oh = train_Y_oh[start:end]

        # Propagare înainte
        z = np.dot(batch_X, W) + b#calcularea produsului scalar dintre datele de intrare și greutăți, urmată de adăugarea bias-ului
        predictions = softmax(z)

        # Calcularea erorii
        error = batch_Y_oh - predictions

        # Calcularea pierderii (entropie încrucișată)
        batch_loss = -np.sum(batch_Y_oh * np.log(predictions + 1e-8)) / batch_size
        epoch_loss += batch_loss

        # Înapoi-propagare
        dW = np.dot(batch_X.T, error)
        db = np.sum(error, axis=0)

        # Actualizarea greutăților și bias-urilor
        W += learning_rate * dW
        b += learning_rate * db

    epoch_loss /= (train_X.shape[0] / batch_size)

    # Calcularea acurateței de antrenament
    if epoch % 10 == 0 or epoch == epochs - 1:
        z_train = np.dot(train_X, W) + b
        train_predictions = softmax(z_train)
        correct_train_predictions = np.argmax(train_predictions, axis=1) == train_Y
        train_accuracy = np.mean(correct_train_predictions)
        print(f"Epoch {epoch}: Training Accuracy = {train_accuracy * 100:.2f}%, Loss = {epoch_loss:.4f}")

# Calcularea acurateței pe setul de testare după antrenament
z_test = np.dot(test_X, W) + b
test_predictions = softmax(z_test)
correct_test_predictions = np.argmax(test_predictions, axis=1) == test_Y
test_accuracy = np.mean(correct_test_predictions)
print(f"Test Accuracy = {test_accuracy * 100:.2f}%")