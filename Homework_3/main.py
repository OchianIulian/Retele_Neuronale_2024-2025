from mlp import MLP
from data_loader import download_mnist, to_categorical

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# Convert labels to one-hot encoding
train_Y = to_categorical(train_Y, 10)
test_Y = to_categorical(test_Y, 10)

mlp = MLP(input_size=784, hidden_size=100, output_size=10, dropout_rate=0.1)
mlp.train(train_X, train_Y, test_X, test_Y, epochs=30, batch_size=20, learning_rate=0.1)