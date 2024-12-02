import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        # input_size - dimensiunea intrării
        # hidden_size - dimensiunea stratului ascuns
        # output_size - dimensiunea stratului de ieșire
        # dropout_rate - rata de dropout pentru regularizare
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.dropout_rate = dropout_rate
        # W1 conectează datele de intrare la stratul ascuns
        # W2 conectează stratul ascuns la stratul de ieșire, unde sunt generate predicțiile finale ale rețelei.
        # b1 este un vector de biasuri pentru fiecare neuron din stratul ascuns
        # b2 este un vector de biasuri pentru fiecare neuron din stratul de ieșire

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X, training=True):
        # Se calculează activările nete (Z1) aplicând greutățile și biasurile.
        # Activările sunt trecute prin funcția sigmoid pentru a introduce non-linearitate:
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        #În timpul antrenării, se dezactivează aleatoriu un procent din neuroni pentru a preveni supraînvățarea:
        if training:
            self.dropout_mask = np.random.rand(*self.A1.shape) > self.dropout_rate  # Mască pentru dropout
            self.A1 *= self.dropout_mask                       # Aplică masca dropout
            self.A1 /= (1 - self.dropout_rate)
        # Activările stratului ascuns sunt folosite pentru a calcula ieșirea finală (probabilități pentru fiecare clasă) folosind softmax:
        self.Z2 = np.dot(self.A1, self.W2) + self.b2           # Activările Z2 sunt trecute prin funcția softmax
        self.A2 = self.softmax(self.Z2)                        # Softmax pentru probabilitățile de ieșire
        return self.A2                                         # Returnează probabilitățile finale

    def backward(self, X, Y, learning_rate):
        #Pentru a ajusta greutățile și biasurile, se calculează derivata (gradientul) funcției de cost față de fiecare parametru.
        m = X.shape[0]
        #dZ2 măsoară diferența dintre predicțiile rețelei (A2) și valorile reale (Y).
        # Este gradientul funcției de cost față de activările stratului final.
        dZ2 = self.A2 - Y

        #dW2: Spune cum trebuie ajustate greutățile care leagă stratul ascuns de stratul de ieșire
        dW2 = np.dot(self.A1.T, dZ2) / m
        # db2: Spune cum trebuie ajustate biasurile stratului final.
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        # dA1 este eroarea propagată înapoi de la stratul de ieșire spre stratul ascuns,
        # obținută prin înmulțirea gradientului de la stratul final cu greutățile stratului de ieșire.
        dA1 = np.dot(dZ2, self.W2.T)
        # e aplicata masca pentru consistenta
        dA1 *= self.dropout_mask
        # dZ1: Gradientul stratului ascuns, folosind derivata sigmoidului.
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        # Gradientele funcției de cost față de greutățile și biasurile stratului ascuns.
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Actualizarea greutăților și biasurilor
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X_train, Y_train, X_val, Y_val, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                Y_batch = Y_train_shuffled[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, Y_batch, learning_rate)

            train_accuracy = self.evaluate(X_train, Y_train)
            val_accuracy = self.evaluate(X_val, Y_val)
            print('Epoch {}, Train Accuracy: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1, train_accuracy, val_accuracy))

            # Reduce rata de învățare dacă acuratețea de validare scade
            if epoch > 0 and val_accuracy <= prev_val_accuracy:
                learning_rate *= 0.9
            prev_val_accuracy = val_accuracy

        # Afișează precizia finală pe setul de validare
        final_val_accuracy = self.evaluate(X_val, Y_val) * 100
        print('Final Validation Accuracy: {:.2f}%'.format(final_val_accuracy))

    # Evaluează precizia modelului pe un set de date dat
    def evaluate(self, X, Y):
        A2 = self.forward(X, training=False)                   # Propagare înainte fără dropout
        predictions = np.argmax(A2, axis=1)                    # Preziceri prin alegerea clasei cu probabilitate maximă
        labels = np.argmax(Y, axis=1)                          # Clasele adevărate
        accuracy = np.mean(predictions == labels)              # Precizia ca media corectitudinilor
        return accuracy                                        # Returnează precizia

