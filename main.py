import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisation des poids
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.hidden_bias = np.random.randn(hidden_size)
        self.output_bias = np.random.randn(output_size)

    def forward(self, x):
        # Passage avant à travers le réseau
        self.hidden = sigmoid(np.dot(x, self.weights_input_hidden) + self.hidden_bias)
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.output_bias)
        return output

    def train(self, x, y, epochs, lr):
        for epoch in range(epochs):
            # Passage avant
            output = self.forward(x)

            # Calcul des erreurs
            error = y - output
            d_output = error * sigmoid_derivative(output)

            # Erreur pour la couche cachée
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * sigmoid_derivative(self.hidden)

            # Mise à jour des poids et des biais
            self.weights_hidden_output += self.hidden.T.dot(d_output) * lr
            self.output_bias += np.sum(d_output, axis=0, keepdims=True).reshape(self.output_bias.shape) * lr
            self.weights_input_hidden += x.T.dot(d_hidden) * lr
            self.hidden_bias += np.sum(d_hidden, axis=0, keepdims=True).reshape(self.hidden_bias.shape) * lr

            if epoch % 1000 == 0:
                loss = np.mean(np.abs(error))
                print(f'Epoch {epoch}, Loss: {loss}')


# Exemple d'utilisation
if __name__ == "__main__":
    # Données d'entraînement (x) et étiquettes (y)
    x = np.array([[0,0],[0,1],[1,0],[1,1]])  # Exemple pour XOR
    y = np.array([[0],[1],[1],[0]])

    # Création et entraînement du modèle
    mlp = SimpleMLP(input_size=2, hidden_size=4, output_size=1)
    mlp.train(x, y, epochs=10000, lr=0.1)

    # Test du modèle entraîné
    for i in range(4):
        print(f"Input: {x[i]}, Predicted: {mlp.forward(x[i])}")
