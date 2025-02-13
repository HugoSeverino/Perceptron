import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement et prétraitement des données
iris = load_iris()
X = iris.data
y = iris.target

# Sélection des deux premières classes (binary classification)
X = X[y != 2]
y = y[y != 2]

# Normalisation et PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# --- Définition des classes ---
class Perceptron:
    def __init__(self, input_size, activation_type, learning_rate=0.01):
        self.activation_type = activation_type
        self.lr = learning_rate
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.random.randn()
    
    def activation(self, x):
        if self.activation_type == 'step':
            return np.where(x >= 0, 1, 0)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
    
    def derivative(self, x):
        if self.activation_type == 'sigmoid':
            sig = self.activation(x)
            return sig * (1 - sig)
        elif self.activation_type == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_type == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            return np.ones_like(x)
    
    def forward(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.activation(linear).reshape(-1, 1), linear.reshape(-1, 1)

    def update(self, X, error):
        grad_w = np.dot(X.T, error)
        grad_b = np.sum(error)

        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b
    
    def fit(self, X_train, y_train, X_test, y_test, epochs=100):
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        train_acc = []
        test_acc = []

        for epoch in range(epochs):
            output, _ = self.forward(X_train)
            error = output - y_train
            self.update(X_train, error)

            train_acc.append(self.accuracy(X_train, y_train))
            test_acc.append(self.accuracy(X_test, y_test))

        return train_acc, test_acc

    def predict(self, X):
        output, _ = self.forward(X)
        return np.where(output >= 0.5, 1, 0)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y.reshape(-1, 1))

# Classe pour deux perceptrons en série
class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, activation1, activation2, learning_rate=0.01, epochs=100):
        self.perceptron1 = Perceptron(input_size, activation1, learning_rate)
        self.perceptron2 = Perceptron(hidden_size, activation2, learning_rate)
        self.epochs = epochs
        self.train_acc = []
        self.test_acc = []
    
    def fit(self, X_train, y_train, X_test, y_test):
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        for epoch in range(self.epochs):
            hidden_output, _ = self.perceptron1.forward(X_train)
            output, _ = self.perceptron2.forward(hidden_output)

            error = output - y_train
            delta_output = error * self.perceptron2.derivative(output)

            delta_hidden = (delta_output @ self.perceptron2.weights.T) * self.perceptron1.derivative(hidden_output)

            self.perceptron2.update(hidden_output, delta_output)
            self.perceptron1.update(X_train, delta_hidden)

            self.train_acc.append(self.accuracy(X_train, y_train))
            self.test_acc.append(self.accuracy(X_test, y_test))
    
    def predict(self, X):
        hidden_output, _ = self.perceptron1.forward(X)
        output, _ = self.perceptron2.forward(hidden_output)
        return np.where(output >= 0.5, 1, 0)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y.reshape(-1, 1))

# Fonction pour tracer les frontières
def plot_decision_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    ax.set_title(title)

# --- Entraînement et affichage ---
activations = ['step', 'sigmoid', 'tanh', 'relu']

fig1, axs1 = plt.subplots(2, 4, figsize=(20, 10))
fig2, axs2 = plt.subplots(2, 4, figsize=(20, 10))

for i, activation in enumerate(activations):
    # 🔵 Perceptron simple
    perceptron = Perceptron(input_size=2, activation_type=activation, learning_rate=0.01)
    train_acc, test_acc = perceptron.fit(X_train, y_train, X_test, y_test, epochs=100)
    
    plot_decision_boundary(perceptron, X_pca, y, f'Perceptron Simple ({activation})', axs1[0, i])
    axs1[1, i].plot(train_acc, label='Train Accuracy')
    axs1[1, i].plot(test_acc, linestyle='--', label='Test Accuracy')
    axs1[1, i].set_title(f'Accuracy Perceptron Simple ({activation})')
    axs1[1, i].legend()

    # 🔴 Perceptron en série
    model = TwoLayerPerceptron(input_size=2, hidden_size=1, activation1=activation, activation2='sigmoid', learning_rate=0.01, epochs=100)
    model.fit(X_train, y_train, X_test, y_test)

    plot_decision_boundary(model, X_pca, y, f'Perceptron en Série ({activation})', axs2[0, i])
    axs2[1, i].plot(model.train_acc, label='Train Accuracy')
    axs2[1, i].plot(model.test_acc, linestyle='--', label='Test Accuracy')
    axs2[1, i].set_title(f'Accuracy Perceptron en Série ({activation})')
    axs2[1, i].legend()

plt.show()
