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

# Classe Perceptron
class Perceptron:
    def __init__(self, activation_type, learning_rate=0.01, epochs=100):
        self.activation_type = activation_type
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_acc = []
        self.test_acc = []
    
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
            return 1
    
    def fit(self, X_train, y_train, X_test, y_test):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            # Mise à jour différente pour step
            if self.activation_type == 'step':
                for i in range(n_samples):
                    linear = np.dot(X_train[i], self.weights) + self.bias
                    pred = self.activation(linear)
                    error = y_train[i] - pred
                    update = self.lr * error
                    self.weights += update * X_train[i]
                    self.bias += update
            else:
                # Forward pass
                linear = np.dot(X_train, self.weights) + self.bias
                activated = self.activation(linear)
                
                # Calcul de l'erreur
                error = activated - y_train
                
                # Backpropagation
                grad_w = (1/n_samples) * np.dot(X_train.T, error * self.derivative(linear))
                grad_b = (1/n_samples) * np.sum(error * self.derivative(linear))
                
                # Mise à jour
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b
            
            # Calcul de la précision
            self.train_acc.append(self.accuracy(X_train, y_train))
            self.test_acc.append(self.accuracy(X_test, y_test))
    
    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.activation(linear)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        if self.activation_type != 'step':
            predictions = np.where(predictions >= 0.5, 1, 0)
        return np.mean(predictions == y)

# Fonction pour tracer les frontières
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if model.activation_type != 'step':
        Z = np.where(Z >= 0.5, 1, 0)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')

# Entraînement et visualisation pour chaque activation
activations = ['step', 'sigmoid', 'tanh', 'relu']
plt.figure(figsize=(15, 10))

for i, activation in enumerate(activations):
    # Entraînement
    model = Perceptron(activation_type=activation, learning_rate=0.1, epochs=100)
    model.fit(X_train, y_train, X_test, y_test)
    
    # Tracé frontière de décision
    plt.subplot(4, 2, 2*i + 1)
    plot_decision_boundary(model, X_pca, y, f'Decision Boundary ({activation})')
    
    # Tracé de l'évolution de la précision
    plt.subplot(4, 2, 2*i + 2)
    plt.plot(model.train_acc, label='Train Accuracy')
    plt.plot(model.test_acc, linestyle='--', label='Test Accuracy')
    plt.title(f'Accuracy over Epochs ({activation})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()