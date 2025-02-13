import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Charger le dataset Iris
iris = load_iris()
X = iris.data  # On prend toutes les features pour appliquer le PCA
y = (iris.target != 0).astype(int)  # Classes binaires (0 vs non-0)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application du PCA (2 composantes principales)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Séparation en train/test
x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Fonctions d'activation
def step(z): return np.where(z >= 0, 1, 0)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -600, 600)))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)

activations = {"step": step, "sigmoid": sigmoid, "tanh": tanh, "relu": relu}

# Classe Perceptron
class Perceptron:
    def __init__(self):
        self.W = None
        self.B = None

    def fit(self, X, y, alpha=0.1, epochs=1000, activation="step"):
        self.W = np.zeros(X.shape[1])
        self.B = 0
        activation_f = activations[activation]

        for _ in range(epochs):
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.W) + self.B
                yi_pred = activation_f(z)
                self.W += alpha * (yi - yi_pred) * xi
                self.B += alpha * (yi - yi_pred)

    def predict(self, X, activation="step"):
        activation_f = activations[activation]
        z = np.dot(X, self.W) + self.B
        return (activation_f(z) > 0.5).astype(int)

# Entraînement et affichage
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for idx, (name, act) in enumerate(activations.items()):
    p = Perceptron()
    p.fit(x_train, y_train, alpha=0.1, epochs=1000, activation=name)

    # Frontière de décision
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = p.predict(np.c_[xx.ravel(), yy.ravel()], activation=name).reshape(xx.shape)

    axs[idx].contourf(xx, yy, Z, alpha=0.3)
    axs[idx].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='Class 0', edgecolor='k')
    axs[idx].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='Class 1', edgecolor='k')
    axs[idx].set_title(f"Activation: {name}")
    axs[idx].set_xlabel("PCA Component 1")
    axs[idx].set_ylabel("PCA Component 2")

plt.suptitle("Frontière de décision après PCA", fontsize=16)
plt.legend()
plt.show()
