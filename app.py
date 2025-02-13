# --------------------------
# Importation des bibliothèques nécessaires
# --------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Chargement et prétraitement des données
# --------------------------
iris = load_iris()               # Chargement du dataset Iris (1100 échantillons, 4 features, 3 classes)
X = iris.data                    # Récupération des features (longueur/largeur sépales, longueur/largeur pétales)
y = iris.target                  # Récupération des labels (classes 0,1,2)

# Sélection des deux premières classes (0 et 1) pour faire de la classification binaire
# On enlève la classe 2
X = X[y != 2]                    # Conserve uniquement les échantillons de classe 0 et 1
y = y[y != 2]                    # Conserve uniquement les étiquettes de classe 0 et 1

# --------------------------
# Normalisation et PCA
# --------------------------
scaler = StandardScaler()        # StandardScaler pour normaliser les données (moyenne 0, variance 1)
X_scaled = scaler.fit_transform(X)  # Ajuste le scaler et transforme X
pca = PCA(n_components=2)       # Réduction dimensionnelle en 2 composantes principales
X_pca = pca.fit_transform(X_scaled)

# --------------------------
# Séparation en jeu d'entraînement et de test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, 
    y, 
    test_size=0.3,    # 30% pour le test
    random_state=42   # Pour la reproductibilité
)

# Fixe la graine random pour garantir la même initialisation des poids à chaque exécution
np.random.seed(42)

# --------------------------
# Définition de la classe Perceptron
# --------------------------
class Perceptron:
    """
    Classe définissant un perceptron simple avec une fonction d'activation paramétrable.
    """
    def __init__(self, input_size, activation_type, learning_rate=0.01):
        """
        Initialise les attributs du perceptron :
          - activation_type : type de fonction d'activation (step, sigmoid, tanh, relu)
          - lr : taux d'apprentissage
          - weights : poids (initialisés aléatoirement)
          - bias : biais (initialisé aléatoirement)
        """
        self.activation_type = activation_type
        self.lr = learning_rate
        self.weights = np.random.randn(input_size, 1)  # input_size poids pour chaque entrée
        self.bias = np.random.randn()                  # un biais unique

    def activation(self, x):
        """
        Calcule la sortie de la fonction d'activation en fonction de activation_type.
        """
        if self.activation_type == 'step':
            # Renvoie 1 si x >= 0, sinon 0
            return np.where(x >= 0, 1, 0)
        elif self.activation_type == 'sigmoid':
            # Sigmoïde : 1 / (1 + e^-x)
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            # Hyperbolic tan : tanh(x)
            return np.tanh(x)
        elif self.activation_type == 'relu':
            # ReLU : max(0, x)
            return np.maximum(0, x)

    def derivative(self, x):
        """
        Calcule la dérivée de la fonction d'activation pour le backprop.
        """
        if self.activation_type == 'sigmoid':
            # Dérivée de la sigmoïde : sig(x)*(1 - sig(x))
            sig = self.activation(x)
            return sig * (1 - sig)
        elif self.activation_type == 'tanh':
            # Dérivée de tanh(x) : 1 - tanh^2(x)
            return 1 - np.tanh(x)**2
        elif self.activation_type == 'relu':
            # Dérivée de ReLU : 1 si x>0, sinon 0
            return np.where(x > 0, 1, 0)
        else:
            # Pour step et autres cas non différentiables : renvoie 1
            return np.ones_like(x)

    def forward(self, X):
        """
        Calcule la sortie linéaire (linear) puis applique la fonction d'activation.
        Retourne :
          - la sortie activée
          - la sortie linéaire (pour usage éventuel dans la backprop)
        """
        linear = np.dot(X, self.weights) + self.bias   # x*w + b
        return self.activation(linear).reshape(-1, 1), linear.reshape(-1, 1)

    def update(self, X, error):
        """
        Met à jour les poids et le biais en utilisant la descente de gradient :
          - grad_w = dérivée w.r.t. w
          - grad_b = dérivée w.r.t. b
        """
        grad_w = np.dot(X.T, error)       # Gradient par rapport aux poids
        grad_b = np.sum(error)           # Gradient par rapport au biais

        # Mise à jour (descente de gradient)
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b

    def fit(self, X_train, y_train, X_test, y_test, epochs=10):
        """
        Entraîne le perceptron (forward + backprop) sur un nombre d'époques donné.
        Retourne l'évolution des précisions en train et test.
        """
        y_train = y_train.reshape(-1,1)  # Assure des dimensions cohérentes
        y_test = y_test.reshape(-1,1)    # Assure des dimensions cohérentes
        train_acc = []
        test_acc = []

        # Boucle d'apprentissage
        for epoch in range(epochs):
            output, _ = self.forward(X_train)    # Sortie du réseau sur le train
            error = output - y_train             # Erreur = prediction - label
            self.update(X_train, error)          # Mise à jour des paramètres

            # On calcule la précision sur train et test à chaque époque
            train_acc.append(self.accuracy(X_train, y_train))
            test_acc.append(self.accuracy(X_test, y_test))

        return train_acc, test_acc

    def predict(self, X):
        """
        Prédit la classe (0 ou 1) à partir de la sortie, en considérant un seuil de 0.5.
        """
        output, _ = self.forward(X)
        return np.where(output >= 0.5, 1, 0)

    def accuracy(self, X, y):
        """
        Calcule la précision (taux de bonnes réponses) du perceptron.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y.reshape(-1, 1))

# --------------------------
# Classe Perceptron en série (réseau à deux couches)
# --------------------------
class TwoLayerPerceptron:
    """
    Réseau de deux perceptrons en série (couche cachée + couche de sortie).
    """
    def __init__(self, input_size, hidden_size, activation1, activation2, learning_rate=0.01, epochs=10):
        """
        Initialise :
          - perceptron1 : couche cachée
          - perceptron2 : couche de sortie
          - epochs : nombre d'époques
          - learning_rate : taux d'apprentissage
        """
        self.perceptron1 = Perceptron(input_size, activation1, learning_rate)
        self.perceptron2 = Perceptron(hidden_size, activation2, learning_rate)
        self.epochs = epochs
        self.train_acc = []
        self.test_acc = []
    
    def fit(self, X_train, y_train, X_test, y_test):
        """
        Boucle d'apprentissage sur un nombre d'époques donné.
        Effectue la propagation avant et la rétropropagation des erreurs.
        """
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        for epoch in range(self.epochs):
            # Forward
            hidden_output, _ = self.perceptron1.forward(X_train)  # Sortie de la 1ère couche
            output, _ = self.perceptron2.forward(hidden_output)   # Sortie finale

            # Calcul de l'erreur
            error = output - y_train
            # Calcul de la dérivée pour la couche de sortie
            delta_output = error * self.perceptron2.derivative(output)

            # Calcul de la dérivée pour la couche cachée
            delta_hidden = (delta_output @ self.perceptron2.weights.T) * self.perceptron1.derivative(hidden_output)

            # Mise à jour des poids et biais
            self.perceptron2.update(hidden_output, delta_output)
            self.perceptron1.update(X_train, delta_hidden)

            # Stocke la précision en train et test à chaque époque
            self.train_acc.append(self.accuracy(X_train, y_train))
            self.test_acc.append(self.accuracy(X_test, y_test))
    
    def predict(self, X):
        """
        Prédit la classe en effectuant un forward complet.
        """
        hidden_output, _ = self.perceptron1.forward(X)
        output, _ = self.perceptron2.forward(hidden_output)
        return np.where(output >= 0.5, 1, 0)
    
    def accuracy(self, X, y):
        """
        Calcule la précision de prédiction du modèle.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y.reshape(-1, 1))

# --------------------------
# Classe Perceptron en parallèle (réseau à deux perceptrons en parallèle + perceptron de sortie)
# --------------------------
class ParallelPerceptronNetwork:
    """
    Réseau composé de deux perceptrons en parallèle, suivis d'un perceptron de sortie.
    """
    def __init__(self, input_size, activation1, activation2, activation_output, learning_rate=0.01, epochs=100):
        """
        Initialise :
          - perceptron1 : 1er perceptron
          - perceptron2 : 2ème perceptron
          - perceptron_output : perceptron de sortie qui combine les deux sorties précédentes
        """
        self.perceptron1 = Perceptron(input_size, activation1, learning_rate)
        self.perceptron2 = Perceptron(input_size, activation2, learning_rate)
        self.perceptron_output = Perceptron(2, activation_output, learning_rate)
        self.epochs = epochs
        self.train_acc = []
        self.test_acc = []
    
    def fit(self, X_train, y_train, X_test, y_test):
        """
        Boucle d'apprentissage sur un nombre d'époques donné, 
        avec propagation avant et rétropropagation parallèle.
        """
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        for epoch in range(self.epochs):
            # Forward vers perceptron1 et perceptron2
            output1, _ = self.perceptron1.forward(X_train)
            output2, _ = self.perceptron2.forward(X_train)

            # Combine leurs sorties pour former l'entrée du perceptron de sortie
            combined_input = np.hstack((output1, output2))
            output, _ = self.perceptron_output.forward(combined_input)

            # Calcul de l'erreur
            error = output - y_train
            # Dérivée pour la couche de sortie
            delta_output = error * self.perceptron_output.derivative(output)

            # Dérivée des couches précédentes
            delta_hidden1 = (delta_output @ self.perceptron_output.weights[:1].T) * self.perceptron1.derivative(output1)
            delta_hidden2 = (delta_output @ self.perceptron_output.weights[1:].T) * self.perceptron2.derivative(output2)

            # Mise à jour du perceptron de sortie et des deux perceptrons en entrée
            self.perceptron_output.update(combined_input, delta_output)
            self.perceptron1.update(X_train, delta_hidden1)
            self.perceptron2.update(X_train, delta_hidden2)

            # Stocker la précision en train et test
            self.train_acc.append(self.accuracy(X_train, y_train))
            self.test_acc.append(self.accuracy(X_test, y_test))
    
    def predict(self, X):
        """
        Retourne la prédiction de classe finale (0/1).
        """
        output1, _ = self.perceptron1.forward(X)
        output2, _ = self.perceptron2.forward(X)
        combined_input = np.hstack((output1, output2))
        output, _ = self.perceptron_output.forward(combined_input)
        return np.where(output >= 0.5, 1, 0)

    def accuracy(self, X, y):
        """
        Calcule la précision de prédiction du réseau parallèle.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y.reshape(-1, 1))

# --------------------------
# Fonction pour tracer la frontière de décision
# --------------------------
def plot_decision_boundary(model, X, y, title, ax):
    """
    Trace la frontière de décision d'un modèle binaire sur une grille 2D.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Création d'une grille de points (xx, yy)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Prédictions sur la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # On combine pour créer la matrice d'entrée
    Z = Z.reshape(xx.shape)

    # Affichage d'informations debug sur la grille
    print(f"{title} - Z min/max: {Z.min()}, {Z.max()}")

    # Création des zones colorées (contours)
    ax.contourf(xx, yy, Z, alpha=0.4)
    # Affichage des points réels
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    ax.set_title(title)


# --------------------------
# Visualisation des résultats
# --------------------------
activations = ['step', 'sigmoid', 'tanh', 'relu']  # Différents types de fonctions d'activation à tester

# Création des figures/subplots pour Perceptron Simple
fig1, axs1 = plt.subplots(2, 4, figsize=(20, 10))

# Création des figures/subplots pour Perceptron en Série
fig2, axs2 = plt.subplots(2, 4, figsize=(20, 10))

# Création des figures/subplots pour Perceptron en Parallèle
fig3, axs3 = plt.subplots(2, 4, figsize=(20, 10))

# Boucle sur chaque type d'activation
for i, activation in enumerate(['step', 'sigmoid', 'tanh', 'relu']):
    # ----------
    # 🔵 Perceptron Simple
    # ----------
    perceptron = Perceptron(2, activation, 0.01)          # Initialise un perceptron simple
    train_acc, test_acc = perceptron.fit(X_train, y_train, X_test, y_test, 100)  # Entraînement

    # Trace la frontière de décision du perceptron simple
    plot_decision_boundary(perceptron, X_pca, y, f'Perceptron Simple ({activation})', axs1[0, i])

    # Affichage des courbes d'apprentissage (précision train vs test)
    axs1[1, i].plot(train_acc, label='Train Accuracy')
    axs1[1, i].plot(test_acc, linestyle='--', label='Test Accuracy')
    axs1[1, i].legend()

    # ----------
    # 🔴 Perceptron en Série
    # ----------
    model_series = TwoLayerPerceptron(2, 1, activation, 'sigmoid', 0.01, 100)  # 1 neurone caché, activation sortie = sigmoid
    model_series.fit(X_train, y_train, X_test, y_test)

    # Trace la frontière de décision du perceptron en série
    plot_decision_boundary(model_series, X_pca, y, f'Perceptron en Série ({activation})', axs2[0, i])

    # Affiche les courbes d'apprentissage (précision en train/test)
    axs2[1, i].plot(model_series.train_acc, label='Train Accuracy')
    axs2[1, i].plot(model_series.test_acc, linestyle='--', label='Test Accuracy')
    axs2[1, i].legend()

    # ----------
    # 🟠 Perceptron en Parallèle
    # ----------
    model_parallel = ParallelPerceptronNetwork(2, activation, activation, 'sigmoid', 0.01, 100)
    model_parallel.fit(X_train, y_train, X_test, y_test)

    # Trace la frontière de décision du perceptron en parallèle
    plot_decision_boundary(model_parallel, X_pca, y, f'Perceptron Parallèle ({activation})', axs3[0, i])

    # Affiche les courbes d'apprentissage (précision en train/test)
    axs3[1, i].plot(model_parallel.train_acc, label='Train Accuracy')
    axs3[1, i].plot(model_parallel.test_acc, linestyle='--', label='Test Accuracy')
    axs3[1, i].legend()

# Affiche l'ensemble des figures générées
plt.show()
