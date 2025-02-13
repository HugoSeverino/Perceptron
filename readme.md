# README : Implémentations de Perceptrons et Réseaux Simples en Python

Ce dépôt présente une implémentation **from scratch** de différents modèles de perceptrons et petites architectures neuronales pour la classification binaire sur l’exemple classique des **données Iris** (en ne gardant que deux classes). Nous illustrons également la **réduction de dimension** avec la méthode PCA, la **normalisation** des données, et le **tracé de la frontière de décision** finale.

---

## Sommaire

1. [Contexte et objectifs](#contexte-et-objectifs)  
2. [Description des fichiers](#description-des-fichiers)  
3. [Théorie mathématique](#théorie-mathématique)  
   1. [Rappels sur le Perceptron simple](#rappels-sur-le-perceptron-simple)  
   2. [Fonctions d’activation](#fonctions-dactivation)  
   3. [Descente de gradient et mise à jour des poids](#descente-de-gradient-et-mise-à-jour-des-poids)  
   4. [Perceptron en série (réseau à deux couches)](#perceptron-en-série-réseau-à-deux-couches)  
   5. [Perceptron en parallèle](#perceptron-en-parallèle)  
   6. [Réduction de dimension : PCA](#réduction-de-dimension--pca)  
   7. [Visualisation de la frontière de décision](#visualisation-de-la-frontière-de-décision)  
4. [Utilisation](#utilisation)  
5. [Résultats attendus et interprétation](#résultats-attendus-et-interprétation)  
6. [Références](#références)  

---

## Contexte et objectifs

Ce projet a pour but de mettre en œuvre **plusieurs variantes de perceptrons** (simple, en série, en parallèle) pour réaliser une **classification binaire** (fleurs Iris Setosa vs Versicolor) et d’étudier l’impact de **différentes fonctions d’activation** (step, sigmoid, tanh, ReLU).

L’intégralité du code est implémentée **en pur NumPy**, sans utiliser d’autres bibliothèques de réseaux de neurones (comme TensorFlow ou PyTorch). L’objectif est de comprendre les bases du **calcul de la propagation avant (forward pass)** et **de la rétropropagation (backprop)**, ainsi que de visualiser la **frontière de décision** résultante.

---

## Description des fichiers

- **`main.py`** : Contient l’ensemble du code Python partagé (ou un code similaire) :
  - Chargement du dataset Iris,
  - Normalisation par `StandardScaler`,
  - Réduction de dimension par `PCA`,
  - Séparation en jeu d’entraînement/test,
  - Définition des différentes classes :
    - `Perceptron` (perceptron simple)
    - `TwoLayerPerceptron` (perceptron en série)
    - `ParallelPerceptronNetwork` (perceptron en parallèle)
  - Entraînement et évaluation pour chaque fonction d’activation
  - Tracé des frontières de décision et des courbes d’apprentissage.

---

## Théorie mathématique

### 1. Rappels sur le Perceptron simple

Un **Perceptron** est le bloc de base d’un réseau de neurones artificiel.  
Il réalise un calcul linéaire, suivi d’une fonction d’activation non linéaire.

#### Calcul linéaire

$$
z = \mathbf{W}^T \mathbf{x} + b
$$

- \(\mathbf{x}\) : vecteur d’entrée (de taille \((n, )\) si on a \(n\) features)  
- \(\mathbf{W}\) : vecteur de poids (de taille \((n, )\))  
- \(b\) : biais (scalaire)

#### Sortie (activation)

$$
\hat{y} = \sigma(z)
$$

où \(\sigma(\cdot)\) est une fonction d’activation (voir [Fonctions d’activation](#fonctions-dactivation)).

### 2. Fonctions d’activation

Nous implémentons quatre fonctions d’activation différentes :

1. **Step** :

   $$
   \text{step}(z) = 
   \begin{cases} 
      1 & \text{si } z \ge 0 \\ 
      0 & \text{sinon} 
   \end{cases}
   $$

2. **Sigmoid** :

   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}
   $$

3. **Tanh** :

   $$
   \tanh(z) = \frac{e^z - e^{-z}}{\,e^z + e^{-z}\,}
   $$

4. **ReLU** :

   $$
   \text{ReLU}(z) = \max(0, z)
   $$

Leurs dérivées (approximatives pour la step, exacte pour les autres) sont essentielles lors de la **rétropropagation**.

### 3. Descente de gradient et mise à jour des poids

Pour entraîner un perceptron, on minimise (souvent) une **erreur quadratique** ou une **log-likelihood**. Ici, par simplicité, on fait :

$$
\text{Erreur} = \hat{y} - y
$$

où \(\hat{y}\) est la sortie du perceptron et \(y\) la cible (0 ou 1). Puis on met à jour les paramètres \(W\) et \(b\) via la descente de gradient :

$$
\begin{aligned}
\frac{\partial \text{Erreur}}{\partial W} &= X^T \cdot (\hat{y} - y) \\
\frac{\partial \text{Erreur}}{\partial b} &= \sum (\hat{y} - y)
\end{aligned}
$$

Le **taux d’apprentissage** \(\eta\) (learning rate) permet de contrôler la vitesse de mise à jour :

$$
\begin{aligned}
W &\leftarrow W - \eta \cdot \frac{\partial \text{Erreur}}{\partial W} \\
b &\leftarrow b - \eta \cdot \frac{\partial \text{Erreur}}{\partial b}
\end{aligned}
$$

### 4. Perceptron en série (réseau à deux couches)

Dans un **réseau en série** à deux couches, la sortie de la première couche (couche cachée) sert d’entrée à la deuxième (couche de sortie).

- Couche 1 :

  $$
  z_1 = W_1^T X + b_1 \quad \Rightarrow \quad h = \sigma_1(z_1)
  $$

- Couche 2 :

  $$
  z_2 = W_2^T h + b_2 \quad \Rightarrow \quad \hat{y} = \sigma_2(z_2)
  $$

L’erreur \(\hat{y} - y\) est rétropropagée :  
- On calcule d’abord la dérivée de la couche de sortie :

  $$
  \delta_{\text{sortie}} = (\hat{y} - y) \times \sigma_2'(z_2)
  $$

- Puis on remonte à la couche cachée :

  $$
  \delta_{\text{cachée}} = \delta_{\text{sortie}} \times W_2^T \times \sigma_1'(z_1)
  $$

- Enfin, on met à jour les poids \(W_2\) et \(b_2\) puis \(W_1\) et \(b_1\).

### 5. Perceptron en parallèle

Dans un **réseau parallèle**, deux perceptrons reçoivent la **même** entrée \(\mathbf{x}\) en parallèle. On concatène leurs sorties pour alimenter un troisième perceptron de sortie :

$$
\begin{aligned}
z_1 &= W_1^T X + b_1 \quad\Rightarrow\quad h_1 = \sigma_1(z_1) \\
z_2 &= W_2^T X + b_2 \quad\Rightarrow\quad h_2 = \sigma_2(z_2) \\
\text{Entrée de la couche de sortie} &= [\,h_1 \;\; h_2\,]
\end{aligned}
$$

La sortie finale est :

$$
\hat{y} = \sigma_{\text{out}}\bigl(W_{\text{out}}^T \,[h_1 \; h_2] + b_{\text{out}}\bigr).
$$

On réalise alors la rétropropagation en dérivant d’abord par rapport aux poids de la couche de sortie, puis en rétropropageant la partie de l’erreur attribuable à chacun des deux perceptrons parallèles.

### 6. Réduction de dimension : PCA

Afin de **visualiser facilement** la frontière de décision, on applique une **Analyse en Composantes Principales** (PCA) pour réduire les données Iris de **4 dimensions** (longueur/largeur sépale, longueur/largeur pétale) à **2 dimensions**.

Théoriquement, la PCA effectue une décomposition en valeurs singulières de la matrice centrée \(\tilde{X}\) (après normalisation). Les **2 premières composantes principales** retiennent la plus grande part de variance des données.

### 7. Visualisation de la frontière de décision

Pour tracer la frontière de décision, on :  
1. Crée une grille dense de points dans le plan \((x, y)\) (issu de la PCA).  
2. On fait prédire à notre modèle (ex. Perceptron) la classe pour chacun de ces points.  
3. On reconstruit l’image 2D (en coloriant les zones prédites 0 ou 1).

Cela permet de **visualiser** l’effet de la fonction d’activation et du type d’architecture.

---

## Utilisation

1. **Cloner** le dépôt ou copier le code.
2. **Installer les dépendances** si besoin :
   ```bash
   pip install numpy matplotlib scikit-learn



3. **Exécuter** le script principal (par exemple `python main.py`).

Dans le code, on trouve :
- La création et l’entraînement de :
  - Un perceptron simple `Perceptron(2, activation, 0.01)`
  - Un perceptron en série `TwoLayerPerceptron(2, 1, ...)`
  - Un perceptron en parallèle `ParallelPerceptronNetwork(2, ...)`
- Des **boucles** pour tester quatre fonctions d’activation : *step*, *sigmoid*, *tanh*, *relu*
- Le **tracé** des frontières de décision et des **courbes d’apprentissage** (précision en entraînement et en test).

---

## Résultats attendus et interprétation

- **Frontières de décision** :
  - Avec la *step*, la frontière est souvent plus **brusque** (décision tout ou rien).  
  - Avec des fonctions continues comme *sigmoid* ou *tanh*, on peut obtenir des frontières plus **douces** et une convergence plus stable.  
  - *ReLU* se comporte parfois comme un classifieur linéaire selon les données, avec des zones à 0.
- **Courbes d’apprentissage** :
  - On observe l’amélioration de la précision en fonction des époques.
  - La stabilité dépend aussi du taux d’apprentissage et de la fonction d’activation.

Selon la difficulté du jeu de données (ici binaire sur Iris, qui est relativement simple), on obtient rapidement de bons scores (souvent proches de 100% de précision).

---

## Références

- **Rosenblatt, F. (1958).** *The Perceptron: a probabilistic model for information storage and organization in the brain.* Psychological Review, 65(6).
- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature.
- Documentation [**NumPy**](https://numpy.org/doc/stable/).
- Documentation [**scikit-learn**](https://scikit-learn.org/stable/).

---

**Fin du document**
```
