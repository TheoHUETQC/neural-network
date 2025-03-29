# Introduction aux Réseaux de Neurones en Python

Ce projet est une introduction pratique aux réseaux de neurones artificiels en Python. Il vise à familiariser les débutants avec les concepts de base du machine learning à travers un exemple simple de classification binaire.

## 🔮 Objectif du projet

L'objectif est de construire et d'entraîner un **réseau de neurones simple** pour classifier des points dans un espace 2D. Le réseau apprendra à distinguer deux classes de points (rouge et bleu) à l'aide d'un modèle de type Multi-Layer Perceptron (MLP).

## 🎓 Concepts abordés
- Perceptron et neurones artificiels
- Fonction d'activation (ReLU, sigmoïde)
- Entraînement par descente de gradient et rétropropagation
- Évaluation et visualisation des résultats

## 📝 Plan du projet

1. **Génération de données** : Création d'un dataset simple avec des points en 2D.
2. **Construction du réseau de neurones** : Un MLP avec une ou deux couches cachées.
3. **Entraînement du modèle** : Optimisation avec la descente de gradient.
4. **Visualisation des résultats** : Représentation graphique de la classification.
5. **Expérimentations** : Ajustement des hyperparamètres et test sur d'autres datasets.

## 🔧 Technologies utilisées
- **Python** (3.x)
- **NumPy** (gestion des matrices et des vecteurs)
- **Matplotlib** (visualisation des points)
- **TensorFlow / PyTorch** (création et entraînement du réseau de neurones)

## 🛠 Installation et exécution

1. Cloner ce repository :
   ```bash
   git clone https://github.com/theohuetqc/neural-network.git
   cd neural-network
   ```
2. Installer les dépendances :
   ```bash
   pip install numpy matplotlib tensorflow  # ou torch
   ```
3. Lancer le script principal :
   ```bash
   python neurostart.py
   ```
