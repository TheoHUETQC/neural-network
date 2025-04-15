# 🧠🔬 Classifier des états quantiques avec un réseau de neurones

## 🚀 Introduction
Ce projet vise à explorer l'application des réseaux de neurones à la classification des états quantiques. Nous commencerons par un **projet introductif** sur les réseaux de neurones classiques avant d'étendre notre approche à des problématiques de l'informatique quantique.

📌 **Objectif final** : Utiliser un réseau de neurones pour classifier des états quantiques en fonction de certaines propriétés physiques (ex: entanglement, phase topologique, etc.).

---

## 📚 Plan du projet

### **1️⃣ Introduction aux réseaux de neurones** *(Projet de base)*
Avant d'appliquer un réseau de neurones à des problèmes quantiques, nous devons comprendre les fondamentaux.

- Implémentation d'un **perceptron simple** (classification de points 2D)
- Extension vers un **MLP (Multi-Layer Perceptron)** avec TensorFlow/PyTorch
- Expérimentations avec différents hyperparamètres

👉 **Objectif** : Comprendre comment un réseau apprend et généralise une classification binaire.

---

### **2️⃣ Génération et manipulation d'états quantiques**
Pour appliquer le deep learning en quantique, nous devons générer et manipuler des **données quantiques**.

- Introduction à **Qiskit** pour simuler des états quantiques
- Génération d'**états aléatoires de qubits** (pur et mixte)
- Extraction de **features** à partir des matrices de densité

👉 **Objectif** : Avoir un dataset d'états quantiques prêts à être classifiés.

---

### **3️⃣ Construction d'un classificateur quantique**
Nous appliquerons un réseau de neurones classique pour classifier ces états.

- Conception d'un **MLP** capable de distinguer des classes d'états quantiques
- Entraînement sur des **mesures physiques** (entropie de von Neumann, concurrence, etc.)
- Test et validation du modèle

👉 **Objectif** : Observer si un réseau de neurones peut apprendre des caractéristiques physiques d'un système quantique.

---

### **4️⃣ Vers un classificateur quantique hybride** *(Bonus avancé)*
Une fois les bases posées, nous pourrons aller plus loin avec des architectures hybrides :

- Introduction aux **Quantum Neural Networks (QNNs)** avec Pennylane
- Entraînement d'un modèle **hybride classique-quantique**
- Comparaison des performances entre approche classique et quantique

👉 **Objectif** : Comprendre comment les QNNs peuvent améliorer la classification d'états quantiques.

---

## 🛠 Structure du dossier

```
📂 neural-network/
│
├── 📂 introduction/
│   │
│   ├── 📂 perceptron/
│   │   ├── perceptron.py         ← Classe Perceptron
│   │   ├── utils.py              ← Fonctions auxiliaires (visualisation, métriques...)
│   │   ├── main.py               ← Script principal (data + apprentissage + affichage)
│   │   │
│   │   └── 📂 results/
│   │       ├── perceptron-animation-evolution.py    ← Script pour voir l'évolution de l'apprentissage d'un perceptron
│   │       ├── perceptron-training-evolution.mp4       
│   │       └── perceptron-result.png 
│   │
│   ├── 📂 multi-layer-perceptron/
│   │   ├── layer.py              ← Classe layer pour ce qui se passe dans une couche
│   │   ├── mlp.py                ← Classe du reseau entier
│   │   ├── utils.py              ← Fonctions auxiliaires (visualisation, métriques...)
│   │   ├── main.py               ← Script principal (data + apprentissage + affichage)
│   │   │
│   │   └── 📂 results/
│   │       └── multi-layer-perceptron-result.png
│   │
│   └── 📜 README.md              ← (readme associé au projet d'introduction)
│ 
└── 📜 README.md                  ← (ce fichier)
```

## 📦 Installation et Prérequis

1. **Cloner ce repo**
```bash
git clone https://github.com/theohuetqc/neural-network.git
cd neural-network
```

2. **Installer les dépendances Python**
```bash
pip install numpy matplotlib tensorflow torch qiskit pennylane
```

---

## 🛠 Progression (Soon)
- [X] Implémentation d'un perceptron simple
- [ ] Ajout d'un MLP plus complexe
- [ ] Intégration de Qiskit pour générer des états quantiques
- [ ] Construction du classificateur quantique
- [ ] Expérimentations et analyses
