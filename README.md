# Neural Network for Reinforcement learning

## Introduction
This project explores the intersection between **machine learning** and **quantum information**.  
It aims to demonstrate how a **neural network** can learn to **stabilize nonequilibrium phases of matter with active feedback** using partial information.

We start from the basics, building a **perceptron** and a **multi-layer perceptron (MLP)** from scratch, before extending these concepts to a **quantum circuit**.

---

## Project Overview

### **1. Classical Neural Network Foundations**
Before applying machine learning to quantum systems, we first build and understand the fundamentals of classical neural networks.

- Implementation of a **simple perceptron** (binary classification in 2D)
- Extension to a **multi-layer perceptron (MLP)** for more complex problems
- Experimentation with architectures and learning parameters

**Goal:** Understand how neural networks learn and generalize decision boundaries.

### **2. Stabilize nonequilibrium phases of matter with active feedback** using partial information *(Main Project)*
The core part of this repository focuses on **Reinforcement learning**.

- 
- 
- 

**Goal:** .

### **3. Classical Image Classification** *(my Personal Bonus)*
As an optional exploration, the project can include a small image classification demo to test the MLP architecture on standard data before applying it to quantum decoding.

---

## Repository Structure

```
📂 Reinforcement-learning/
│
├── 📂 neural-network/
│   │
│   ├── 📂 perceptron/
│   │   ├── perceptron.py         ← Perceptron Classe
│   │   ├── utils.py              ← Fonctions auxiliaires (visualisation, métriques...)
│   │   ├── main.py               ← Main script (data + learning + plot)
│   │   └── perceptron-animation-evolution.py    ← Script pour voir l'évolution de l'apprentissage d'un perceptron
│   │
│   ├── 📂 multi-layer-perceptron/
│   │   ├── layer.py              ← Classe layer pour ce qui se passe dans une couche
│   │   ├── mlp.py                ← Classe du reseau entier
│   │   ├── utils.py              ← Fonctions auxiliaires (visualisation, métriques...)
│   │   └── mlp-test.ipynb        ← Main script (data + learning + plot)
│   │
│   └── 📜 README.md              ← readme associate to the introduction project 
│
│
├── 📄 RL_talk_note_Quant25.pdf   ← Notes from Cemin's Talk about RL
│
└── 📜 README.md                  ← (this file)
```

---

## 🚧 Progress (Soon)
| Task                                         | Status         |
| -------------------------------------------- | -------------- |
| Implemented perceptron                       | ✅              |
| Implemented customizable MLP                 | ✅              |
|                   | 🔄 In progress |
|  | 🔄 Planned     |
|                 | 🔄 Planned     |
| (Optional) Image classification demo         | ⏸ Optional     |

---

## References
- Giovanni Cemin (MPIPKS), "Reinforcement learning to stabilize nonequilibrium phases of matter with active feedback using partial information", Quant25 Conference, 2025.
- Related work: “Entanglement Transitions in Unitary Circuit Games”, ResearchGate, 2024
- [My notes from Cemin's Talk](RL_talk_note_Quant25.pdf)
