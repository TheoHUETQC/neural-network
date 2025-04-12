import numpy as np
import matplotlib.pyplot as plt

Nt = 100 #nombre de prediction
Nx = 100 #nombre d'entré

weights = np.random.randn(2) # les poids (des entiers random)
bias = np.random.randn() #le biais (des random)
lr = 0.1 #taux d aprentissage

def f(x) : # Sigmoïde qui va de 0 a 1
    return 1/(1+ np.exp(-x))

def predict(x) : # Fonction de prédiction
    return f(np.dot(x, weights) + bias)

x, y = np.random.rand(Nx) , np.random.rand(Nx)
inputs = np.stack([x, y], axis=1)

z_reel = (x > y).astype(int) # Classe 1 si y > x, sinon 0

"""# Affichage des points
plt.scatter(x, y, c=z_reel, cmap="bwr", edgecolors="k")
plt.plot(np.linspace(0, 1, 100), np.linspace(0,1,100), color = "g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Données classifiées")
plt.show()
"""
# Apprentissage
for t in range(Nt) :
    for i in range(Nx) :
        pred = predict(inputs[i])
        error = z_reel[i] - pred
        
        weights = weights + lr * error * inputs[i]
        bias = bias + lr * error

# Prédictions finales
z_pred = np.array([predict(inputs[i]) for i in range(Nx)])

# Affichage des résultats
plt.scatter(x, y, c=z_pred, cmap="bwr", edgecolors="k")
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="g")
plt.title("Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()