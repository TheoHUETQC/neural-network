import numpy as np
import matplotlib.pyplot as plt

Nt = 100 #nombre de prediction
Nx = 100 #nombre d'entrée

xmin, xmax = 0, 10
ymin, ymax = 0, 10

def g(x) :
    return x

class Perceptron :
    def __init__(self, input_size, lr=0.1) :
        self.weights = np.random.randn(input_size) # les poids (des entiers random)
        self.bias = np.random.randn() #le biais (des random)
        self.lr = 0.1 #taux d aprentissage
        
    def f(self, x) : # Sigmoïde qui va de 0 a 1
        return 1/(1+ np.exp(-x))
    
    def predict(self, x) : # Fonction de prédiction
        return self.f(np.dot(x, self.weights) + self.bias)
    
    def train(self, inputs, z_reel, Nt) : # Apprentissage
        for t in range(Nt) :
            for i in range(len(inputs)) :
                pred = self.predict(inputs[i])
                error = z_reel[i] - pred # calcul de l'erreur entre la prédiction et le bon résultat
                
                self.weights = self.weights + self.lr * error * inputs[i] # Mise à jour des poids
                self.bias = self.bias + self.lr * error # Mise à jour du biais
        #pas de return car on souhaite juste ajuster bias et weights pour améliorer nos futurs predictions

#initialise des points aléatoires
x, y = np.random.rand(Nx) * xmax + xmin , np.random.rand(Nx) * ymax + ymin
inputs = np.stack([x, y], axis=1)

# bon résultat a comparer avec nos prédictions 
z_reel = (g(x) > y).astype(int) # Classe 1 si y > x, sinon 0

"""# Affichage des points
plt.scatter(x, y, c=z_reel, cmap="bwr", edgecolors="k")
plt.plot(np.linspace(xmin, xmax, 100), g(np.linspace(xmin, xmax,100)), color = "g")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Données classifiées")
plt.show()"""

# creer notre perceptron
perceptron = Perceptron(2) # 2 car (x,y)

# entraine notre perceptron
perceptron.train(inputs, z_reel, Nt)

# Prédictions finales
z_pred = np.array([perceptron.predict(inputs[i]) for i in range(Nx)])

# Affichage des résultats
plt.scatter(x, y, c=z_pred, cmap="bwr", edgecolors="k")
plt.plot(np.linspace(xmin, xmax, 100), g(np.linspace(xmin, xmax, 100)), color="g")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.title("Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
