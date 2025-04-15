import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from perceptron import Perceptron

############################### paramètres ###############################

#constante de numérisation :
Nt = 100 #nombre de prediction
Nx = 100 #nombre d'entrée

#constante du problème :
xmin, xmax = 0, 10
ymin, ymax = 0, 10

############################### fonctions ###############################

def g(x) : # on veut regarder si le point (x,y) est au dessus de (x,g(x))
    return x

############################### Main ###############################

#initialisation des points aléatoires
x, y = np.random.rand(Nx) * xmax + xmin , np.random.rand(Nx) * ymax + ymin
inputs = np.stack([x, y], axis=1)

# initialisation du bon résultat a comparer avec nos prédictions 
z_reel = (g(x) > y).astype(int) # Classe 1 si y > x, sinon 0

"""# Affichage des points réél
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
prediction_for_animation = perceptron.train(inputs, z_reel, Nt)

# Prédictions finales
z_pred = np.array([perceptron.predict(inputs[i]) for i in range(Nx)]) #donne des valeurs entre 0 et 1
z_pred_int = np.array([perceptron.predict_label(inputs[i]) for i in range(Nx)]) #donne soit 1 soit 0

#precision de notre perceptron
acc = np.mean(z_pred_int == z_reel)
print(f"Précision : {acc*100:.2f}%")

# Affichage des résultats finaux
plt.scatter(x, y, c=z_pred, cmap="bwr", edgecolors="k")
plt.plot(np.linspace(xmin, xmax, 100), g(np.linspace(xmin, xmax, 100)), color="g")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.title("Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
