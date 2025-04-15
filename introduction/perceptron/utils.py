import numpy as np
import matplotlib.pyplot as plt

def generate_data(Nx, frontiere, xmin=0, xmax=10, ymin=0, ymax=10): # on veut regarder si le point (x,y) est au dessus de (x,g(x))
    #initialisation des points aléatoires
    x, y = np.random.rand(Nx) * (xmax - xmin) + xmin, np.random.rand(Nx) * (ymax - ymin) + ymin
    inputs = np.stack([x, y], axis=1)

    # initialisation du bon résultat a comparer avec nos prédictions 
    z_reel = (frontiere(x) > y).astype(int) # Classe 1 si y > x, sinon 0

    return x, y, inputs, z_reel

def plot_classification(x, y, z, frontiere, xmin=0, xmax=10, ymin=0, ymax=10, title=""):
    plt.scatter(x, y, c=z, cmap="bwr", edgecolors="k")
    plt.plot(np.linspace(xmin, xmax, 100), frontiere(np.linspace(xmin, xmax, 100)), color="g")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()
    
def accuracy(y_true, y_pred): #precision de notre perceptron
    return np.mean(y_true == y_pred)