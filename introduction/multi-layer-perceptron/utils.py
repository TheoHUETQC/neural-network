import numpy as np
import matplotlib.pyplot as plt
#import networkx as nx

def generate_data(Nx, frontiere, xmin=0, xmax=10, ymin=0, ymax=10): # on veut regarder si le point (x,y) est au dessus de (x,g(x))
    #initialisation des points aléatoires
    x, y = np.random.rand(Nx) * (xmax - xmin) + xmin, np.random.rand(Nx) * (ymax - ymin) + ymin
    inputs = np.stack([x, y], axis=1)

    # initialisation du bon résultat a comparer avec nos prédictions 
    z_true = (frontiere(x) > y).astype(int) # Classe 1 si y > x, sinon 0

    return x, y, inputs, z_true

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

def mlp_simplify_view(mlp) : #plot un schéma du reseau de neuronne
    Lx, Ly = 10, 10
    dist_x = Lx / len(mlp.perceptron_by_layer)
    
    plt.figure()
    layers = []
    
    for lay_idx in range(len(mlp.perceptron_by_layer)) :
        perceptrons = []
        
        percept_color = ('blue', 'red')[lay_idx == 0 or lay_idx == len(mlp.perceptron_by_layer) - 1]
        
        x = dist_x/2 + dist_x * lay_idx
        dist_y = Ly / mlp.perceptron_by_layer[lay_idx]

        for percept_idx in range(mlp.perceptron_by_layer[lay_idx]) :
            y = dist_y/2 + dist_y * percept_idx
            perceptrons.append([x, y])
            
            if layers != [] : # evite la premiere couche
                lay_color = ('purple','orange')[lay_idx%2] #change de couleur une fois sur deux
                for i in range(len(layers[-1])) : #on prends les neuronnes de la couche precedente
                    r = layers[-1][i]
                    lay_label = str(round(mlp.layers[lay_idx-1].weights[percept_idx][i], 2))
                    plt.text((x+r[0])/2,(y+r[1])/2, lay_label)
                    plt.plot([x,r[0]], [y, r[1]], color = lay_color, label = lay_label, linewidth=2, alpha=1) #, 'go-' pour que ca fasse des points
            
            #circle to plot (gca = "get current axis")
            if layers != [] :
                percept_label = str(round(mlp.layers[lay_idx-1].bias[percept_idx], 2))
                plt.text(x,y, percept_label)
            percept_cercle = plt.Circle((x, y), radius=0.25, color=percept_color, label= "perceptron "+str(percept_idx))
            plt.gca().add_artist(percept_cercle)
            
        layers.append(perceptrons)
    plt.show()
    
