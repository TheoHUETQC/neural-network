import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

############################### paramètres ###############################

#constante de numérisation :
Nt = 100 #nombre de prediction
Nx = 100 #nombre d'entrée

#constante du problème :
xmin, xmax = 0, 10
ymin, ymax = 0, 10

#pour l'aniamtion :
animation_interval = 5  #Intervalle pour l'animation (tout les combiens de step on sauvegarde les données)
save_animation = False #si on sauvegarde l'animation sur la machine
save_frames = True  #si on fait une annimation

############################### fonctions ###############################

def g(x) :
    return x

def animate_wavefunction_2D(prediction_for_animation):
    fig, ax = plt.subplots()
    extentX = [xmin, xmax]
    extentY = [ymin, ymax]
    extent = extentX + extentY

    # Création de l'image pour l'animation
    im = ax.imshow(prediction_for_animation[0], cmap="inferno", extent=extent, origin="lower", alpha=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")

    def update(frame_index):
        im.set_array(prediction_for_animation[frame_index])
        ax.set_title(f"Prédictions du perceptron Frame {frame_index}/{len(prediction_for_animation)-1}")
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(prediction_for_animation), interval=50)
    plt.colorbar(im, label="prediction")

    if save_animation:
        ani.save("perceptron.mp4", writer="ffmpeg", fps=20)
        plt.close(fig)
    else:
        plt.show()

############################### Class ###############################

# Création d'une grille 2D pour afficher la prédiction sur tout l'espace
x_grid = np.linspace(xmin, xmax, Nx)
y_grid = np.linspace(ymin, ymax, Nx)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_inputs = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (Nx*Nx, 2)

class Perceptron :
    def __init__(self, input_size, lr=0.1) :
        self.weights = np.random.randn(input_size) # les poids (des entiers random)
        self.bias = np.random.randn() #le biais (des random)
        self.lr = lr #taux d aprentissage
        
    def f(self, x) : # Sigmoïde qui va de 0 a 1
        return 1/(1+ np.exp(-x))
    
    def predict(self, x) : # Fonction de prédiction
        return self.f(np.dot(x, self.weights) + self.bias)
    
    def predict_label(self, x): #donne une valeur 0 ou 1
        return int(self.predict(x) > 0.5)

    def train(self, inputs, z_reel, Nt) : # Apprentissage
        #initialisation pour l'annimation
        prediction_for_animation = []
        pred_for_animation = []
        for t in range(Nt) :
            for i in range(len(inputs)) :
                #calculs
                pred = self.predict(inputs[i])
                error = z_reel[i] - pred # calcul de l'erreur entre la prédiction et le bon résultat
                # ajustements des parametres
                self.weights = self.weights + self.lr * error * inputs[i] # Mise à jour des poids
                self.bias = self.bias + self.lr * error # Mise à jour du biais
            if t % animation_interval == 0:
                # Prédiction sur la grille complète pour l'animation
                preds_grid = np.array([self.predict(pt) for pt in grid_inputs])
                preds_grid_2D = preds_grid.reshape(Nx, Nx)
                prediction_for_animation.append(preds_grid_2D)
                print(f"Image {t/animation_interval + 1}/{Nt/animation_interval}")
        return prediction_for_animation #pas de vrai return car on souhaite juste ajuster bias et weights pour améliorer nos futurs predictions

############################### Main ###############################

#initialisation des points aléatoires
x, y = np.random.rand(Nx) * xmax + xmin , np.random.rand(Nx) * ymax + ymin
inputs = np.stack([x, y], axis=1)

# initialisation du bon résultat a comparer avec nos prédictions 
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
prediction_for_animation = perceptron.train(inputs, z_reel, Nt)

# Prédictions finales
z_pred = np.array([perceptron.predict(inputs[i]) for i in range(Nx)]) #donne des valeurs entre 0 et 1
z_pred_int = np.array([perceptron.predict_label(inputs[i]) for i in range(Nx)]) #donne soit 1 soit 0

#precision de notre perceptron
acc = np.mean(z_pred_int == z_reel)
print(f"Précision : {acc*100:.2f}%")

# lance l'animation de l'évolution
if save_frames and len(prediction_for_animation) > 1 :
    animate_wavefunction_2D(prediction_for_animation)

# Affichage des résultats finaux
plt.scatter(x, y, c=z_pred, cmap="bwr", edgecolors="k")
plt.plot(np.linspace(xmin, xmax, 100), g(np.linspace(xmin, xmax, 100)), color="g")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.title("Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
