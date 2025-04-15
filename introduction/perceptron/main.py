import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from utils import generate_data, plot_classification, accuracy

############################### paramètres ###############################

#constante de numérisation :
Nt = 100 #nombre de prediction
Nx = 100 #nombre d'entrée

#constante du probleme
g = lambda x: x  # frontière de séparation
input_size = 2 # 2 car (x,y)

############################### Main ###############################

# creer notre perceptron
perceptron = Perceptron(input_size)

#initialisation des points aléatoires et du bon résultat a comparer avec nos prédictions 
x, y, inputs, z_reel = generate_data(Nx, g) # on veut regarder si le point (x,y) est au dessus de (x,g(x))

# entraine notre perceptron
prediction_for_animation = perceptron.train(inputs, z_reel, Nt)

# Prédictions finales
z_pred = np.array([perceptron.predict(inputs[i]) for i in range(Nx)]) #donne des valeurs entre 0 et 1
z_pred_int = np.array([perceptron.predict_label(inputs[i]) for i in range(Nx)]) #donne soit 1 soit 0

#precision de notre perceptron
print(f"Précision : {accuracy(z_reel, z_pred_int) * 100:.2f}%")

# Affichage des résultats finaux
plot_classification(x, y, z_pred, g, title="Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")
