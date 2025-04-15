import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
from utils import generate_data, plot_classification, accuracy

############################### paramètres ###############################

# constante de numérisation :
Nt = 200 #nombre de prediction
Nx = 500 #nombre d'entrée

# constante du probleme
g = lambda x: np.exp(x)/4  # frontière de séparation
input_size = 2 # 2 car (x,y)
hide_layer_size = 3
output_size = 1

############################### Main ###############################

# creer notre reseau de neuronne simple (MLP)
mlp = MLP([input_size, hide_layer_size, output_size])

# initialisation des points aléatoires et du bon résultat a comparer avec nos prédictions 
x, y, inputs, z_true = generate_data(Nx, g) # on veut regarder si le point (x,y) est au dessus de (x,g(x))

# entraine notre perceptron
mlp.train(inputs, z_true, Nt)

# Prédictions finales
z_pred = np.array([mlp.forward(inputs[i]) for i in range(Nx)]) # donne des valeurs entre 0 et 1
z_pred_int =  np.array([mlp.predict_label(inputs[i]) for i in range(Nx)]) # donne soit 1 soit 0

# precision de notre perceptron
print(f"Précision : {accuracy(z_true, z_pred_int) * 100:.2f}%")

# Affichage des résultats finaux
plot_classification(x, y, z_pred, g, title="Prédictions du perceptron (bleu = au dessus, rouge = en dessous)")
