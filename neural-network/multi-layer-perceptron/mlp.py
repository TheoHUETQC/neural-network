import numpy as np
import matplotlib.pyplot as plt
from layer import Layer

class MLP :
    def __init__(self) :
        self.layers = []
        self.perceptron_by_layer = []

    def add(self, output_dim : int, input_dim : int=-1, activation : str='relu', lr : float=0.1) :
        if input_dim == -1 :
            input_dim = self.layers[-1].output_size
        else :
            self.perceptron_by_layer.append(input_dim)
        self.perceptron_by_layer.append(output_dim)
        self.layers.append(Layer(input_dim, output_dim, activation, lr))

    def forward(self, output) : #on parcours les couches
        for layer in self.layers :
            output = layer.forward(output) # on avance a la couche d apres
        return output
    
    def backward(self, d_input) :
        for layer in reversed(self.layers) :
            d_input = layer.backward(d_input)

    def train(self, inputs, z_true, epochs) :
        self.loss = [] #pour sauvegarder la perte moyenne par époque
        for epoch in range(epochs) :
            loss_epoch = 0
            for i in range(len(inputs)) :
                # prediction
                z_pred = self.forward(inputs[i])
                # regarde la difference de sortie
                d_out = (z_pred - z_true[i])
                # ajustement des parametres en revenant en arrière
                self.backward(d_out)
                #calculs de la perte moyenne
                loss_epoch += self.binary_cross_entropy(z_true[i], z_pred)
            self.loss.append(loss_epoch / len(inputs))

            if epoch % 10 == 0 or epoch == epochs - 1 : #pour ne pas afficher a chaque pas de temps
                print(f"Epoch {epoch+1}/{epochs} - Loss: {self.loss[-1]:.4f}")

    def binary_cross_entropy(self, z_true, z_pred) : # Calculer la perte (coût) Pour un problème de classification binaire 
        epsilon = 1e-8  # éviter log(0)
        return -np.mean(z_true * np.log(z_pred + epsilon) + (1 - z_true) * np.log(1 - z_pred + epsilon))

    def predict(self, inputs): #donne des valeurs entre 0 et 1
        z_pred = self.forward(inputs)
        return z_pred
    
    def predict_label(self, input): # donne soit 1 soit 0
        return int(self.forward(input) > 0.5)
    
    def plot_history(self) :
      plt.figure()
      plt.title("loss")
      plt.plot(self.loss)
      plt.show()