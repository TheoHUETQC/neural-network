import numpy as np
import matplotlib.pyplot as plt

class Layer :
    def __init__(self, input_size, output_size, lr = 0.1) :
        self.weights = np.random.randn(output_size, input_size) # les poids (des entiers random) une matrice de taille entrée x sortie
        self.bias = np.random.randn(output_size) # le biais (des random)
        self.lr = lr # taux d aprentissage
    
    def sigmoid(self, x) : # Sigmoïde qui va de 0 a 1
        return 1/(1+ np.exp(-x))
    
    def sigmoid_deriv(self, x): #dérivée de la fonction sigmoïde
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, inputs) : # on calcul le resultat en sortie de couche
        self.last_inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self.sigmoid(self.z) 
        return self.output
    
    def backward(self, d_out) : # on revient en arrière pour modifier les poids et biais (d_out dérivée de la perte par rapport à la sortie de cette couche)
        delta = d_out * self.sigmoid_deriv(self.z)  # élément par élément

        # gradients des poids et biais
        dW = np.outer(delta, self.last_inputs)  # produit extérieur si delta est (sortie,) et last_inputs est (entrée,)
        db = np.sum(delta, axis=0)

         # update les poids
        self.weights -= self.lr * dW
        self.bias -= self.lr * db

        # gradient à propager à la couche précédente
        d_input = np.dot(self.weights.T, delta)
        return d_input