import numpy as np
import matplotlib.pyplot as plt

class Layer :
    def __init__(self, input_size, output_size, lr = 0.1) :
        self.weights = np.random.randn(output_size, input_size) # les poids (des entiers random) une matrice de taille entrée x sortie
        self.bias = np.random.randn(output_size) # le biais (des random)
        self.lr = lr # taux d aprentissage
    
    def sigmoid(self, x) : # Sigmoïde qui va de 0 a 1
        return 1/(1+ np.exp(-x))
    
    def forward(self, inputs) :
        self.last_input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.a = self.sigmoid(self.z)  # ou relu(self.z)
        return self.a