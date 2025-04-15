import numpy as np
import matplotlib.pyplot as plt
from layer import Layer

class MLP :
    def __init__(self, output_input_size = [2,3,1]) :
        self.layers = []
        for i in range(len(output_input_size) - 1) :
            self.layers.append(Layer(output_input_size[i], output_input_size[i+1]))
            print(self.layers[i].weights)
    
    def forward(self, x): #on parcours les couches
        for i in range(len(self.layers)) :
            x = self.layers[i].forward(x) #on avance a la couche d apres
        return x