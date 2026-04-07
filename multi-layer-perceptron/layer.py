import numpy as np

class Layer :
    def __init__(self, input_size, output_size, activation_fonction, lr = 0.1) :
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) # les poids (des entiers random) une matrice de taille entrée x sortie
        self.bias = np.random.randn(output_size) # le biais (des random)
        self.lr = lr # taux d aprentissage
        self.activation_fonction = activation_fonction
    
    def sigmoid(self, x) : # Sigmoïde qui va de 0 a 1
        cc = np.array(x, dtype=np.float128) # pour eviter les probles avec numpy
        sig = 1/((1+np.exp(-cc)))
        return sig
    
    def sigmoid_deriv(self, x): # dérivée de la fonction sigmoïde
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def tanh(self, x) : # tangente hyperbolique
        return np.tanh(x)
      
    def tanh_deriv(self, x) : # derivée de la tangente hyper bolique
        return 1 - np.tanh(x)**2

    def relu(seld, x) :
        return np.maximum(np.zeros(len(x)), x)

    def relu_deriv(self, x):
        # Create a copy of x to avoid modifying the original array directly
        dx = np.copy(x)
        # Set elements <= 0 to 0
        dx[x <= 0] = 0
        # Set elements > 0 to 1
        dx[x > 0] = 1
        return dx

    def forward(self, inputs) : # on calcul le resultat en sortie de couche
        self.last_inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        if self.activation_fonction == 'sigmoid' :
          self.output = self.sigmoid(self.z) 
        elif self.activation_fonction == 'tanh' :
          self.output = self.tanh(self.z) 
        else :
          self.output = self.relu(self.z) 
        return self.output
    
    def backward(self, d_out) : # on revient en arrière pour modifier les poids et biais (d_out dérivée de la perte par rapport à la sortie de cette couche)
        if self.activation_fonction == 'sigmoid' :
          delta = d_out * self.sigmoid_deriv(self.z)  # élément par élément
        elif self.activation_fonction == 'tanh' :
          delta = d_out * self.tanh_deriv(self.z)
        else :
          delta = d_out * self.relu_deriv(self.z)

        # gradients des poids et biais
        dW = np.outer(delta, self.last_inputs)  # produit extérieur si delta est (sortie,) et last_inputs est (entrée,)
        db = np.sum(delta, axis=0)

         # update les poids
        self.weights -= self.lr * dW
        self.bias -= self.lr * db

        # gradient à propager à la couche précédente
        d_input = np.dot(self.weights.T, delta)
        return d_input