import numpy as np
import matplotlib.pyplot as plt

class Perceptron :
    def __init__(self, input_size, lr=0.1) :
        self.weights = np.random.randn(input_size) # les poids (des entiers random)
        self.bias = np.random.randn() #le biais (des random)
        self.lr = lr #taux d aprentissage
        
    def f(self, x) : # Sigmoïde qui va de 0 a 1
        return 1/(1+ np.exp(-x))
    
    def predict(self, x) : # Fonction de prédiction
        return self.f(np.dot(x, self.weights) + self.bias)
    
    def predict_label(self, x) : # donne une valeur 0 ou 1
        return int(self.predict(x) > 0.5)
    
    def compute_loss(self, inputs, z_reel) : # fonction de cout
        preds = np.array([self.predict(inputs[i]) for i in range(len(inputs))])
        errors = z_reel - preds
        return np.mean((errors)**2) # erreur quadratique moyenne (MSE)

    def train(self, inputs, z_reel, epochs) : # Apprentissage
        for epoch in range(epochs) :
            for i in range(len(inputs)) :
                #calculs
                pred = self.predict(inputs[i])
                error = z_reel[i] - pred # calcul de l'erreur entre la prédiction et le bon résultat
                # ajustements des parametres
                self.weights = self.weights + self.lr * error * inputs[i] # Mise à jour des poids
                self.bias = self.bias + self.lr * error # Mise à jour du biais

            if epoch % 10 == 0 :
                loss = self.compute_loss(inputs, z_reel)
                print(f"Epoch {epoch}, loss = {loss:.4f}")

        #pas de vrai return car on souhaite juste ajuster bias et weights pour améliorer nos futurs predictions
