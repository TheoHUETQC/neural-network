import numpy as np
import matplotlib.pyplot as plt

# un Qubit logique dans plusieur Qubit physique
# ici : 1 qubit logique → 3 qubits physiques ∣0_L​⟩ = ∣000⟩ ,  ∣1_L⟩ = ∣111⟩

# On suppose des bit-flips aléatoires : X_1, X_2, X_3

# On mesure des stabilisateurs S_1 et S_2 → cela donne un syndrome = un ensemble de bits qui indique “où” se trouve l’erreur (mais sans révéler l’état logique)
# S_1 ​= Z_1 ​Z_2 ​,  S_2​ = Z_2 ​Z_3

# on corrige l'erreur

"""
 Table syndromes :
 erreur | (S1,S2) | en binaire +1→0 et -1→1 | correction | Label de l'erreur | vecteur one-hot | 
 aucune | (+1,+1) | (0,0)                   | I          | 0                 | [1,0,0,0]       |
 X_1    | (-1,+1) | (1,0)                   | X_1        | 1                 | [0,1,0,0]       |
 X_2    | (-1,-1) | (1,1)                   | X_2        | 2                 | [0,0,1,0]       |
 X_3    | (+1,-1) | (0,1)                   | X_3        | 3                 | [0,0,0,1]       |
"""
# Label pour Single-error model / vecteur one-hot permet de considerer plusieur bitflip en meme temps e.g [0,1,1,0]

# notre bute : Générer un dataset de N paires (syndrome, correction), où correction <=> label/vecteur one-hot

Nbr_Qubit = 3

# init 

# chaque resultat (S1,S2) en binaire = un label
mapping = {(0,0):0, (1,0):1, (1,1):2, (0,1):3} 

# matrice de pauli
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

#def des erreurs possibles :
I = np.eye(2**Nbr_Qubit)                     # I x I x I
X1 = np.kron(X,np.kron(np.eye(2),np.eye(2))) # X x I x I
X2 = np.kron(np.eye(2),np.kron(X,np.eye(2))) # I x X x I
X3 = np.kron(np.eye(2),np.kron(np.eye(2),X)) # I x I x X        
error_list = [I, X1, X2, X3]

#def des qubit logique :
P0 = np.array([1,0])             # qubit physique |0>
P1 = np.array([0,1])             # qubit physique |1>
L0 = np.kron(P0, np.kron(P0,P0)) # qubit logique |0L> = |000>
L1 = np.kron(P1, np.kron(P1,P1)) # qubit logique |1L> = |111>
qubit_list = [L0, L1]

#def des stabilisateurs :
S1 = np.kron(Z,np.kron(Z,np.eye(2))) # Z x Z x I
S2 = np.kron(np.eye(2),np.kron(Z,Z)) # I x Z x Z


# main 

N = 1
for n in range(N) :
    qubit = qubit_list[np.random.randint(len(qubit_list))]
    error = error_list[np.random.randint(len(error_list))]
    result = error * qubit
    S1_result = S1 * result
    S2_result = S2 * result
    print(S1,S2)
