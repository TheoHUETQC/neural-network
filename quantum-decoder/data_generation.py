import numpy as np
import matplotlib.pyplot as plt

# un Qubit logique dans plusieur Qubit physique
# Exemple : 1 qubit logique → 3 qubits physiques ∣0_L​⟩ = ∣000⟩ ,  ∣1_L⟩ = ∣111⟩

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

Nbr_Qubit = 10

# init

# chaque resultat (S1,S2,..,Sj,...Sn) en binaire = un label
"""version 3 qubit seulement :
syndrome_to_label = {(0,0):0, (1,0):1, (1,1):2, (0,1):3} """
syndrome_to_label = {}
for i in range(Nbr_Qubit+1) :
    S_list = []
    for j in range(1, Nbr_Qubit) :
        Sj = int(i==j or i==j+1)
        S_list.append(Sj)
    syndrome_to_label[str(S_list)] = i


# matrice de pauli
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

#def des erreurs possibles :
""" version 3 qubit seulement :
I = np.eye(2**Nbr_Qubit)                     # I x I x I
X1 = np.kron(X,np.kron(np.eye(2),np.eye(2))) # X x I x I
X2 = np.kron(np.eye(2),np.kron(X,np.eye(2))) # I x X x I
X3 = np.kron(np.eye(2),np.kron(np.eye(2),X)) # I x I x X        
error_list = [I, X1, X2, X3]"""
error_list = []
for i in range(Nbr_Qubit+1) : #X0 = Id
    Xi = X * int(i==1) + np.eye(2)* int(i!=1)
    for j in range(2, Nbr_Qubit+1) :
        Xi = np.kron(Xi, X * int(i==j) + np.eye(2) * int(i!=j))
    error_list.append(Xi)

#def des qubit logique :
P0 = np.array([[1],[0]])             # qubit physique |0>
P1 = np.array([[0],[1]])             # qubit physique |1>
L0, L1 = P0, P1
for i in range(Nbr_Qubit-1) :
    L0 = np.kron(L0,P0) # qubit logique |0L> = |00...0>
    L1 = np.kron(L1,P1) # qubit logique |1L> = |11...1>
qubit_list = [L0, L1]

#def des stabilisateurs :
""" version 3 qubit seulement :
S1 = np.kron(Z,np.kron(Z,np.eye(2))) # Z x Z x I
S2 = np.kron(np.eye(2),np.kron(Z,Z)) # I x Z x Z
stab_list = [S1, S2]"""
stab_list = []
for i in range(1, Nbr_Qubit) :
    S = Z * int(i==1) + np.eye(2) * int(i!=1)
    for j in range(2, Nbr_Qubit+1) :
        S = np.kron(S, Z * int(i==j or i+1==j) + np.eye(2) * int(i!=j and i+1!=j))
    stab_list.append(S)

# main 

N = 1000
for n in range(N) :
    qubit = qubit_list[np.random.randint(len(qubit_list))]
    type_derreur = np.random.randint(len(error_list))
    error = error_list[type_derreur]
    result = np.dot(error, qubit)
    
    S_eigen_list = []
    for S in stab_list :
        S_result = np.dot(S, result) # S|result>
        S_eigen = np.dot(np.transpose(np.conjugate(result)), S_result)[0][0] # on fait <result|S|result> = valeur propre de S correspondant a l etat propre S_result
        S_eigen_binary = int((S_eigen - 1)/(-2)) # = 0 si S donne +1 et = 1 si S donne -1
        S_eigen_list.append(S_eigen_binary)

    print("on a une erreur de type :", syndrome_to_label[str(S_eigen_list)], ", bonne erreur ? ", type_derreur == syndrome_to_label[str(S_eigen_list)])
