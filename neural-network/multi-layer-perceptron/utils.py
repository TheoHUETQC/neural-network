import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_data(Nx, frontiere, xmin=0, xmax=10, ymin=0, ymax=10): # on veut regarder si le point (x,y) est au dessus de (x,g(x))
    #initialisation des points aléatoires
    x, y = np.random.rand(Nx) * (xmax - xmin) + xmin, np.random.rand(Nx) * (ymax - ymin) + ymin
    inputs = np.stack([x, y], axis=1)

    # initialisation du bon résultat a comparer avec nos prédictions 
    z_true = (frontiere(x) > y).astype(int) # Classe 1 si y > x, sinon 0

    return x, y, inputs, z_true

def plot_classification(x, y, z, frontiere, xmin=0, xmax=10, ymin=0, ymax=10, title=""):
    plt.scatter(x, y, c=z, cmap="bwr", edgecolors="k")
    plt.plot(np.linspace(xmin, xmax, 100), frontiere(np.linspace(xmin, xmax, 100)), color="g")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()
    
def accuracy(y_true, y_pred): #precision de notre perceptron
    return np.mean(y_true == y_pred)

def mlp_matplotlib_view(mlp) : # ancienne version que je garde permettant de plot un schéma du reseau de neuronne avec matplotlib
    Lx, Ly = 10, 10
    dist_x = Lx / len(mlp.perceptron_by_layer)
    
    plt.figure()
    layers = []
    
    for lay_idx in range(len(mlp.perceptron_by_layer)) :
        perceptrons = []
        
        percept_color = ('blue', 'red')[lay_idx == 0 or lay_idx == len(mlp.perceptron_by_layer) - 1]
        
        x = dist_x/2 + dist_x * lay_idx
        dist_y = Ly / mlp.perceptron_by_layer[lay_idx]

        for percept_idx in range(mlp.perceptron_by_layer[lay_idx]) :
            y = dist_y/2 + dist_y * percept_idx
            perceptrons.append([x, y])
            
            if layers != [] : # evite la premiere couche
                lay_color = ('purple','orange')[lay_idx%2] #change de couleur une fois sur deux
                for i in range(len(layers[-1])) : #on prends les neuronnes de la couche precedente
                    r = layers[-1][i]
                    lay_label = str(round(mlp.layers[lay_idx-1].weights[percept_idx][i], 2))
                    plt.text((x+r[0])/2,(y+r[1])/2, lay_label)
                    plt.plot([x,r[0]], [y, r[1]], color = lay_color, label = lay_label, linewidth=2, alpha=1) #, 'go-' pour que ca fasse des points
            
            #circle to plot (gca = "get current axis")
            if layers != [] :
                percept_label = str(round(mlp.layers[lay_idx-1].bias[percept_idx], 2))
                plt.text(x,y, percept_label)
            percept_cercle = plt.Circle((x, y), radius=0.25, color=percept_color, label= "perceptron "+str(percept_idx))
            plt.gca().add_artist(percept_cercle)
            
        layers.append(perceptrons)
    plt.show()

def mlp_networkx_view(mlp, filename="graphic_view_mlp.png"):
    G = nx.DiGraph()
    positions = {}
    colors = []
    labels = {}
    layer_sizes = mlp.perceptron_by_layer
    x_spacing = 2
    y_spacing = 1.5

    color_map = {
        'input': 'skyblue',
        'hidden': 'orange',
        'output': 'lightgreen',
        'bias': 'lightgray'
    }

    node_types = {}
    node_count = 0
    node_per_layer = []
    weight_labels = []

    for layer_idx, layer_size in enumerate(layer_sizes):
        nodes = []
        x = layer_idx * x_spacing
        y_offset = (max(layer_sizes) - layer_size) * y_spacing / 2
        for neuron_idx in range(layer_size):
            y = neuron_idx * y_spacing + y_offset
            node_name = f"N{node_count}"
            positions[node_name] = (x, y)
            G.add_node(node_name)
            nodes.append(node_name)
            labels[node_name] = ''
            if layer_idx == 0:
                node_types[node_name] = 'input'
            elif layer_idx == len(layer_sizes) - 1:
                node_types[node_name] = 'output'
            else:
                node_types[node_name] = 'hidden'
            node_count += 1
        node_per_layer.append(nodes)

    for layer_idx, layer in enumerate(mlp.layers):
        prev_nodes = node_per_layer[layer_idx]
        curr_nodes = node_per_layer[layer_idx + 1]

        for j, target_node in enumerate(curr_nodes):
            bias_node = f"B{target_node}"
            x_bias = positions[target_node][0] - 0.5
            y_bias = positions[target_node][1] + 0.5
            G.add_node(bias_node)
            positions[bias_node] = (x_bias, y_bias)
            G.add_edge(bias_node, target_node)
            labels[bias_node] = f"b = {layer.bias[j]:.2f}"
            node_types[bias_node] = 'bias'

            for i, source_node in enumerate(prev_nodes):
                G.add_edge(source_node, target_node)
                weight = layer.weights[j][i]
                x1, y1 = positions[source_node]
                x2, y2 = positions[target_node]
                x_text = (x1 + x2) / 2
                y_text = (y1 + y2) / 2 + 0.15 * (i - len(prev_nodes)/2)
                weight_labels.append((x_text, y_text, f"w = {weight:.2f}"))

    for node in G.nodes():
        colors.append(color_map[node_types[node]])

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, positions, node_color=colors, node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, positions, arrowstyle='->', arrowsize=10, width=1.5)
    nx.draw_networkx_labels(G, positions, labels=labels, font_size=8)

    for (x, y, label) in weight_labels:
        plt.text(x, y, label, fontsize=7, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['input'], edgecolor='black', label='Entrée'),
        Patch(facecolor=color_map['hidden'], edgecolor='black', label='Cachée'),
        Patch(facecolor=color_map['output'], edgecolor='black', label='Sortie'),
        Patch(facecolor=color_map['bias'], edgecolor='black', label='Biais')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
