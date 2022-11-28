#%%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 
import pickle 
from tqdm import tqdm
from fa2 import ForceAtlas2
plt.style.use("seaborn")
#%%-----------------------Cargamos el multigrafo---------------------------

with open(f"../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

#%%--------------Calculando la distancia media entre los nodos---------------
dist_promedio = nx.average_shortest_path_length(G,weighted = True)
print(f"La distancia promedio entre los nodos es {dist_promedio}")
# %%
grado = nx.degree(G) #Definimos el grado de la red
lista_nodos = [v[0] for v in grado] # Definimos una lista con los nodos para pasarsela al gráfico junto al tamaño.
tamaño_nodos = [v[1]*3 for v in grado] #definimos el tamaño del nodo como su grado por un factor 3 (factor decidido para mejorar la visualización)

componentes =[G.subgraph(componente) for componente in sorted(nx.connected_components(G), key=len, reverse=True)]  # Ahora aplicamos el algoritmo a las no dirigidas
arbol = nx.minimum_spanning_tree(G, algorithm='kruskal')
#%%
plt.figure(figsize = (10,8))
nx.draw_kamada_kawai(arbol,width = 0.4, alpha = 0.5, nodelist = lista_nodos,node_size=tamaño_nodos,node_color = 'g')
plt.show()
# %%
forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)

positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
nx.draw_networkx_nodes(G, positions, node_size=20, with_labels=False, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.05)
plt.axis('off')
plt.show()

# %%
