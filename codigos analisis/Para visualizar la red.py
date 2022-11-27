#%%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 
import pickle 
from tqdm import tqdm
import forceatlas2 as fa2
plt.style.use("seaborn")
#%%-----------------------Cargamos el multigrafo---------------------------

with open(f"../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)
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
