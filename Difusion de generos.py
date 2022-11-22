#%% 
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation

#%%-----------------------Cargamos el multigrafo---------------------------
i = 1496

with open(f"red_final/Iteracion {i}/red_final_hasta_indice_{i}.gpickle", "rb") as f:
    G = pickle.load(f)

matriz_adyacencia = nx.to_pandas_adjacency(G).to_numpy()
#%%
genero = "vintagetango"
vector_generos = []

for nodo in G.nodes():
    lista_generos_nodo = G.nodes()[nodo]["generos_musicales"] 
    if genero in lista_generos_nodo:
        vector_generos.append(1)
    elif len(lista_generos_nodo) == 0:
        vector_generos.append(-1)
    else:
        vector_generos.append(0)
        
#%%

label_prop_model = LabelPropagation()
label_prop_model.fit(matriz_adyacencia, vector_generos)
prediccion = label_prop_model.predict_proba(matriz_adyacencia)

print(prediccion)
# %%
