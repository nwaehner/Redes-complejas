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

