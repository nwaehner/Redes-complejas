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

matriz_adyacencia = nx.adjacency_matrix(G)
#%%
genero = "vintagetango"
vector_generos = []
for nodo in G.nodes():


vector_generos = [G.nodes(nodo)["generos_musicales"] for nodo in G.nodes()]


#%%


label_prop_model = LabelPropagation()
iris = datasets.load_iris()
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels = np.copy(iris.target)
labels[random_unlabeled_points] = -1
label_prop_model.fit(iris.data, labels)


# %%
