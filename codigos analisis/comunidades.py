#%%
import networkx as nx
import musicbrainzngs as mb 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import igraph as ig
from community import community_louvain as com
from wordcloud import WordCloud
#%%-----------------------Cargamos el multigrafo---------------------------

with open(f"../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)
G_ig = ig.Graph.from_networkx(G)

# %%
infomap = G_ig.community_infomap() #Infomap   
louvain = com.best_partition(G) #Louvain
# %%
def cluster_to_dict(cluster, g):
    dic = {}
    for i, c in enumerate(sorted(list(cluster), key = len, reverse = True)):
        for n in c:
            dic[g.vs[n]['_nx_name']] = i
    return dic

dic_louvain = com.best_partition(G)
dic_infomap = cluster_to_dict(infomap, G_ig)
# %%
#recorro los nodos por cada cluster y veo la cantidad de artistas
#por género y me construyo una matriz de confusión

generos_representativos = ["Trap","Jazz", "Pop", "HipHop", "Clasico", "Indie",
                           "R&B", "Tango", "Cumbia",
                           "Chamame", "Electronica", "Folklore", "Rock", "Punk", "Rap", "Metal",
                           "Reggae","Alternative"]
#pruebo con louvain
clusters = max(dic_louvain.values())  #cantidad de clusteres
# %%
matriz_confusion = np.zeros((clusters,len(generos_representativos)))
for cluster in range(clusters):
    nodos_en_cluster = [nodo for (nodo, value) in dic_louvain.items() if value == cluster] 
    for artista in nodos_en_cluster:
        if len(G.nodes()[artista]['generos_musicales'])>0:
            for genero in G.nodes()[artista]['generos_musicales']:
                matriz_confusion[cluster,generos_representativos.index(genero)] +=1

# %% dibujo la matriz de confusion
fig, ax = plt.subplots(1,1, figsize=(8,8))
img = ax.imshow(matriz_confusion,interpolation='none',cmap='RdPu')
for (i, j), z in np.ndenumerate(matriz_confusion):
    ax.text(j, i, int(z), ha='center', va='center')
ax.set_xticks(np.arange(len(generos_representativos))) 
ax.set_yticks(np.arange(clusters))

ax.set_xticklabels(generos_representativos, rotation = 90)
ax.set_ylabel("Comunidades por cluster", fontsize=12)
ax.set_xlabel("Géneros Musicales", fontsize=12)
plt.show()
# %%
