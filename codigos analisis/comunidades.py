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

with open(f"../red_filtrada/red_filtrada_difusion.gpickle", "rb") as f:
    G = pickle.load(f)
G_ig = ig.Graph.from_networkx(G)

# %%
infomap = G_ig.community_infomap() #Infomap   
louvain = com.best_partition(G) #Louvain
com_bt = G_ig.community_edge_betweenness(clusters = 18, directed = False, weights = None) #betweenness
# %%
def cluster_to_dict(cluster, g):
    dic = {}
    for i, c in enumerate(sorted(list(cluster), key = len, reverse = True)):
        for n in c:
            dic[g.vs[n]['_nx_name']] = i
    return dic
dic_betweenness = cluster_to_dict(com_bt.as_clustering(), G_ig)
dic_louvain = com.best_partition(G)
dic_infomap = cluster_to_dict(infomap, G_ig)

print(com_bt) # Cantidad de elementos y de mergeos hechos

#%% Grafico del dendrograma

# %%
#recorro los nodos por cada cluster y veo la cantidad de artistas
#por género y me construyo una matriz de confusión

generos_representativos = ["Trap","HipHop","Rap","Rock","Pop","Alternative","R&B","Folklore","Tango", "Cumbia",
                           "Jazz", "Clasico",
                           "Chamame", "Electronica",  "Punk",  "Metal",
                           "Reggae"]
orden = [0,5,17,9,12,13,7,8,26,15,3,1,21,25,2,10,4,6,14,11,16,18,19,20,22,23,24]
#pruebo con louvain
clusters = max(dic_betweenness.values())  #cantidad de clusteres
# %%
matriz_confusion = np.zeros((clusters,len(generos_representativos)))
for cluster in range(clusters):
    # indice = orden[cluster]
    nodos_en_cluster = [nodo for (nodo, value) in dic_betweenness.items() if value == cluster] 
    for artista in nodos_en_cluster:
        G.nodes()[artista]['betweenness'] = cluster
        if len(G.nodes()[artista]['generos_musicales'])>0:
            for genero in G.nodes()[artista]['generos_musicales']:
                matriz_confusion[cluster,generos_representativos.index(genero)] +=1
        # else:
        #     matriz_confusion[cluster,len(generos_representativos)] +=1

# %% dibujo la matriz de confusion
fig, ax = plt.subplots(1,1, figsize=(8,8))
img = ax.imshow(matriz_confusion,interpolation='none',cmap='RdPu')
for (i, j), z in np.ndenumerate(matriz_confusion):
    ax.text(j, i, int(z), ha='center', va='center')
ax.set_xticks(np.arange(len(generos_representativos))) 
ax.set_yticks(np.arange(clusters))
matriz_confusion.sort(axis=- 1, kind='stable', order=None)
#ax.set_ylim(0,8)
ax.set_xticklabels(generos_representativos, rotation = 90)
ax.set_ylabel("Comunidades por cluster", fontsize=12)
ax.set_xlabel("Géneros Musicales", fontsize=12)
plt.savefig("../imagenes del analisis/matriz de betweenness.png")
plt.show()
# %% dibujo la matriz de confusion
fig, ax = plt.subplots(1,1, figsize=(8,8))
matriz_confusion_reducida = matriz_confusion[0:8]
img = ax.imshow(matriz_confusion_reducida,interpolation='none',cmap='RdPu')
for (i, j), z in np.ndenumerate(matriz_confusion_reducida):
    ax.text(j, i, int(z), ha='center', va='center')
ax.set_xticks(np.arange(len(generos_representativos))) 
ax.set_yticks(np.arange(8))
matriz_confusion_reducida.sort(axis=- 1, kind='stable', order=None)
ax.set_xticklabels(generos_representativos, rotation = 90)
ax.set_ylabel("Comunidades por cluster", fontsize=12)
ax.set_xlabel("Géneros Musicales", fontsize=12)
plt.savefig("../imagenes del analisis/matriz de betweenness reducida.png")
plt.show()
# %%
pickle.dump(G, open(f'../red_filtrada/red_filtrada_comunidades_betwenness.gpickle', 'wb'))
# %%
# %%

wc_atributos = {'height' : 800,
                'width' : 1200,
                'background_color' : 'white',
                'max_words' : 20
                } 
labels = dic_betweenness.values()
fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (12,5), facecolor='None')
for i, ax in enumerate(fig.axes):
    index = np.nonzero(labels==i)[0]      
    artistas_por_cluster = [artista for artista in dic_betweenness.keys() if dic_betweenness[artista] == i]
    g = []
    for artista in artistas_por_cluster:
        g.extend(G.nodes()[artista]["generos_musicales"])
    dict_generos_del_cluster = {i:g.count(i) for i in set(g)}
    grados_por_cluster = [G.degree(nodo) for nodo in artistas_por_cluster]
    dict_artistas_del_cluster =dict(zip(artistas_por_cluster,grados_por_cluster))
    wc = WordCloud(**wc_atributos).generate_from_frequencies(dict_generos_del_cluster)
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title("Comunidad " + str(i), fontsize=20, color = "white")

plt.savefig("../imagenes del analisis/worcloud por genero.png")
# %%

wc_atributos = {'height' : 800,
                'width' : 1200,
                'background_color' : 'white',
                'max_words' : 15
                } 
labels = dic_betweenness.values()
fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (12,5), facecolor="None")

for i, ax in enumerate(fig.axes):
    index = np.nonzero(labels==i)[0]      
    artistas_por_cluster = [artista for artista in dic_betweenness.keys() if dic_betweenness[artista] == i]
    g = []
    for artista in artistas_por_cluster:
        g.extend(G.nodes()[artista]["generos_musicales"])
    dict_generos_del_cluster = {i:g.count(i) for i in set(g)}
    grados_por_cluster = [G.degree(nodo) for nodo in artistas_por_cluster]
    dict_artistas_del_cluster =dict(zip(artistas_por_cluster,grados_por_cluster))
    wc = WordCloud(**wc_atributos).generate_from_frequencies(dict_artistas_del_cluster)
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title("Comunidad " + str(i), fontsize=20, color = "white")

plt.savefig("../imagenes del analisis/worcloud por artista.png")

# %%
