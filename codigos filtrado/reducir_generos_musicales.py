#%%
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
from unicodedata import normalize
from wordcloud import WordCloud
#%%
with open('../red_filtrada/red_filtrada_difusion.gpickle', "rb") as f:
    G = pickle.load(f)
# %%
lista_nodos = []
lista_generos_por_nodo = []
for nodo in G.nodes():
    generos = G.nodes()[nodo]["generos_musicales"]
    if generos != []:
        lista_generos_por_nodo.append(G.nodes()[nodo]["generos_musicales"])
        lista_nodos.append(nodo)
    d = {'generos':lista_generos_por_nodo}
df = pd.DataFrame(d, index = lista_nodos)

# %% Aplico One-Hot-Encoder
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df_encoded = pd.DataFrame(mlb.fit_transform(df['generos']), columns=mlb.classes_, index=df.index)
# %%
# Clase para realizar componentes principales
from sklearn.decomposition import PCA
#importamos el algoritmo de K-means
from sklearn.cluster import KMeans 
# importamos el puntaje de silhouette
from sklearn.metrics import silhouette_score
#%%
pca = PCA()
pca.fit(df_encoded)
X_pca = pca.transform(df_encoded)
#%%
# Creamos una lista para guardar de los coeficientes de silhouette para cada valor de k
silhouette_coefficients = []

# Se necesita tener al menos 2 clusters y a los sumo N-1 (con N el numero de muestras) para obtener coeficientes de Silohuette
for k in range(2, 20):
     kkkmeans = KMeans(n_clusters=k)
     kkkmeans.fit(X_pca)
     score = silhouette_score(X_pca, kkkmeans.labels_)
     silhouette_coefficients.append(score)
#%%
fig, ax = plt.subplots(figsize = (12, 8))

# estas lineas son el grafico de SSEvsK
ax.plot(range(2, 20), silhouette_coefficients, '--o', c='red')            
ax.set_xticks(range(2, 20))
ax.set_xlabel("Número de clusters",fontsize = 16)
ax.grid(20)
ax.set_ylabel("Promedio coeficientes de Silhouette",fontsize = 16)

# Guardo las posiciones de los centroids
kmeans = KMeans(n_clusters=8)   #k=8
kmeans.fit(X_pca)
centroids = kmeans.cluster_centers_
#%%

fig, ax = plt.subplots(figsize = (12, 8))

# Hacemos un scatter plot de cada uno de los datos
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
#ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, linewidths=1,
#            c=np.unique(kmeans.labels_), edgecolors='black')
ax.grid(True)
ax.legend()
ax.set_xlabel('Primer componente principal')
ax.set_ylabel('Segunda componente principal')
plt.show()
# %%
artistas_por_cluster = dict(zip(df.index,kmeans.labels_))
wc_atributos = {'height' : 800,
                'width' : 1200,
                'background_color' : 'black',
                'max_words' : 20
                } 
labels = kmeans.labels_
fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (12,5), facecolor='#f4e8f4')
for i, ax in enumerate(fig.axes):
    index = np.nonzero(labels==i)[0]      
    artistas_por_cluster = np.array(df.index)[index]
    g = []
    for artista in artistas_por_cluster:
        g.extend(G.nodes()[artista]["generos_musicales"])
    dict_generos_del_cluster = {i:g.count(i) for i in set(g)}
    grados_por_cluster = [G.degree(nodo) for nodo in artistas_por_cluster]
    dict_artistas_del_cluster =dict(zip(artistas_por_cluster,grados_por_cluster))
    wc = WordCloud(**wc_atributos).generate_from_frequencies(dict_generos_del_cluster)
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title("Cluster " + str(i), fontsize=20)
# %%
