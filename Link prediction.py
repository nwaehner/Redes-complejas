#%%
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec as n2v

#Librerias ari:

# selección de modelos
# clasificador: red neuronal (perceptron multicapa)
from sklearn.neural_network import MLPClassifier
# clasificador: support vector classifier
from sklearn.svm import SVC
# clasificador: ensemble de árboles (random forest)
from sklearn.ensemble import RandomForestClassifier

#GridSearchCV  paraajuste de hiperparámetros con cross-validation
from sklearn.model_selection import GridSearchCV



###     ¿ QUE MEDIDA DE DISTANCIA USAR? COSENO TIENE SENTIDO? 
#        preguntar por el algoritmo de machine learning y que hace. 
#         que nos dice MCC score?
#        ¿Esta bien predecir links segun la similaridad?
#       ¿Como encontrar la cant de dimensiones optimas?

#%%-----------------------Cargamos el multigrafo---------------------------
with open(f"red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

    
#%% NODE TO VEC:

g_emb = n2v(G, dimensions=16)
WINDOW = 1 # Node2Vec fit window
MIN_COUNT = 1 # Node2Vec min. count
BATCH_WORDS = 4 # Node2Vec batch words

mdl = g_emb.fit(
    window=WINDOW,
    min_count=MIN_COUNT,
    batch_words=BATCH_WORDS
)

# create embeddings dataframe
emb_df = (
    pd.DataFrame(
        [mdl.wv.get_vector(str(n)) for n in G.nodes()],
        index = G.nodes
    )
)



# %%
def predict_links(G, df, article_id):
    
    # separate target article with all others
    article = df[df.index == article_id]
    
    # other articles are all articles which the current doesn't have an edge connecting
    all_nodes = G.nodes()
    other_nodes = [n for n in all_nodes if n not in list(G.adj[article_id]) + [article_id]]
    other_articles = df[df.index.isin(other_nodes)]
    
    # get similarity of current reader and all other readers
    sim = cosine_similarity(article, other_articles)[0].tolist()
    idx = other_articles.index.tolist()
    
    # create a similarity dictionary for this user w.r.t all other users
    idx_sim = dict(zip(idx, sim))
    idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=True)

    similar_articles = idx_sim
    articles = [art[0] for art in similar_articles]
    similaridad_articles = [art[1] for art in similar_articles]
    return articles,similaridad_articles
#%%
nodos_recorridos = []
similaridades_no_conectados = []

for i in list(G.nodes()):
    nodos_similares,similaridad = predict_links(G, emb_df, i)
    nodos_recorridos.append(i)
    #Hago esto para no agregar similaridad de nodos dos veces.
    for j,nodo in enumerate(nodos_similares):
        if nodo not in nodos_recorridos:
            similaridades_no_conectados.append(similaridad[j])

# %%
def predict_links_conectados(G, df, article_id):
    
    # separate target article with all others
    article = df[df.index == article_id]
    
    # other articles are all articles which the current doesn't have an edge connecting
    all_nodes = G.nodes()
    other_nodes = [n for n in all_nodes if n in list(G.adj[article_id])]
    other_articles = df[df.index.isin(other_nodes)]
    
    # get similarity of current reader and all other readers
    sim = cosine_similarity(article, other_articles)[0].tolist()
    idx = other_articles.index.tolist()
    
    # create a similarity dictionary for this user w.r.t all other users
    idx_sim = dict(zip(idx, sim))
    idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=True)

    # similar_articles = idx_sim[:N]
    similar_articles = idx_sim
    articles = [art[0] for art in similar_articles]
    similaridad_articles = [art[1] for art in similar_articles]
    return articles,similaridad_articles

# %%

nodos_recorridos = []
similaridades_conectados = []
for i in list(G.nodes()):
    nodos_similares,similaridad = predict_links_conectados(G, emb_df, i)
    nodos_recorridos.append(i)
    #Hago esto para no agregar similaridad de nodos dos veces.
    for i,nodo in enumerate(nodos_similares):
        if nodo not in nodos_recorridos:
            similaridades_conectados.append(similaridad[i])

# %%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(similaridades_no_conectados, bins=30)
ax.hist(bins[:-1], bins,weights=counts/max(counts),range = [0,1], rwidth = 0.80, facecolor='r', alpha=0.75,label= "Artistas no enlazados")

counts, bins = np.histogram(similaridades_conectados, bins=30)
ax.hist(bins[:-1], bins,weights=counts/max(counts),range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75,label= "Artistas enlazados")

ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel("Similaridad", fontsize=12)
ax.set_ylabel("Frecuencia normalizada", fontsize=12)
plt.title("Similaridad entre artistas enlazados y no enlazados",fontsize = 18)
ax.legend(loc = 'best')
#plt.savefig("Similaridad entre artistas.png")
plt.show()
 
# %%
nodos_recorridos = []
artistas_y_similaridades = []
similaridades= []
for nodo in list(G.nodes()):
    nodos_similares,similaridad = predict_links(G, emb_df, nodo)
    nodos_recorridos.append(nodo)
    #Hago esto para no agregar similaridad de nodos dos veces.
    for j,nodo_similar in enumerate(nodos_similares):
        if nodo_similar not in nodos_recorridos:
            artistas_y_similaridades.append([nodo,nodo_similar,similaridad[j]])
            

# %%
def tomartercero(elem):
    return elem[2]

artistas_y_similaridades.sort(key=tomartercero,reverse=True)

# %%
print(artistas_y_similaridades[0:100])
# %%
