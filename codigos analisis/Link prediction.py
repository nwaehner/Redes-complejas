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
with open(f"../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)
#%% ----------SIMILARIDAD COSENO PERO CON NODE2VEC----------------
g_emb = n2v(G, dimensions=8)
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
def predict_links_no_conect(G, df, article_id):
    
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

def predict_links_conect(G, df, article_id):
    
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

#%% SIMILARIDAD COSENO PERO A MANO.


def cosine_similarity2(nodo1,nodo2):
    vecinos_nodo1 = [i[1] for i in G.edges(nodo1)]
    grado_nodo1 = G.degree(nodo1)
    vecinos_nodo2 = [i[1] for i in G.edges(nodo2)]
    grado_nodo2 = G.degree(nodo2)
    interseccion = set(vecinos_nodo1).intersection(set(vecinos_nodo2))
    denominador = np.sqrt(grado_nodo1*grado_nodo2)
    return len(interseccion)/denominador
        
def predict_links_no_conect(G, artista):
    
    # other articles are all articles which the current doesn't have an edge connecting
    all_nodes = G.nodes()
    other_nodes = [n for n in all_nodes if ((n not in list(G.adj[artista])) and (n != artista))]
    
    # get similarity of current reader and all other readers
    nodos_y_sim = []
    for nodo in other_nodes:
        nodos_y_sim.append([nodo,cosine_similarity2(artista,nodo)])
    
    nodos_y_sim.sort(key=lambda x: x[1],reverse=True)
    nodos = [i[0] for i in nodos_y_sim]
    sim = [i[1] for i in nodos_y_sim]
    return nodos,sim

def predict_links_conect(G, artista):
    
    # other articles are all articles which the current doesn't have an edge connecting
    all_nodes = G.nodes()
    other_nodes = [n for n in all_nodes if ((n in list(G.adj[artista])) and (n != artista))]
    
    # get similarity of current reader and all other readers
    nodos_y_sim = []
    for nodo in other_nodes:
        nodos_y_sim.append([nodo,cosine_similarity2(artista,nodo)])
    
    nodos_y_sim.sort(key=lambda x: x[1],reverse=True)
    nodos = [i[0] for i in nodos_y_sim]
    sim = [i[1] for i in nodos_y_sim]
    return nodos,sim


#%%
nodos_recorridos = []
artistas_y_similaridades_conectados = []
artistas_y_similaridades_no_conectados = []
for nodo in list(G.nodes()):
    nodos_sim_no_conect , sim_no_conect = predict_links_no_conect(G, nodo)
    
    nodos_sim_conect , sim_conect = predict_links_conect(G,  nodo)
    
    nodos_recorridos.append(nodo)
    #Hago esto para no agregar similaridad de nodos dos veces.
    
    for j,nodo_sim_conect in enumerate(nodos_sim_conect):
        if nodo_sim_conect not in nodos_recorridos:
            artistas_y_similaridades_conectados.append([nodo,nodo_sim_conect,sim_conect[j]])
            
    for j,nodo_sim_no_conect in enumerate(nodos_sim_no_conect):
        if nodo_sim_no_conect not in nodos_recorridos:
            artistas_y_similaridades_no_conectados.append([nodo,nodo_sim_no_conect,sim_no_conect[j]])
            
# %% ARMO LISTA CON LAS SIMILARIDADES
similaridades_conectados = [i[2] for i in artistas_y_similaridades_conectados]
similaridades_no_conectados = [i[2] for i in artistas_y_similaridades_no_conectados]

#%%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(similaridades_no_conectados, bins=30)
ax.hist(bins[:-1], bins,weights=counts/max(counts),range = [0,1], rwidth = 0.80, facecolor='r', alpha=0.75,label= "Artistas no enlazados")

counts, bins = np.histogram(similaridades_conectados, bins=30)
ax.hist(bins[:-1], bins,weights=counts/max(counts),range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75,label= "Artistas enlazados")

ax.vlines(x = np.mean(similaridades_conectados), ymin = 0, ymax = 1, linewidth = 3, linestyle = '-', alpha = 0.8, color = 'b', label = 'Media conectados')
ax.vlines(x = np.mean(similaridades_no_conectados), ymin = 0, ymax = 1, linewidth = 3, linestyle = '-', alpha = 0.8, color = 'b', label = 'Media no conectados')

ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel("Similaridad", fontsize=12)
ax.set_ylabel("Frecuencia normalizada", fontsize=12)
plt.title("Similaridad entre artistas enlazados y no enlazados",fontsize = 18)
ax.legend(loc = 'best')
plt.savefig("Similaridad entre artistas.png")
plt.show()
 
#%%
def tomartercero(elem):
    return elem[2]
artistas_y_similaridades_no_conectados.sort(key=tomartercero,reverse=True)
print(artistas_y_similaridades_no_conectados[0:10])
for i in artistas_y_similaridades_no_conectados:
    if "Anibal Troilo" in i:
        print(i)
    
    
# %%
