#%%
import networkx as nx
import pandas as pd
import numpy as np
import pickle

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

print(emb_df.head())

#%%
mdl.wv.get_vector("Bizarrap")

#%%

unique_nodes = list(G.nodes())
all_possible_edges = [(x,y) for (x,y) in product(unique_nodes, unique_nodes)]

# generate edge features for all pairs of nodes
edge_features = [
    (mdl.wv.get_vector(str(i)) + mdl.wv.get_vector(str(j))) for i,j in all_possible_edges
]

# get current edges in the network
edges = list(G.edges())

# create target list, 1 if the pair exists in the network, 0 otherwise
is_con = [1 if e in edges else 0 for e in all_possible_edges]

print(sum(is_con))
# %%TRAIN MODEL 

X = np.array(edge_features)
y = is_con

# train test split
x_train, x_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size = 0.3
)
#%%
param_grid = { 
    'n_estimators': [200, 400, 800],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [int(x) for x in np.linspace(10, 110, 11)],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'criterion' :['gini', 'entropy'],
    }
rfc = RandomForestClassifier(random_state=1)
CV_rfc = GridSearchCV(estimator=rfc , param_grid=param_grid,verbose=2)
CV_rfc.fit(x_train, y_train)
print(f'RANDOM FOREST - score: {CV_rfc.score(x_test, y_test)}')
print(f'RANDOM FOREST - best_params: {CV_rfc.bestparams} ')
#%%
# 2) SUPPORT VECTOR MACHINE
param_grid = {'C': [0.001, 0.005, 0.1,0.2,0.5,1,5, 10, 25, 50, 100],
              'gamma': [1,0.5,0.2,0.1,0.05,0.02,1e-2, 1e-3, 1e-4, 1e-5],
              'kernel': ['rbf']
              }
svc = SVC(random_state=1)
# CV_svc = GridSearchCV(estimator = svc,param_grid=param_grid)
svc.fit(x_train, y_train)
print(f'SUPPORT VECTOR MACHINE - score: {svc.score(x_test, y_test)}')
print(f'SUPPORT VECTOR MACHINE - best_params: {svc.bestparams}')
# 3) MULTI-LAYER PERCEPTRON
#%%
param_grid = {
    'hidden_layer_sizes': [(100,),(50,50,50), (50,100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': np.linspace(0.0001, 0.05,50), 
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(random_state=1)
CV_mlp = GridSearchCV(estimator= mlp, param_grid=param_grid,refit = True)
CV_mlp.fit(X_train, y_train)
print(f'MULTI-LAYER PERCEPTRON - score: {CV_mlp.score(X_test, y_test)}')
print(f'MULTI-LAYER PERCEPTRON - best_params: {CV_mlp.bestparams}')
# GBC classifier
clf = GradientBoostingClassifier()

# train the model
clf.fit(x_train, y_train)

#%% EVALUATION  

y_pred = clf.predict(x_test)
y_true = y_test

y_pred = clf.predict(x_test)
x_pred = clf.predict(x_train)
test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, x_pred)
print("Testing Accuracy : ", test_acc)
print("Training Accuracy : ", train_acc)

print("MCC Score : ", matthews_corrcoef(y_true, y_pred))

print("Test Confusion Matrix : ")
print(confusion_matrix(y_pred,y_test))

print("Test Classification Report : ")
print(classification_report(y_test, clf.predict(x_test)))
#%%
print(x_pred)
#%%
for nodo in G.nodes():
    for nodo2 in G.nodes():
        pred_ft = [(mdl.wv.get_vector(str(nodo))+mdl.wv.get_vector(str(nodo2)))]
        print(f"Los nodos {nodo} y {nodo2} colaboraran:")
        print(clf.predict(pred_ft)[0])

        print(clf.predict_proba(pred_ft))
        print("")


# %%
def predict_links(G, df, article_id, N):
    '''
    This function will predict the top N links a node (article_id) should be connected with
    which it is not already connected with in G.
    
    params:
        G (Netowrkx Graph) : The network used to create the embeddings
        df (DataFrame) : The dataframe which has embeddings associated to each node
        article_id (Integer) : The article you're interested 
        N (Integer) : The number of recommended links you want to return
        
    returns:
        This function will return a list of nodes the input node should be connected with.
    '''
    
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
    
    similar_articles = idx_sim[:N]
    articles = [art[0] for art in similar_articles]
    return articles
  
predict_links(G = G, df = emb_df, article_id = "Duki", N = 20)
# %%
print(G.edges("Bizarrap"))

G.edges("Duki")
# %%
