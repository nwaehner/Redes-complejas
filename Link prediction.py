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

print(emb_df.head())



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
        if clf.predict(pred_ft)[0] != 0:
            print(f"Los nodos {nodo} y {nodo2} colaboraran:")
            print(clf.predict(pred_ft)[0])

            print(clf.predict_proba(pred_ft))
            print("")


# %%
