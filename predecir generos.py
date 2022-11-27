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
# %% Predictor de géneros musicales, me agarro lo otro 
#del otro codigo mientras para definirle un genero a cada uno

generos_representativos = ["Trap","Jazz", "Pop", "Hip Hop", "Clas", 
                           "r&b", "Tango", "Cumbia", "Indie", 
                           "Chamame", "Tronica", "Fol", "Rock", "Punk", "Rap", "Metal",
                           "reggae"]

G_copia = G.copy()
dict_generos = dict(zip(generos_representativos,[i for i in range(len(generos_representativos))]))
lista_nodos = list(G.nodes())
Y = []
nodos_sin_etiquetas = []  #artistas que no tienen genero musical
for nodo in lista_nodos:
    generos_nuevos =[]
    generos_spotify = (G_copia.nodes())[nodo]["generos_musicales"]
    # Recorremos los generos que creemos que son importantes
    for genero in generos_representativos:
        # Recorremos los generos de spotify del cantante
        for genero_spotify in generos_spotify:
            # Comprobamos que el genero entre en alguno de los generos de spotify
            if genero.lower().replace(" ", "") in genero_spotify.lower().replace(" ", ""):
                generos_nuevos.append(genero)
    generos_nuevos = list(np.unique(generos_nuevos))
    if len(generos_nuevos)>0:
        vector_generos = [0 for i in range(len(generos_representativos))]
        for genero_nuevo in generos_nuevos:
            vector_generos[dict_generos[genero_nuevo]] = 1
        G_copia.nodes[nodo]["label"] = vector_generos
        Y.append(vector_generos)
    else:
        nodos_sin_etiquetas.append(nodo)
        G_copia.nodes[nodo]["label"] = np.nan
        Y.append(np.nan)

emb_df['target'] = Y
# clasificador: ensemble de árboles (random forest)
from sklearn.ensemble import RandomForestClassifier

emb_df.dropna(inplace=True)
# %%
#librerias utilizadas

# selección de modelos
from sklearn.model_selection import train_test_split
# clasificador: red neuronal (perceptron multicapa)
from sklearn.neural_network import MLPClassifier
# clasificador: support vector classifier
from sklearn.svm import SVC
# clasificador: ensemble de árboles (random forest)
from sklearn.ensemble import RandomForestClassifier

#modelos de regresión
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#GridSearchCV  paraajuste de hiperparámetros con cross-validation
from sklearn.model_selection import GridSearchCV  

#One-hot encoding para las variables categóricas
from sklearn.preprocessing import OneHotEncoder

# performance (matriz de confusión)
from sklearn.metrics import confusion_matrix
#%%
rfc = RandomForestClassifier(random_state=1)
# %%
y = emb_df['target'].to_numpy() #target para clasificación por niveles
X = emb_df.loc[:, emb_df.columns != 'target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# %%
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_test_knn = knn.predict(X_test)
# %%
