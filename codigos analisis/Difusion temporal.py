#%% 
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
import copy
import random as random
#%%-----------------------Cargamos el multigrafo---------------------------
with open('../red_filtrada/red_filtrada.gpickle', "rb") as f:
    G = pickle.load(f)
    
    
#%%----------------------CREO MATRICES y copio la lista de todos los generos---------------------------
W = nx.to_pandas_adjacency(G).to_numpy()
lista_grados = [i[1] for i in G.degree()]
D = np.diag(lista_grados)
inversa_D =  np.linalg.inv(D) 

matriz = np.dot(inversa_D,W)
generos = ["Trap","Jazz", "Pop", "HipHop", "Clasico", "Indie",
                           "R&B", "Tango", "Cumbia",
                           "Chamame", "Electronica", "Folklore", "Rock", "Punk", "Rap", "Metal",
                           "Reggae","Alternative"]



#%%
vectores_finales = []
valores_intermedios = []

for genero in generos[0:1]:

    vector = []

    for i in G.nodes(data=True):
        
        if len(i[1]["generos_musicales"]) != 0: #SI TIENE GENEROS, LE ASIGNO 1 O -1 SEGUN SEA EL GENERO CORRESPONDIENTE
            if genero in i[1]["generos_musicales"]:
                vector.append(1)
            else:
                vector.append(0)
        else: #SI NO TIENE GENEROS, LE ASIGNO 0.
            vector.append(0)

    for i in range(1000):
        vector_nuevo = np.matmul(matriz,vector) #Esto multiplica la matriz por el vector
        for indice,elemento in enumerate(vector): ##Esto vuelve a setear los elementos que eran 1 a -1 a esos valores

            if elemento == 1:
                vector_nuevo[indice] = 1

            elif elemento == -1:
                vector_nuevo[indice] = -1
        
        vector = vector_nuevo
        if i% 10 == 0:
            print(vector[-1])
            valores_intermedios.append(vector[-1])
        

#%%

G_copia = copy.deepcopy(G)  #TENGO Q COPIAR ASI PQ SINO LOS ATRIBUTOS SE MODIFICAN EN AMBAS REDES.wtff
# %%
for indice_nodo,nodo in enumerate(G_copia.nodes(data=True)): #Recorro todos los nodos
    if len(nodo[1]["generos_musicales"]) == 0: #Agarro los nodos que no tienen genero musical
            
        for indice_genero,vector in enumerate(vectores_finales): #Recorro la lista de vectores finales (todos los generos)
            if vector[indice_nodo] > 0: #si es mayor a cero el valor, le meto el genero
                nodo[1]["generos_musicales"].append(generos[indice_genero])
            


#%% PRINTEO QUIENES NO TENIAN GENERO
for nodo in G.nodes(data=True):
    if len(nodo[1]["generos_musicales"]) == 0:
        print(nodo)
        
        
#%% FIJENSE QUE HAY 10 NODOS A LOS QUE NO LES DETECTO NINGUN GENERO. AGREGO ABAJO PARA ELLOS (busque en internet)
nodos_sin_genero = []
for nodo in G_copia.nodes(data=True):
    if len(nodo[1]["generos_musicales"]) == 0:
        nodos_sin_genero.append(nodo[0])
        
print(nodos_sin_genero)


#%%
#Creo lista en orden con los generos de cada uno.
generos_para_estos_nodos = [["Cumbia","Rap"], ["Cumbia"],["Reggae"],["Cumbia"],["Rock"],["Cumbia"]
                            ,["Alternative","Tango"],["Trap","Pop"],["Tango"],["Folklore"]]


for indice,nodo in enumerate(nodos_sin_genero):
    G_copia.nodes()[nodo]["generos_musicales"] = generos_para_estos_nodos[indice]


# %% HAGO ESTO PARA VER LOS NODOS A LOS QUE LES DETECTO GENERO Y NO LO TENIAN.
j = 0
for nodo in list(G.nodes(data=True)):
    #Me fijo que en G no tenga generos pero en G_copia si, y printeo sus generos nuevos.
    if (len(nodo[1]["generos_musicales"]) == 0) and (len(G_copia.nodes()[nodo[0]]["generos_musicales"]) != 0):
        print(f'{nodo[0]} tiene finalmente géneros musicales:  {G_copia.nodes()[nodo[0]]["generos_musicales"]}')
        j+=1
print(len(G.nodes())-j) #Con esto me fijo de que de la cantidad q tenia inicialmente genero. Da 741. ta bien. :D


# %% -----------------------------------------------------------------------------
# ACa calculo que tan bien lo hace. Voy a sacarle a aprox el 10% de los nodos aleatoriamente los generos musicales
#Y quiero ver si me reconoce bien al menos uno de los generos q tenia (tomo como exitoso con tan solo 1).
#copio todo lo anterior.
#Itero 100 veces y calculo la efectividad
from tqdm import tqdm
def TPR_FPR(cm): # computa la tasa de verdaderos y falsos positivos a partir de la matriz de confusion
  TPR = cm[1,1]/(cm[1,1]+cm[0,1])
  FPR = cm[1,0]/(cm[1,0]+cm[0,0])
  return [TPR,FPR] # devuelve una lista donde el primer elemento es la tasa de verdaderos positivos y el segundo la tasa de falsos negativos
generos = ["Trap","Jazz", "Pop", "HipHop", "Clasico", "Indie",
                            "R&B", "Tango", "Cumbia",
                            "Chamame", "Electronica", "Folklore", "Rock", "Punk", "Rap", "Metal",
                            "Reggae","Alternative"]

porcentajes = []
TPRs_final = []
FPRs_final = []

for i in tqdm(range(1)):
    G_testeo = copy.deepcopy(G)

    nodos_de_testeo = []
    #Quito los generos musicales de aprox el 10% de los nodos.
    for nodo in G_testeo.nodes(data=True):
        if len(nodo[1]["generos_musicales"]) != 0:
            probabilidad = random.random() 
            if probabilidad < 0.1:
                nodo[1]["generos_musicales"] = []
                nodos_de_testeo.append(nodo[0])
                
    #HAGO LO MISMO QUE ANTES PARA LA RED DE TESTEO.

    W = nx.to_pandas_adjacency(G_testeo).to_numpy()
    lista_grados = [i[1] for i in G_testeo.degree()]
    D = np.diag(lista_grados)
    inversa_D =  np.linalg.inv(D) 

    matriz = np.dot(inversa_D,W)
    

    vectores_finales = []
    for genero in generos:
        vector = []

        for i in G_testeo.nodes(data=True):
            
            if len(i[1]["generos_musicales"]) != 0: #SI TIENE GENEROS, LE ASIGNO 1 O -1 SEGUN SEA EL GENERO CORRESPONDIENTE
                if genero in i[1]["generos_musicales"]:
                    vector.append(1)
                else:
                    vector.append(0)
            else: #SI NO TIENE GENEROS, LE ASIGNO 0.
                vector.append(0)

        for i in range(1000):
            vector_nuevo = np.matmul(matriz,vector) #Esto multiplica la matriz por el vector
            for indice,elemento in enumerate(vector): ##Esto vuelve a setear los elementos que eran 1 a -1 a esos valores

                if elemento == 1:
                    vector_nuevo[indice] = 1

                elif elemento == -1:
                    vector_nuevo[indice] = -1
            
            vector = vector_nuevo
            
        vectores_finales.append(vector_nuevo)
        probabilidad = np.linspace(0,1,50)
        TPRs = []
        FPRs = []
        
        for proba in probabilidad:
            G_testeo2 = copy.deepcopy(G_testeo)
            for indice_nodo,nodo in enumerate(G_testeo2.nodes(data=True)): #Recorro todos los nodos
                if len(nodo[1]["generos_musicales"]) == 0: #Agarro los nodos que no tienen genero musical
                        
                    for indice_genero,vector in enumerate(vectores_finales): #Recorro la lista de vectores finales (todos los generos)
                        if vector[indice_nodo] > proba: #si es mayor a proba el valor, le meto el genero
                            nodo[1]["generos_musicales"].append(generos[indice_genero])

            #Recien esto es nuevo
            TP,TN,FP,FN = 0,0,0,0
        
            for nodo in nodos_de_testeo:
                generos_verdaderos = G.nodes()[nodo]["generos_musicales"]
                generos_nuevos = G_testeo2.nodes()[nodo]["generos_musicales"]
                
                #TRUE_POSITIVES = LOS QUE TENIA Y LOS QUE ACERTÓ
                TP += len(set(generos_verdaderos).intersection(set(generos_nuevos)))
                #True_negative = los que no están y no tenian que estar.
                TN += len(set(generos) - (set(generos_nuevos + generos_verdaderos)))
                #False_positives = los que están pero no deberian
                FP += len((set(generos_nuevos) ^ set(generos_verdaderos)) - (set(generos_verdaderos)))
                #False_negatives = los que no estan y deberian
                FN += len((set(generos_nuevos) ^ set(generos_verdaderos)) - (set(generos_nuevos)))
        
            
            matriz_confusion = np.array([[TP,FP],[FN,TN]])
            metricas = TPR_FPR(matriz_confusion)
            
            TPRs.append(metricas[0]) # guardamos los resultados en las listas respectivas
            FPRs.append(metricas[1])
            
        TPRs_final.append(TPRs)
        FPRs_final.append(FPRs)

#%%

FPRs_final_grafico = np.array([sum(l) for l in zip(*FPRs_final)])/(len(FPRs_final))
TPRs_final_grafico = np.array([sum(l) for l in zip(*TPRs_final)])/(len(TPRs_final))


fig, ax = plt.subplots(figsize = (10,7))
ax.set_title('Verdaderos positivos vs. falsos positivos')
ax.scatter(FPRs_final_grafico,TPRs_final_grafico)
ax.scatter()
ax.set_xlabel('Tasa de falsos positivos') # Etiqueta del eje x
ax.set_ylabel('Tasa de verdaderos positivos') # Etiqueta del eje y

# %%TODO ESTO FUE PARA DEFINIR LA PROBABILIDAD QUE PARECE MEJORAR LA TASA DE VERDADEROS POSITIVOS Y 
#MINIMIZANDO LA DE TASA DE FALSOS POSITIVOS.
