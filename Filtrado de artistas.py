#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unicodedata import normalize
import pickle 
import textdistance as td
from scipy.spatial.distance import pdist, squareform
#%%

def normalizar(string):
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    string_normalizado = normalize('NFKC', normalize('NFKD', string).translate(trans_tab)).lower().replace(" ", "")
    return string_normalizado

# %%
lista_artistas = pd.read_csv("artistas.csv")

# %%
artistas_repetidos = []
artistas_normalizados = []
for artista_i in lista_artistas["nombre"]:
    nombre_normalizado_i = normalizar(artista_i)
    for artista_j in lista_artistas["nombre"]:
        nombre_normalizado_j = normalizar(artista_j)
        if nombre_normalizado_j in nombre_normalizado_i and artista_i != artista_j and (nombre_normalizado_i, nombre_normalizado_j) not in artistas_normalizados:
            artistas_repetidos.append((artista_i, artista_j)) 
            artistas_normalizados.append((nombre_normalizado_i, nombre_normalizado_j))

#%%
artistas_repetidos_filtrados = []

palabras_filtro = ["UN","Fernando","Rodrigo","Axel","Emilia","TINI","Rei","Wen","Árbol",'Cacho Lafalce, Bernardo Baraj, Cacho Arce, Domingo Cura & Chino Rossi',
                   'David Lebón Jr',"Vandera",'Jairo',"Julio Martinez","Karina Cohen",'Lalo Schifrin',
                   'Lagartijeando',"MYA","Juanse","MAX","ACRU","Oscar Alem",'Sandro',"Carca"
                   ,"La Mississippi","Roberto Diaz Velado"]

for i in artistas_repetidos:
    filtro = True
    j = 0
    while filtro and j < len(palabras_filtro):
        
        if palabras_filtro[j] in i:
            filtro = False
        j+= 1

    if filtro:
        artistas_repetidos_filtrados.append(i)
#%%
for i in artistas_repetidos_filtrados:
    print(i)
#%%
i = 1496
with open(f"red_final/Iteracion {i}/red_final_hasta_indice_{i}.gpickle", "rb") as f:
    G = pickle.load(f)

G_copia = G.copy()

nodos_para_remover = ['David Lebón Jr',"Julio Martinez Oyanguren",'Lalo Schifrin Conducts Stravinsky, Schifrin, And Ravel',
                      "Lagartijeando, Sajra", 'Mya feat. Spice','Mya feat. Stacie & Lacie','Mya feat. Trina'
                      ,'KR3TURE', 'Feral Fauna', "MAX"]

G_copia.remove_nodes_from(nodos_para_remover)

# %%
# enlaces = list(G.edges(data= True))

#artistas_repetidos_filtrados = sorted(artistas_repetidos_filtrados)


artistas_a_matar = []

for i,j in artistas_repetidos_filtrados:
    if len(i) <= len(j):
        enlaces = list(G.edges(j,data=True))
        for data_enlace in enlaces:
           # print(i,j)
            if i not in list(G_copia.nodes()):
                print(f"agregue a {i}")
              #  print(i,j)
              #  print("ajdhajdhja")
            G_copia.add_edge(i,data_enlace[1],nombre = data_enlace[2]["nombre"],fecha = data_enlace[2]["fecha"])

    else:
        enlaces = list(G.edges(i,data=True))
        for data_enlace in enlaces:
            if j not in list(G_copia.nodes()):
                print(f"agregue a {j}")
               # print(i,j)
                #print("ajdhajdhja")
            G_copia.add_edge(j,data_enlace[1],nombre = data_enlace[2]["nombre"],fecha = data_enlace[2]["fecha"])
#print(artistas_a_matar)

G_copia.remove_nodes_from(artistas_a_matar)


datos = set(G.nodes()) ^ set(G_copia.nodes()) 
#%%

for i in datos:
    if i not in artistas_repetidos_filtrados:
        print(i)
#%%
lista = [i for i,j in artistas_repetidos_filtrados] + [j for i,j in artistas_repetidos_filtrados]
lista = list(np.unique(lista))

no_interseccion = datos ^ set(lista)

print(len(no_interseccion))
# %%
enlaces = [G.edges("Bizarrap",data = True)]
print(enlaces)
# %%
