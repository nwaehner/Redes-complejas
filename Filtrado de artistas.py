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

def comparar_strings(string_original, string_a_comparar, porcentaje_aceptado = 1):

    # El porcentaje se determino para que emilia mernes esté en la red
    """
    Recibe:
        string_original: string que se quiere comparar con otro
        string_a_comparar: string con el que se quiere comparar
        porcentaje_aceptado: porcentaje de inclusión del string_original en el string_a_comparar para definir que son igual
    Devuelve:
        comparacion: booleano dependiendo si los strings son porcentualmente iguales o no
        porcentaje: porcentaje igualdad de los strings a comparar
    """

    porcentaje = 0
    comparacion = False
    # Sacamos tildes y dieresis

    string_original = normalizar(string_original)
    string_a_comparar = normalizar(string_a_comparar)

    for i, caracter in enumerate(string_a_comparar):
        if (i < len(string_original)) and (caracter == string_original[i]):
            porcentaje += 1
    
    porcentaje = porcentaje/len(string_a_comparar)
    if porcentaje >= porcentaje_aceptado:
        comparacion = True
        
    return comparacion, porcentaje


# %%
lista_artistas = pd.read_csv("artistas.csv")
# %%
artistas_coincidentes = []
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

artistas_repetidos_filtrados = sorted(artistas_repetidos_filtrados)
artistas_a_matar = []
for i,j in artistas_repetidos_filtrados:
    if len(i) <= len(j):
        enlaces = list(G.edges(j,data=True))
        for dataenlace in enlaces:
            G_copia.add_edge(i,dataenlace[1],nombre = dataenlace[2]["nombre"],fecha = dataenlace[2]["fecha"])
        artistas_a_matar.append(j)
    else:
        enlaces = list(G.edges(i,data=True))
        for dataenlace in enlaces:
            G_copia.add_edge(j,dataenlace[1],nombre = dataenlace[2]["nombre"],fecha = dataenlace[2]["fecha"])
            
        artistas_a_matar.append(i)

G_copia.remove_nodes_from(artistas_a_matar)


datos = set(G.nodes()) ^ set(G_copia.nodes())

#%%
lista = [i for i,j in artistas_repetidos_filtrados] + [j for i,j in artistas_repetidos_filtrados]
print(np.unique(lista))

print(set(datos))
# %%
enlaces = [G.edges("Bizarrap",data = True)]
print(enlaces)
# %%
