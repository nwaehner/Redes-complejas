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
        # if comparar_strings(artista_i,artista_j,1)[0] and comparar_strings(artista_j,artista_i)[0] and artista_i != artista_j and (normalizar(artista_i),normalizar(artista_j)) not in artistas_normalizados:
        #     artistas_repetidos.append((artista_i, artista_j)) 
        #     artistas_normalizados.append((normalizar(artista_i), normalizar(artista_j)))

#%%
artistas_repetidos_filtrados = []
palabras_filtro = ["UN","Fernando","Rodrigo","Axel","Emilia","TINI","Rei","Wen","Árbol",'Cacho Lafalce, Bernardo Baraj, Cacho Arce, Domingo Cura & Chino Rossi',
                   'David Lebón Jr',"Vandera",'Jairo',"Julio Martinez","Karina Cohen",'Lalo Schifrin',
                   'Lagartijeando',"MYA","Juanse","MAX","ACRU","Oscar Alem",'Sandro',"Carca",
                   ]


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
#print(artistas_coincidentes)
print(artistas_repetidos_filtrados)
for i in artistas_repetidos_filtrados:
   if 'La Mississippi' in i:
    print(i)
        
# print(artistas_normalizados)
# %%
#intento acá generar una matriz de similaridad de artistas por bloques
with open(f"red_final/Iteracion 1496/lista_artistas_argentinos_hasta_indice_1496.pickle", "rb") as f:
    artistas = pickle.load(f)

artistas_normalizados = sorted([normalizar(artista) for artista in artistas])
transformed_strings = np.array(artistas_normalizados).reshape(-1,1)
distance_matrix = pdist(transformed_strings,lambda x,y: td.hamming.normalized_similarity(x[0],y[0]))
#%%
plt.imshow(squareform(distance_matrix))