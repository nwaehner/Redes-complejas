#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
#%% Cargamos la red
with open('red_filtrada/red_filtrada.gpickle', "rb") as f:
    G = pickle.load(f)
#%% Obtenemos una lista con los géneros
lista_generos = []
for nodo in G.nodes():
    lista_generos_nodo = G.nodes()[nodo]["generos_musicales"]

    lista_generos.extend(lista_generos_nodo)

lista_generos = np.unique(lista_generos)
#%%
generos_representativos = ["Jazz","Trap","Rap", "Hip Hop", "Classic",
                           "Pop", "Cumbia", "Tango", "Folklore", "Indie",
                           "Chamame", "Cuarteto", "Reggae", "Metal", 
                            "Infantil", "Blues", "Punk"]

generos_representativos = ["Trap","Jazz", "Pop", "Hip Hop", "Clas", 
                           "r&b", "Tango", "Cumbia", "Indie", 
                           "Chamame", "Tronica", "Fol", "Rock", "Punk", "Rap", "Metal",
                           "reggae"]

# CUIDADO RAP Y TRAP
#
string_generos = " ".join(generos_representativos)


#%% 
dic_generos = {}
lista_generos_filtrados = []
for nodo in G.nodes():
    lista_generos_nodo = G.nodes()[nodo]["generos_musicales"]

    for genero in lista_generos_nodo:

        if genero in dic_generos.keys():
            dic_generos[genero] += 1
        else:
            dic_generos[genero] = 1
    
#%%
dic_generos_ordenado = dict(sorted(dic_generos.items(), key = lambda x:x[1], reverse=True))
# %%
