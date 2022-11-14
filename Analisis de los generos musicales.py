#%%-----------------------------------------------------------------
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("seaborn")
#%%-----------------------Cargamos el multigrafo---------------------------
i = 1496

with open(f"red_final/Iteracion {i}/red_final_hasta_indice_{i}.gpickle", "rb") as f:
    G = pickle.load(f)

#%%Defino una función para encontrar la fecha donde salió la primera colaboración de un artista
def encontrar_fecha_menor(lista_fechas):

    año_menor = min([fecha[0:4] for fecha in lista_fechas])

    #mes_menor = min([fecha[5:7] for fecha in lista_fechas if fecha[0:4] == año_menor])
    fecha_menor = int(año_menor)

    return fecha_menor

#%%------------------------Agrego el atributo a los nodos----------------------
lista_nodos = list(G.nodes())

for nodo in lista_nodos:
    lista_fechas = []
    for item in G[nodo].items():
        fecha = item[1][0]["fecha"]
        lista_fechas.append(fecha)

    fecha_aparicion = encontrar_fecha_menor(lista_fechas)
    G.nodes[nodo]["fecha_aparicion"] = fecha_aparicion

#%%---------------Definimos los generos que vamos a estudiar--------------

generos_representativos = ["Rock","Jazz","Trap","Rap", "Hip Hop", "Classic",
                           "Pop", "Cumbia", "Tango", "Folklore", "Indie"]

# generos que aparecen muy poco y saque

generos_no_representativos = ["Chamame", "Cuarteto", "Reggae", "Metal", 
                    "Infantil", "Blues", "Punk"]
#%%
for nodo in lista_nodos:
    generos_nuevos =[]
    generos_spotify = (G.nodes())[nodo]["generos_musicales"]
    # Recorremos los generos que creemos que son importantes
    for genero in generos_representativos:
        # Recorremos los generos de spotify del cantante
        for genero_spotify in generos_spotify:
            # Comprobamos que el genero entre en alguno de los generos de spotify
            if genero.lower().replace(" ", "") in genero_spotify.lower().replace(" ", ""):
                generos_nuevos.append(genero)
    
    generos_nuevos = list(np.unique(generos_nuevos))
    G.nodes[nodo]["generos_musicales"] = generos_nuevos

#%% Vemos cuantos artistas aparecen cada año
fig, axs = plt.subplots(ncols = 2, figsize = (14,8))
dic_años = {}

for nodo in lista_nodos:
    año = (G.nodes())[nodo]["fecha_aparicion"]
    if año in dic_años.keys():
        dic_años[año] += 1
    else:
        dic_años[año] = 0

lista_años_ordenados = sorted(dic_años)
dic_años_ordenado = {año_ordenado:dic_años[año_ordenado]  for año_ordenado in lista_años_ordenados}

artistas_acumulados_año = []

for i, año in enumerate(dic_años_ordenado):
    artistas_acumulados_año.append(dic_años_ordenado[año])
    if i > 0:
        artistas_acumulados_año[i] += artistas_acumulados_año[i-1]


axs[0].plot(dic_años_ordenado.keys(),dic_años_ordenado.values(),".", c = "m")
axs[0].set_title("$Artistas\;por\;año$",fontsize = 22)
axs[0].set_xlabel("Año",fontsize = 18)
axs[0].set_ylabel("Cantidad de artistas", fontsize = 18)
axs[0].tick_params(axis = "both", labelsize = 16)


axs[1].plot(dic_años_ordenado.keys(), artistas_acumulados_año, ".")
axs[1].set_title("$Artistas\;por\;año\;acumulado$",fontsize = 22)
axs[1].set_xlabel("Año",fontsize = 18)
axs[1].tick_params(axis = "both", labelsize = 16)

plt.savefig("artistas por año.png")
plt.show()
# %%

for i, genero in enumerate(generos_representativos):
    dic_años = {}
    for nodo in lista_nodos:
        if genero in (G.nodes())[nodo]["generos_musicales"]:
            año = (G.nodes())[nodo]["fecha_aparicion"]
            if año in dic_años.keys():
                dic_años[año] += 1
            else:
                dic_años[año] = 0
    
    dic_años = dict(filter(lambda x: x[1] != 0, dic_años.items()))
    lista_años_ordenados = sorted(dic_años)
    if i%4 == 0:
        plt.figure(figsize = (12,8)) 

    dic_años_ordenado = {año_ordenado:dic_años[año_ordenado]  for año_ordenado in lista_años_ordenados}

    plt.plot(dic_años_ordenado.keys(),dic_años_ordenado.values(), label = genero)
    plt.legend(fontsize = 16)
    plt.xlabel("Año",fontsize = 18)
    plt.ylabel("Artistas nuevos por genero", fontsize = 18)
    plt.tick_params(axis = "both", labelsize = 16)
#plt.savefig("artistas por genero por año.png")
plt.show()