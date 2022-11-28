#%%
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
#%%-----------------------------Cargo la red---------------------------------
with open(f"../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

#%%--------------------------Hago un dataframe con los enlaces--------------------------
enlaces_nuevos = []
for edge in (G.edges(data=True)):
    cuadrivector = [edge[0],edge[1], edge[2]["nombre"], edge[2]["fecha"]]
    enlaces_nuevos.append(cuadrivector)
df = pd.DataFrame(enlaces_nuevos, columns= ["Source", "Target", "Label", "Fecha"])
df = df.set_index("Source")
#%%-------------Consigo las canciones en que colaboraron mas de 2 artistas-----------------
lista_canciones = df["Label"]
i = 0
lista_canciones_repetidas = []
while i < len(lista_canciones)-1:
    if (lista_canciones[i] == lista_canciones[i+1]) and (lista_canciones[i] not in lista_canciones_repetidas):
        lista_canciones_repetidas.append(lista_canciones[i])
    i += 1
#%%------------------Agrego los enlaces que faltan a la red-----------------
G_copia = copy.deepcopy(G)
# Recorro todas las canciones repetidas
# Creo una lista donde voy a ver que enlaces se fueron haciendo
enlaces_nuevos = []
# Hago una lista con los cliques de artistas y la cancion que los une, sirve para chequear
canciones_y_cliques = []
for cancion_repetida in lista_canciones_repetidas:
    # Armo listas para guardar los enlaces y los artistas que voy a unir
    artistas_clique = []
    # Recorro todas las canciones de la red
    for i,cancion in enumerate(lista_canciones):        
        # Chequeo si es la misma cancion
        if cancion == cancion_repetida:
            # Meto los artistas en una lista
            source = df.index[i]
            target = df["Target"][i]
            # Chequeo que no estén en la lista antes
            if source not in artistas_clique:
                artistas_clique.append(df.index[i])
            if target not in artistas_clique:
                artistas_clique.append(df["Target"][i])
            
    # Recorro el clique y genero enlaces entre todos los que no tengan
    for artista_i in artistas_clique:
        for artista_j in artistas_clique:
            # Chequeo no hacer el autoenlace
            if artista_i != artista_j:
                # Chequeo que no tengan el enlace
                enlace = (artista_i, artista_j, cancion_repetida ,df["Fecha"][i])
                enlace_alternado = (artista_j, artista_i, cancion_repetida ,df["Fecha"][i])
                agregar_cancion = False
                # Chequeo que no tengan la cancion
                try:
                    # Recorro las canciones entre los artistas
                    for colaboracion in G[artista_i][artista_j].items():
                        cancion_colaboracion = colaboracion[1]["nombre"]
                        # Chequeo que no sea la misma cancion
                        if cancion_colaboracion != cancion_repetida:
                            agregar_cancion = True
                # En este caso los artistas no habian colaborado antes, entonces agrego la cancion
                except:
                    agregar_cancion = True
                # Chequeo de varias maneras para no repetir la cancion, ambas son necesarias

                if agregar_cancion and enlace not in enlaces_nuevos:
                    # Agrego los enlaces a la lista
                    enlaces_nuevos.append(enlace)
                    enlaces_nuevos.append(enlace_alternado)
                    # Agrego el enlace a la red
                    G_copia.add_edge(artista_i, artista_j, nombre = cancion_repetida ,fecha = df["Fecha"][i])
    # Agrego los cliques con la canción en la que colaboraron
    canciones_y_cliques.append((artistas_clique, cancion_repetida))

#%%-----------------------Casos que use para chequear----------------------------------

# Chequeo la Paco Amoroso: Bzrp Music Sessions
print("Estos dos no tenian la colaboracion y tenian que tenerla")
print(G_copia["Paco Amoroso"]["Axel Fiks"],"\n")

print("Estos dos ya tenian la colaboracion y no tenia que volver a agregarla")
print(G_copia["Paco Amoroso"]["Bizarrap"])
#%%-------------------------------------Guardo al red------------------------------
nx.write_gpickle(G_copia, f"../red_filtrada/red_filtrada.gpickle")
#%%-----------------------Guardo la nueva tabla de enlaces--------------------------
enlaces_nuevos = []
for edge in (G_copia.edges(data=True)):
    cuadrivector = [edge[0],edge[1], edge[2]["nombre"], edge[2]["fecha"]]
    enlaces_nuevos.append(cuadrivector)
df_nuevo = pd.DataFrame(enlaces_nuevos, columns= ["Source", "Target", "Label", "Fecha"])
df_nuevo = df_nuevo.set_index("Source")

filename = f"../red_filtrada/tabla_enlaces.csv"
df_nuevo.to_csv(filename)
