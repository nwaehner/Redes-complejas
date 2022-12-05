#%%-----------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use("seaborn")
#%%-----------------------Cargamos el multigrafo---------------------------
with open("../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

cant_nodos = len(G.nodes())
#%% Recorremos los nodos

dict_generos = {}

for nodo in G.nodes():
    generos_musicales = G.nodes()[nodo]["generos_musicales"]
    
    for genero in generos_musicales:
        if genero in dict_generos.keys():
            dict_generos[genero] += 1
        else:
            dict_generos[genero] = 1

dict_generos_ordenados = {genero: valor for genero, valor in sorted(dict_generos.items(), key=lambda item: item[1])}

df = pd.DataFrame(dict_generos_ordenados, index = ["Cantidad de artistas"])

#%% Graficamos la distribucion
fig, axs = plt.subplots(figsize = (8,4))

sns.barplot(data = df, ax = axs, orient = "h")
axs.tick_params(axis = "both", labelsize = 12)
axs.set_xlabel("Cantidad de artistas",fontsize = 14)
#plt.savefig("../imagenes del analisis/Cantidad de artistas por genero.png")

#%%
dict_generos_ordenados = {genero: valor/cant_nodos for genero, valor in sorted(dict_generos.items(), key=lambda item: item[1])}

df_porcentaje = pd.DataFrame(dict_generos_ordenados, index = ["Porcentaje de artistas"])

#%% Graficamos la distribucion
fig, axs = plt.subplots(figsize = (8,4))
paleta_colores = sns.color_palette("Spectral",17)
sns.barplot(data = df_porcentaje, ax = axs, orient = "h", palette=paleta_colores)
axs.tick_params(axis = "both", labelsize = 12)
axs.set_xlabel("Porcentaje de artistas",fontsize = 14)
#plt.savefig("../imagenes del analisis/Porcentaje de artistas por genero.png")

plt.show()
# %%
