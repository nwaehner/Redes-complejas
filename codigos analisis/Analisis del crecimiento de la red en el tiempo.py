#%%-----------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 
from unicodedata import normalize

plt.style.use("seaborn")
#%%----------Definimos una funcion para sacar tildes y mayusculas a strings----------

def normalizar(string):
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    string_normalizado = normalize('NFKC', normalize('NFKD', string).translate(trans_tab)).lower().replace(" ", "")
    return string_normalizado

#%%-----------------------Cargamos el multigrafo---------------------------
with open("../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

#%%
# Defino una función para encontrar la fecha donde salió la primera colaboración de un artista
def encontrar_fecha_menor(lista_fechas):

    año_menor = min([fecha[0:4] for fecha in lista_fechas])
    # Por si se quiere ver el mes
    #mes_menor = min([fecha[5:7] for fecha in lista_fechas if fecha[0:4] == año_menor])
    fecha_menor = int(año_menor)

    return fecha_menor

#%%------------------------Agrego el atributo de fecha a los nodos----------------------
lista_nodos = list(G.nodes())

for nodo in lista_nodos:
    lista_fechas = []
    for item in G[nodo].items():
        fecha = item[1][0]["fecha"]
        lista_fechas.append(fecha)

    fecha_aparicion = encontrar_fecha_menor(lista_fechas)
    G.nodes[nodo]["fecha_aparicion"] = fecha_aparicion
#%% Vemos cuantos artistas aparecen cada año
# Guardo en un diccionario cuantos nodos aparecen por año
dic_años = {}

for nodo in lista_nodos:
    año = (G.nodes())[nodo]["fecha_aparicion"]
    if año in dic_años.keys():
        dic_años[año] += 1
    else:
        dic_años[año] = 0

# Ordeno el diccionario por año
lista_años_ordenados = sorted(dic_años)
dic_años_ordenado = {año_ordenado:dic_años[año_ordenado]  for año_ordenado in lista_años_ordenados}

# Guardo en una lista la cantidad de artistas totales de la red en cada año
artistas_acumulados_año = []

for i, año in enumerate(dic_años_ordenado):
    artistas_acumulados_año.append(dic_años_ordenado[año])
    if i > 0:
        artistas_acumulados_año[i] += artistas_acumulados_año[i-1]

#%% Graficamos
fig, axs = plt.subplots(ncols = 2, figsize = (12,6))

axs[0].plot(dic_años_ordenado.keys(),dic_años_ordenado.values(),".", c = "m")
axs[0].set_title("$Artistas\;por\;año$",fontsize = 22)
axs[0].set_xlabel("Año",fontsize = 18)
axs[0].set_ylabel("Cantidad de artistas", fontsize = 18)
axs[0].tick_params(axis = "both", labelsize = 16)
#axs[0].xaxis.label.set_color('white')
#axs[0].yaxis.label.set_color('white')

axs[1].plot(dic_años_ordenado.keys(), artistas_acumulados_año, ".")
axs[1].set_title("$Artistas\;por\;año\;acumulado$",fontsize = 22)
axs[1].set_xlabel("Año",fontsize = 18)
axs[1].tick_params(axis = "both", labelsize = 16)
#axs[1].xaxis.label.set_color('white')
#axs[1].yaxis.label.set_color('white')

plt.savefig("../imagenes del analisis/artistas por año.png")
plt.show()
#%%
lista_cant_artistas = dic_años_ordenado.values()
max_artistas = max(lista_cant_artistas)
año_max_artistas = lista_años_ordenados[list(lista_cant_artistas).index(max_artistas)]

print(f"El año en que se unió la mayor cantidad de artistas fue {año_max_artistas} con {max_artistas}")

#%%-------------Analisis de cuantas canciones se hicieron en cada año----------------

# Guardo en un diccionario cuantas colaboraciones hubo en cada año
dic_canciones_por_año = {}
lista_canciones = list(G.edges(data = True))

for cancion in lista_canciones:
    año = int(cancion[2]["fecha"][:4])
 
    if año in dic_canciones_por_año.keys():
        dic_canciones_por_año[año] += 1
    else:
        dic_canciones_por_año[año] = 0

# Ordeno el diccionario por años
lista_años_ordenados = sorted(dic_canciones_por_año)
dic_canciones_por_año_ordenado = {año_ordenado:dic_canciones_por_año[año_ordenado]  for año_ordenado in lista_años_ordenados}

# Guardo en una lista la cantidad de artistas totales de la red en cada año
canciones_acumuladas_por_año = []

for i, año in enumerate(dic_canciones_por_año_ordenado):
    canciones_acumuladas_por_año.append(dic_canciones_por_año_ordenado[año])
    if i > 0:
        canciones_acumuladas_por_año[i] += canciones_acumuladas_por_año[i-1]

#%%-----------------Vemos cantas colaboraciones se hacen en cada año-----------------
fig, axs = plt.subplots(ncols = 2, figsize = (12,6))

axs[0].plot(dic_canciones_por_año_ordenado.keys(),dic_canciones_por_año_ordenado.values(),".", c = "m")
axs[0].set_title("$Colaboraciones\;por\;año$",fontsize = 22)
axs[0].set_xlabel("Año",fontsize = 18)
axs[0].set_ylabel("Cantidad de colaboraciones", fontsize = 18)
axs[0].tick_params(axis = "both", labelsize = 16)
#axs[0].xaxis.label.set_color('white')
#axs[0].yaxis.label.set_color('white')

axs[1].plot(dic_canciones_por_año_ordenado.keys(), canciones_acumuladas_por_año, ".")
axs[1].set_title("$Colaboraciones\;por\;año\;acumuladas$",fontsize = 22)
axs[1].set_xlabel("Año",fontsize = 18)
axs[1].tick_params(axis = "both", labelsize = 16)
#axs[1].xaxis.label.set_color('white')
#axs[1].yaxis.label.set_color('white')

plt.savefig("../imagenes del analisis/Colaboraciones por año.png")
plt.show()
# %%
lista_cant_colaboraciones = dic_canciones_por_año_ordenado.values()
max_colaboraciones = max(lista_cant_colaboraciones)
año_max_colaboraciones = lista_años_ordenados[list(lista_cant_colaboraciones).index(max_colaboraciones)]

print(f"El año en que hubo la mayor cantidad de colaboraciones fue {año_max_colaboraciones} con {max_colaboraciones}")

# %%
