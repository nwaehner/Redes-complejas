#%%
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
import copy
# import plfit
from scipy.optimize import curve_fit
from tqdm import tqdm
import seaborn as sns

# Cargamos el multigrafo
with open("../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

#%% Agregamos pesos a los enlaces

def crear_red_pesada(red):

    G_pesada = nx.Graph()
    enlaces_chequeados = []
    lista_enlaces = list(red.edges())
    for i,enlace_i in (enumerate(lista_enlaces)):
        peso_enlace = 1
        if enlace_i not in enlaces_chequeados:
            for j in range(i+1,len(lista_enlaces)):
                enlace_j = lista_enlaces[j]
                if enlace_i == enlace_j:
                    peso_enlace +=1
        
            enlaces_chequeados.append(enlace_i)
            if enlace_i in G_pesada.edges():
                G[enlace_i[0]][enlace_i[1]]["weight"] += 1
            else:  
                G_pesada.add_edge(enlace_i[0],enlace_i[1], weight = peso_enlace)
            
    return G_pesada

#%% Definimos funciones para calcular la homofilia según los géneros (hombre, mujer,etc) y los géneros musicales

def calcular_homofilia(red):
    homofilia_numerador = 0
    for enlace in red.edges(data=True):
        
        genero_artista1 = red.nodes()[enlace[0]]["genero"]
        genero_artista2 = red.nodes()[enlace[1]]["genero"]
      
        if genero_artista1 == genero_artista2:
            homofilia_numerador += 1

    return homofilia_numerador/len(red.edges())

def calcular_homofilia_generos_musicales(red):
    homofilia_numerador = 0
    for enlace in red.edges(data=True):
        # print(enlace)
        genero_artista1 = red.nodes()[enlace[0]]["generos_musicales"]
        genero_artista2 = red.nodes()[enlace[1]]["generos_musicales"]
        if set(genero_artista1).intersection(set(genero_artista2)) != set():
            homofilia_numerador += 1
            
    return homofilia_numerador/len(red.edges())

homofilia_real = calcular_homofilia(G)
homofilia_generos = calcular_homofilia_generos_musicales(G)

#%%------------Recoloreamos la red para comparar con un modelo nulo------------------
# Creamos una copia de la red para recolorear
G_copia = copy.deepcopy(G) 
# Creamos una lista con los géneros
lista_genero = [i[1]["genero"] for i in G.nodes(data=True)]
# Guardamos una lista con los valores de homofilia, para despues hacer un histograma
homofilia = []
# Iteramos las veces que queramos
n = 5000
for i in tqdm(range(n)): 
    # Mezclamos la lista de géneros
    random.shuffle(lista_genero) 
    # Reasignamos los géneros a los artistas
    for j, nodo in enumerate(list(G_copia.nodes())): 
        G_copia.nodes()[nodo]['genero'] = lista_genero[j] 
    # Guardamos el valor de homofilia en la lista
    homofilia.append(calcular_homofilia(G_copia)) 
#%%------------------Recableamos la red para comparar con un modelo nulo----------------

iteracion = 0
# Hacemos una lista con los nodos/artistas
lista_nodos = list(G_copia.nodes())
# Creamos listas para las homofilias y los clusterings de cada red recableada
homofilia_recableo = []
homofilia_recableo_generos_musicales = []
clustering_recableo = []
# Iteramos
for iteracion in tqdm(range(1000)):
    # Copiamos la red original
    G_copia = copy.deepcopy(G)
    # Recableamos la red 
    nueva_red = nx.double_edge_swap(G_copia, nswap=len(list(G_copia.edges()))*4, max_tries=len(list(G_copia.edges()))*10)
    # Calculamos homofilia, homofilia por género musical y clustering global
    homofilia_recableo.append(calcular_homofilia(nueva_red))
    homofilia_recableo_generos_musicales.append(calcular_homofilia_generos_musicales(nueva_red))
    # Transformamos la red a pesada ya que la funcion no sirve para la red multienlace
    G_pesada = crear_red_pesada(G_copia)
    clustering_recableo.append(nx.average_clustering(G_pesada, weight = "weight"))


#%%--------------Guardamos los datos por si tardamos mucho en conseguirlos---------

with open("../datos analisis/Homofilia_por_recoloreo.pickle", "rb") as f:
    homofilia = pickle.load(f)
with open("../datos analisis/Homofilia_por_recableo.pickle", "rb") as f:
    homofilia_recableo = pickle.load(f)
with open("../datos analisis/Homofilia_generos_musicales_por_recableo.pickle", "rb") as f:
    homofilia_recableo_generos_musicales = pickle.load(f)

#%%---------------------------Graficamos el coeficiente de clustering----------------
print(iteracion)
G_pesada_original = crear_red_pesada(G)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8))
counts, bins = np.histogram(clustering_recableo, bins=20)
sns.histplot(clustering_recableo, bins=20,ax=ax,stat = "probability")
ax.vlines(x = np.mean(clustering_recableo), ymin = 0, ymax = 0.5, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'm', label = 'Media')
ax.vlines(x = nx.average_clustering(G_pesada_original, weight = "weight"), ymin = 0, ymax = 0.5, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Clustering de la red original')
ax.fill_between(x = [np.mean(clustering_recableo)-np.std(clustering_recableo),np.std(clustering_recableo)+np.mean(clustering_recableo)], y1 = 0.5, color = 'b', alpha = 0.2, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_ylim(0,0.2)
ax.set_xlabel("Clustering", fontsize=25)
ax.set_ylabel("Frecuencia normalizada", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=16)
#axs.title("Clustering por recableo (n = 1000)",fontsize = 30)
ax.legend(loc = 'upper center',fontsize = 18)
plt.savefig("../imagenes del analisis/Clustering.png")
plt.show()

#%%---------------------Graficamos la homofilia de género por recoloreo---------------------

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(homofilia, bins=20)
ax.hist(bins[:-1], bins, weights=counts/5000, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(homofilia), ymin = 0, ymax = 0.2, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = homofilia_real, ymin = 0, ymax = 0.2, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Homofilia de la red original')
ax.fill_between(x = [np.mean(homofilia)-np.std(homofilia),np.std(homofilia)+np.mean(homofilia)], y1 = 0.2, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel("Homofilia", fontsize=25)
ax.set_ylabel("Frecuencia normalizada", fontsize=25)
plt.title("Homofilia por recoloreo (n = 5000)",fontsize = 30)
ax.legend(loc = 'best')
plt.savefig("Homofilia por recoloreo.png")
plt.show()
  

#%%-------------------Graficamos la homofilia de género por recableo------------------

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(homofilia_recableo, bins=20)
ax.hist(bins[:-1], bins, weights=counts/1000, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(homofilia_recableo), ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = homofilia_real, ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Homofilia de la red original')
ax.fill_between(x = [np.mean(homofilia_recableo)-np.std(homofilia_recableo),np.std(homofilia_recableo)+np.mean(homofilia_recableo)], y1 = 0.3, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_ylim(0,0.2)
ax.set_xlabel("Homofilia", fontsize=25)
ax.set_ylabel("Frecuencia normalizada", fontsize=25)
plt.title("Homofilia por recableo (n = 1000)",fontsize = 30)
ax.legend(loc = 'best')
plt.savefig("Homofilia por recableo.png")
plt.show()

#%%------------------Graficamos la homofilia por géneros musicales-------------------

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(homofilia_recableo_generos_musicales, bins=20)
ax.hist(bins[:-1], bins, weights=counts/1000, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(homofilia_recableo_generos_musicales), ymin = 0, ymax = 0.5, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = homofilia_generos, ymin = 0, ymax = 0.5, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Homofilia de la red original')
ax.fill_between(x = [np.mean(homofilia_recableo_generos_musicales)-np.std(homofilia_recableo_generos_musicales),np.std(homofilia_recableo_generos_musicales)+np.mean(homofilia_recableo_generos_musicales)], y1 = 0.5, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_ylim(0,0.2)
ax.set_xlabel("Homofilia", fontsize=25)
ax.set_ylabel("Frecuencia normalizada", fontsize=25)
plt.title("Homofilia géneros musicales por recableo (n = 1000)",fontsize = 30)
ax.legend(loc = 'best')
plt.savefig("Homofilia generos musicales por recableo.png")
plt.show()


#%%------------------Ejemplo de como cargamos los datos para clustering-----------------
with open("../datos analisis/Clustering_por_recableo.pickle", "rb") as f:
    clustering_recableo = pickle.load(f)
    
G_pesada = crear_red_pesada(G)

print(f"El valor del clustering es {nx.average_clustering(G_pesada, weight = 'weight')}")
print(f"El valor del clustering al recablear es de {np.mean(clustering_recableo)} +- {np.std(clustering_recableo)}")

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(clustering_recableo, bins=20)
ax.hist(bins[:-1], bins, weights=counts/1000, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(clustering_recableo), ymin = 0, ymax = 0.5, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = nx.average_clustering(G_pesada, weight = "weight"), ymin = 0, ymax = 0.5, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Clustering de la red original')
ax.fill_between(x = [np.mean(clustering_recableo)-np.std(clustering_recableo),np.std(clustering_recableo)+np.mean(clustering_recableo)], y1 = 0.5, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_ylim(0,0.2)
ax.set_xlabel("Clustering", fontsize=25)
ax.set_ylabel("Frecuencia normalizada", fontsize=25)
plt.title("Clustering por recableo (n = 1000)",fontsize = 30)
ax.legend(loc = 'best')
plt.savefig("Clustering.png")
plt.show()

# %%
pickle.dump(homofilia, open(f'Homofilia_por_recoloreo.pickle', 'wb'))
pickle.dump(homofilia_recableo, open(f'Homofilia_por_recableo.pickle', 'wb'))
pickle.dump(clustering_recableo, open(f'Clustering_por_recableo.pickle', 'wb'))
pickle.dump(homofilia_recableo_generos_musicales, open(f'Homofilia_generos_musicales_por_recableo.pickle', 'wb'))
# %%
