#%%
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import json
import re
from multiprocessing import Queue
import networkx as nx
import musicbrainzngs as mb 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import wikipedia as wiki
import time
from unicodedata import normalize
import pandas as pd
import random 
i = 1496
# Cargamos el multigrafo
with open(f"red_final/Iteracion {i}/red_final_hasta_indice_{i}.gpickle", "rb") as f:
    G = pickle.load(f)
#%%
def hacer_lista_grados(red): #devuelve una lista con los nodos de la red.
  lista_grados=[grado for (nodo,grado) in red.degree()]
  return lista_grados

lista_grados = hacer_lista_grados(G)

#Distribución de grado, con escala log y bineado log también.
bins = np.logspace(np.log10(1),np.log10(max(lista_grados)), 15)
plt.hist(lista_grados, bins = bins, color='#901c8e',rwidth = 0.80, alpha= 0.8)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Grado",fontsize = 16)
plt.ylabel("Cantidad de nodos",fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title("Distribución de grado (log-log)")
# plt.savefig("distribucion de grado.png")
plt.show()



#%%
#RESHUFFLEAR LOS ENLACES RANDOM Y RECOLOREAR. EN AMBOS CASOS CALCULAR HOMOFILIA.



def calcular_homofilia(red):
  homofilia_numerador = 0
  for enlace in red.edges(data=True):
      genero_artista1 = red.nodes()[enlace[0]]["genero"]
      genero_artista2 = red.nodes()[enlace[1]]["genero"]
      
      if genero_artista1 == genero_artista2:
          homofilia_numerador += 1

  return homofilia_numerador/len(red.edges())

homofilia_real = calcular_homofilia(G)

#%%
G_copia = G.copy()
#%%
lista_genero = [i[1]["genero"] for i in G.nodes(data=True)]

homofilia = []
n = 5000

for i in range(n): 
    random.shuffle(lista_genero) 
    for j, nodo in enumerate(list(G_copia.nodes())): 
        G_copia.nodes()[nodo]['genero'] = lista_genero[j] 
        
    homofilia.append(calcular_homofilia(G_copia)) 


#%%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6), facecolor='#D4CAC6')
counts, bins = np.histogram(homofilia, bins=20)
ax.hist(bins[:-1], bins, weights=counts/n, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(homofilia), ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = homofilia_real, ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Homofilia de la red original')
ax.fill_between(x = [np.mean(homofilia)-np.std(homofilia),np.std(homofilia)+np.mean(homofilia)], y1 = 0.3, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel("Homofilia", fontsize=12)
ax.set_ylabel("Frecuencia normalizada", fontsize=12)
ax.set_ylim(0,0.2)
ax.set_yticks(np.arange(0,0.21,0.05))
ax.set_xticks(np.arange(0.3,0.8,0.1))
plt.title("Homofilia por recoloreo (n = 5000)",fontsize = 18)
ax.set_xlim(0.3,0.8)
ax.legend(loc = 'best')
# plt.savefig("Homofilia por recoloreo.png")
plt.show()


#%% CODIGO PRUEBA DE SUFFLEAR ENLACES
red_random1 = nx.MultiGraph()
red_random1.add_edges_from([(0, 2), (0, 3), (0, 1), (1, 2), (1, 3), (2, 3)])

nx.draw(red_random1,with_labels = True)
#%%
# G_copia = G.copy()

def shuflear_enlaces(red):
  enlace1 = random.choice(list(red.edges()))
  enlace2 = random.choice(list(red.edges()))
  print("enlaces a remover:",[enlace1,enlace2])
  
  if enlace1 != enlace2 and enlace1[0] != enlace2[1] and enlace1[1] != enlace2[0]:
    enlaces_nuevos = [(enlace1[0],enlace2[1]),(enlace1[1],enlace2[0])]
    print("enlaces a agregar:", enlaces_nuevos)
    print("enlaces iniciales:", red.edges())
    red.remove_edges_from([enlace1,enlace2])
    print("enlaces cuando saco:", red.edges())
    red.add_edges_from(enlaces_nuevos)
    print("enlaces cuando agrego:",red.edges())
  
  return red

  

# %%
red_random = shuflear_enlaces(red_random1)
nx.draw(red_random,with_labels = True)
# %%
