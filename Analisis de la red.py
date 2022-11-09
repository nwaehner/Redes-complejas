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

#%%
#RESHUFFLEAR LOS ENLACES RANDOM Y RECOLOREAR. EN AMBOS CASOS CALCULAR HOMOFILIA.

homofilia_numerador = 0

for enlace in G.edges(data=True):
    genero_artista1 = G.nodes()[enlace[0]]["genero"]
    genero_artista2 = G.nodes()[enlace[1]]["genero"]
    
    if genero_artista1 == genero_artista2:
        homofilia_numerador += 1

print(homofilia_numerador/len(G.edges()))

#%%


# %%
