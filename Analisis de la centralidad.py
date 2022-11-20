#%%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import random
import time
import pickle 
from tqdm import tqdm

plt.style.use("seaborn")
#%%-----------------------Cargamos el multigrafo---------------------------
i = 1496

with open(f"red_final/Iteracion {i}/red_final_hasta_indice_{i}.gpickle", "rb") as f:
    G = pickle.load(f)

#%%---------------------Paso la red a pesada para analizarla-------------------

G2 = nx.Graph()# Lo escribo así a cambio de perder la información de los enlaces
G2.add_nodes_from(G.nodes())
G2.add_edges_from(G.edges())

# %%
def centralidades(Red):
    df= pd.DataFrame(dict(
    Grado = dict(Red.degree),
    Centralidad = nx.degree_centrality(Red),
    Autovalores= nx.eigenvector_centrality(Red, max_iter=1000, tol=1e-06, nstart=None, weight='weight'),
    Intermediatez = nx.betweenness_centrality(Red,k=None, normalized=True, weight=None, endpoints=False, seed=None),
    Cercania = nx.closeness_centrality(Red, u=None, distance=None, wf_improved=True),
    Popularidad = {nodo: G.nodes()[nodo]["popularidad"] for nodo in G.nodes()}))
    return df

df_centralidad = centralidades(G2)   
#%%
def armar_componente_gigante(Red):
    Conjunto_nodos_en_gigante = max(nx.connected_components(Red), key=len)
    Componente_Gigante = Red.subgraph(Conjunto_nodos_en_gigante).copy()
    return Componente_Gigante

def long_segunda_comp_gigante(Red):
    lista_componentes=[Red.subgraph(componente) for componente in sorted(nx.connected_components(Red), key=len, reverse=True)]
    segunda_comp_gigante =  lista_componentes[1]
    return len(segunda_comp_gigante)


def desarmar_red(red,lista_sacar):
    componente_gigante = armar_componente_gigante(red)
    tamano_inicial = len(componente_gigante.nodes())
    tamano_componente = tamano_inicial
    fraccion_en_gigante = [1] # Defino que empiece en 1 para tener el valor inicial
    fraccion_quitados = [0] # Defino que empiece en 0 para tener el valor inicial
    
    i = 0

    while tamano_inicial/2 < tamano_componente:
        # Tomamos el nodo a sacar usando la funcion pasada
        nodo_sacar = lista_sacar[i]
        # Lo sacamos
        if nodo_sacar in componente_gigante.nodes():
            componente_gigante.remove_node(nodo_sacar)
        # Nos quedamos con la nueva componente gigante
        componente_gigante = armar_componente_gigante(componente_gigante)
        # Calculamos el tamaño de la componente gigante
        tamano_componente = len(componente_gigante.nodes())
        # Guardamos la fracción que queda
        fraccion_restante = (tamano_componente)/tamano_inicial
        fraccion_en_gigante.append(fraccion_restante)
        # Guardamos la fracción que ya sacamos
        fraccion_quitados.append(i/tamano_inicial)
        i +=1
    return fraccion_quitados, fraccion_en_gigante

#%% Calculamos la ruptura por aleatoriedad
frac_quit_random_iterada = []
frac_en_gig_random_iterada = []

for i in tqdm(range(100)):

    lista_random = list(G2.nodes())
    random.shuffle(lista_random)
    frac_quit_random, frac_en_gig_random = desarmar_red(G2,lista_random) 
    frac_quit_random_iterada.append(np.array(frac_quit_random))
    frac_en_gig_random_iterada.append(np.array(frac_en_gig_random))

#Redefino la lista para q todos tengan el minimo de longitud(sino no puedo sumar los array):

minimo_quit = min([len(i) for i in frac_quit_random_iterada])

frac_quit_random = [i[0:minimo_quit] for i in frac_quit_random_iterada]
frac_en_gig_random = [i[0:minimo_quit] for i in frac_en_gig_random_iterada]

#Creo mis listas para plotear: promedio sobre todas las random

frac_quit_random_x = sum(np.array(frac_quit_random))/len(frac_quit_random)
frac_en_gig_random_y = sum(np.array(frac_en_gig_random))/len(frac_en_gig_random)

#%%
fig, axs = plt.subplots(figsize = (12, 8))
for columna in tqdm(df_centralidad.columns[0:2]):
    lista_centralidad = df_centralidad.sort_values(by = columna, ascending = False).index

    frac_quitados, frac_en_gigante = desarmar_red(G2, lista_centralidad)
    
    axs.plot(frac_quitados, frac_en_gigante, label = columna)


axs.plot(frac_quit_random_x, frac_en_gig_random_y,label = "Aleatorio",c = "k")
axs.grid(True)
axs.set_xlabel("Fracción de nodos quitados",fontsize = 16)
axs.set_ylabel("Fracción de nodos en la componente gigante", fontsize = 16)
axs.legend(fontsize = 16)
axs.tick_params(axis='both', which='major', labelsize=14)
plt.show()
# %%
