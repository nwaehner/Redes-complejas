#%%
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
# import plfit
from scipy.optimize import curve_fit

# Cargamos el multigrafo
with open(f"red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

#%%
def hacer_lista_grados(red): #devuelve una lista con los nodos de la red.
  lista_grados=[grado for (nodo,grado) in red.degree()]
  return lista_grados

lista_grados = np.array(hacer_lista_grados(G))

#%% Distribución de grado, con escala log y bineado log también.
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

#%%---------------------------Hacemos el ajuste--------------------------------
ajuste_grafo = plfit.plfit(lista_grados)
ajuste_grafo.plotpdf() #Grafica directamente el histograma, sin normalizar
# Grafica la frecuencia, no la probabilidad
Kmin = ajuste_grafo._xmin #este sería nuestro Kmin
gamma = ajuste_grafo._alpha #este sería nuestro gamma
plt.xlabel("Grado",fontsize = 16)
plt.ylabel("Cantidad de nodos",fontsize = 16)
print('Kmin = '+str(Kmin))
print('gamma = '+str(gamma))
p,sims = ajuste_grafo.test_pl(usefortran=True, niter=10000, nosmall=False)
print(f"El p_valor de la red es {p}")



#%%--------------------Análisis de la asortatividad por Barabási-----------------------

def f_ajuste(x, a, mu): return a*x**mu

# Creamos un dataframe con el grado y el grado medio de los vecinosdataframe 
df = pd.DataFrame(dict(
    GRADO = dict(G.degree),
    GRADO_MEDIO_VECINOS = nx.average_neighbor_degree(G),
    POPULARIDAD = {nodo: G.nodes()[nodo]["popularidad"] for nodo in G.nodes()}
    )) 

#calculo el grado medio por grado
grado_medio_por_grado = df.groupby(['GRADO'])['GRADO_MEDIO_VECINOS'].mean()

# Defino las variables x e y de lo que voy a ajustar
var_x = grado_medio_por_grado.index 
var_y = grado_medio_por_grado.values

# Utilizo curve_fit() para el ajuste
popt, pcov = curve_fit(f_ajuste, var_x, var_y)

# Imprimo en pantalla los valores de popt y pcov
a, mu = popt
err_a, err_mu = np.sqrt(np.diag(pcov))
print("Los parametros de ajuste son:")
print(f'a: {a} ± {err_a}')
print(f'mu: {mu} ± {err_mu}')

a, mu = popt
var_x_ajuste = np.linspace(1, max(var_x),100000)
var_y_ajuste = f_ajuste(var_x_ajuste, a, mu)



# Graficamos
fig, axs = plt.subplots(figsize = (8,6), facecolor='#f4e8f4')
axs.loglog(grado_medio_por_grado.index,grado_medio_por_grado.values,'.',color='#901c8e', alpha= 0.8, label = "Datos")
axs.loglog(var_x_ajuste,var_y_ajuste,color='g', alpha= 0.8, label = "Ajuste")
axs.grid('on', linestyle = 'dashed', alpha = 0.5)
axs.set_xlabel("Grado",fontsize = 16)
axs.set_ylabel("Grado medio de los vecinos",fontsize = 16)
axs.legend(fontsize = 16)
axs.tick_params(axis='both', which='major', labelsize=14)
plt.show()
#%%--------------------Análisis de la asortatividad de popularidad-----------------------
# Iteramos sobre los nodos y los vecinos para calcular la popularidad media de los vecinos

lista_nodos = G.nodes()
popularidad_media_vecinos = []

for nodo in lista_nodos:
    popularidad_media = 0
    lista_vecinos = list(G.neighbors(nodo))
    for vecino in lista_vecinos:
        popularidad_media += df["POPULARIDAD"][nodo]

    popularidad_media = popularidad_media/len(lista_vecinos)
    popularidad_media_vecinos.append(popularidad_media)
# Agregamos la columna al dataframe
df["POPULARIDAD_MEDIA_VECINOS"] = popularidad_media_vecinos

# Calculamos la popularidad media de los vecinos teniendo en cuenta la cant de colaboraciones
popularidad_media_vecinos_por_cancion = []
for nodo in lista_nodos:
    popularidad_media = 0
    lista_vecinos = list(G.neighbors(nodo))
    for vecino in lista_vecinos:
        cant_colaboraciones = len(G[nodo][vecino])
        popularidad_media += (df["POPULARIDAD"][nodo]*cant_colaboraciones)

    popularidad_media = popularidad_media/len(lista_vecinos)
    popularidad_media_vecinos_por_cancion.append(popularidad_media)
# Agregamos la columna al dataframe
df["POPULARIDAD_MEDIA_VECINOS_POR_CANCION"] = popularidad_media_vecinos_por_cancion

#%%------------------Pasamos a graficar la popularidad media de los vecinos-----------------------------------

#NO ME ACUERDO APRA QUE ESTABA LS SIGUIENTE LINEAAA, LA COPIE DE LO QUE HICIMOS PARA EL GRADO

#calculo la popularidad media por popularidad
popularidad_media_por_popularidad = df.groupby(['POPULARIDAD'])['POPULARIDAD_MEDIA_VECINOS'].mean()

# Defino las variables x e y de lo que voy a ajustar
var_x1 = popularidad_media_por_popularidad.index 
var_y1 = popularidad_media_por_popularidad.values

# Utilizo curve_fit() para el ajuste
popt, pcov = curve_fit(f_ajuste, var_x1, var_y1)

# Imprimo en pantalla los valores de popt y pcov
a1, mu1 = popt
err_a1, err_mu1 = np.sqrt(np.diag(pcov))
print("Los parametros de ajuste son:")
print(f'a1: {a1} ± {err_a1}')
print(f'mu1: {mu1} ± {err_mu1}')


var_x1_ajuste = np.linspace(1, max(var_x1),100)
var_y1_ajuste = f_ajuste(var_x1_ajuste, a1, mu1)

#calculo la popularidad media por cancion por popularidad
popularidad_media_por_cancion_por_popularidad = df.groupby(['POPULARIDAD'])['POPULARIDAD_MEDIA_VECINOS_POR_CANCION'].mean()

# Defino las variables x e y de lo que voy a ajustar
var_x2 = popularidad_media_por_cancion_por_popularidad.index 
var_y2 = popularidad_media_por_cancion_por_popularidad.values

# Utilizo curve_fit() para el ajuste
popt, pcov = curve_fit(f_ajuste, var_x2, var_y2)

# Imprimo en pantalla los valores de popt y pcov
a2, mu2 = popt
err_a2, err_mu2 = np.sqrt(np.diag(pcov))
print("Los parametros de ajuste son:")
print(f'a2: {a2} ± {err_a2}')
print(f'mu2: {mu2} ± {err_mu2}')

var_x2_ajuste = np.linspace(1, max(var_x2),100)
var_y2_ajuste = f_ajuste(var_x2_ajuste, a, mu2)

#%%------------------------Graficamos las asortatividades----------------------
fig, axs = plt.subplots(ncols = 2,figsize = (14,8), facecolor='#f4e8f4')
axs[0].loglog(popularidad_media_por_popularidad.index,popularidad_media_por_popularidad.values,'.',color='#901c8e', alpha= 0.8, label = "Datos")
axs[0].loglog(var_x1_ajuste,var_y1_ajuste,color='g', alpha= 0.8, label = "Ajuste")
axs[0].grid('on', linestyle = 'dashed', alpha = 0.5)
axs[0].set_xlabel("Popularidad",fontsize = 16)
axs[0].set_ylabel("Popularidad media de los vecinos",fontsize = 16)
axs[0].legend(fontsize = 16)
axs[0].tick_params(axis='both', which='major', labelsize=16)
axs[0].set_title("Sin considerar colaboraciones",fontsize = 18)

axs[1].loglog(popularidad_media_por_cancion_por_popularidad.index,popularidad_media_por_cancion_por_popularidad.values,'.',color='#901c8e', alpha= 0.8, label = "Datos")
axs[1].loglog(var_x2_ajuste,var_y2_ajuste,color='g', alpha= 0.8, label = "Ajuste")
axs[1].grid('on', linestyle = 'dashed', alpha = 0.5)
axs[1].set_xlabel("Popularidad",fontsize = 16)
axs[1].set_ylabel("Popularidad media de los vecinos",fontsize = 16)
axs[1].legend(fontsize = 16)
axs[1].tick_params(axis='both', which='major', labelsize=16)
axs[1].set_title("Considerando colaboraciones",fontsize = 18)
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
#%% HOMOFILIA POR RECOLOREO
lista_genero = [i[1]["genero"] for i in G.nodes(data=True)]
homofilia = []
n = 5000
for i in range(n): 
    random.shuffle(lista_genero) 
    for j, nodo in enumerate(list(G_copia.nodes())): 
        G_copia.nodes()[nodo]['genero'] = lista_genero[j] 
        
    homofilia.append(calcular_homofilia(G_copia)) 
#%% HOMOFILIA POR RECABLEO

from tqdm import tqdm
iteracion = 0
lista_nodos = list(G.nodes())
homofilia_recableo = []
clustering_recableo = []
for iteracion in tqdm(range(1000)):
  nueva_red = nx.double_edge_swap(G, nswap=len(list(G_copia.edges()))*2, max_tries=len(list(G_copia.edges()))*10)
  homofilia_recableo.append(calcular_homofilia(nueva_red))
  nueva_red_simple = nx.Graph()
  nueva_red_simple.add_nodes_from(lista_nodos)
  nueva_red_simple.add_edges_from(nueva_red.edges())
  clustering_recableo.append(nx.average_clustering(nueva_red_simple))
#%%
print(iteracion)
#%% GRAFICO HOMOFILIA RECOLOREO
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(homofilia, bins=20)
ax.hist(bins[:-1], bins, weights=counts/n, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(homofilia), ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = homofilia_real, ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Homofilia de la red original')
ax.fill_between(x = [np.mean(homofilia)-np.std(homofilia),np.std(homofilia)+np.mean(homofilia)], y1 = 0.3, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel("Homofilia", fontsize=12)
ax.set_ylabel("Frecuencia normalizada", fontsize=12)
plt.title("Homofilia por recoloreo (n = 5000)",fontsize = 18)
ax.legend(loc = 'best')
plt.savefig("Homofilia por recoloreo.png")
plt.show()
  

# %% GRAFICO HOMOFOLIA RECABLEO

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (14, 8), facecolor='#D4CAC6')
counts, bins = np.histogram(homofilia_recableo, bins=20)
ax.hist(bins[:-1], bins, weights=counts/iteracion, range = [0,1], rwidth = 0.80, facecolor='g', alpha=0.75)
ax.vlines(x = np.mean(homofilia_recableo), ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'r', label = 'Media')
ax.vlines(x = homofilia_real, ymin = 0, ymax = 0.3, linewidth = 3, linestyle = '--', alpha = 0.8, color = 'k', label = 'Homofilia de la red original')
ax.fill_between(x = [np.mean(homofilia_recableo)-np.std(homofilia_recableo),np.std(homofilia_recableo)+np.mean(homofilia_recableo)], y1 = 0.3, color = 'g', alpha = 0.4, label = 'Desviación estándar')
ax.grid('on', linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel("Homofilia", fontsize=12)
ax.set_ylabel("Frecuencia normalizada", fontsize=12)
plt.title("Homofilia por recableo (n = 1000)",fontsize = 18)
ax.legend(loc = 'best')
#plt.savefig("Homofilia por recableo.png")
plt.show()

# # %% CALCULO DE CLUSTERING POR RECABLEO
# clustering_recableo = pd.read_pickle("Clustering_por_recableo.pickle")

nueva_red_simple = nx.Graph()
nueva_red_simple.add_nodes_from(list(G.nodes()))
nueva_red_simple.add_edges_from(list(G.edges()))

print(f"El valor del clustering es {nx.average_clustering(nueva_red_simple)}")
print(f"El valor del clustering al recablear es de {np.mean(clustering_recableo)} +- {np.std(clustering_recableo)}")

# %%
pickle.dump(homofilia, open(f'Homofilia_por_recoloreo.pickle', 'wb'))
pickle.dump(homofilia_recableo, open(f'Homofilia_por_recableo.pickle', 'wb'))
pickle.dump(clustering_recableo, open(f'Clustering_por_recableo.pickle', 'wb'))
# %%
