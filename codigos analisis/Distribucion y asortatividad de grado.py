#%%
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
import plfit
from scipy.optimize import curve_fit

# Cargamos el multigrafo
with open("../red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)
#%%---------Creamos una tabla con datos generales de nuestra red------------------------
G_simple = nx.Graph(G)
df = pd.DataFrame({'n° nodos': np.round(G.number_of_nodes(),0),
                'n° enlaces': np.round(G.number_of_edges(),0),
                'Grado medio': np.round(np.mean([G.degree(n) for n in G.nodes()]),2),
                'Coeficiente de clustering medio': np.round(nx.average_clustering(G_simple),2),
                "Distancia media entre nodos": np.round(nx.average_shortest_path_length(G),2),
                "Diámetro": nx.diameter(G)},
                index = ["Red"])
df = df.transpose()
display(df)

#%%----------Creamos una lista con los grados de los nodos-----------------
def hacer_lista_grados(red): #devuelve una lista con los nodos de la red.
  lista_grados=[grado for (nodo,grado) in red.degree()]
  return lista_grados

lista_grados = np.array(hacer_lista_grados(G))

#%% Distribución de grado, con escala log y bineado log también.
fig, axs = plt.subplots(figsize = (6,4),facecolor='#f4e8f4')
bins = np.logspace(np.log10(1),np.log10(max(lista_grados)), 15)
axs.hist(lista_grados, bins = bins, color='#901c8e',rwidth = 0.80, alpha= 0.8)
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_xlabel("Grado",fontsize = 16)
axs.set_ylabel("Cantidad de nodos",fontsize = 16)
axs.tick_params(axis='both', which='major', labelsize=14)
axs.set_title("Distribución de grado (log-log)")
plt.savefig("distribucion de grado.png")
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
#%% Graficamos
fig, axs = plt.subplots(figsize = (8,4))
axs.loglog(grado_medio_por_grado.index,grado_medio_por_grado.values,'.',color='#901c8e', alpha= 0.8, label = "Datos")
axs.loglog(var_x_ajuste,var_y_ajuste,color='g', alpha= 0.8, label = "Ajuste")
axs.grid('on', linestyle = 'dashed', alpha = 0.5)
axs.set_xlabel("Grado",fontsize = 16)
axs.set_ylabel("Grado medio de los vecinos",fontsize = 16)
axs.legend(fontsize = 16)
axs.tick_params(axis = "both", labelsize = 16)

#axs.xaxis.label.set_color('white')
#axs.yaxis.label.set_color('white')
plt.savefig("../imagenes del analisis/Asortatividad de grado.png", bbox_inches = 'tight')
plt.show()
#%%---------------------Calculamos asortatividad de Newman-------------------
dict_grados = G.degree()
enlaces = G.edges()

S_e = 2*sum([dict_grados[i[0]]*dict_grados[i[1]] for i in enlaces]) 
S_1 = sum(np.array(lista_grados,dtype = np.float64))
S_2 =  sum(np.array(lista_grados, dtype=np.float64)**2)
S_3 =  sum(np.array(lista_grados, dtype=np.float64)**3)


# Se calcula de esta manera porque sino son enteros muy grandes para python
r = S_e/(S_1*S_3-S_2**2)
r = (r*S_1)-((S_2**2)/(S_1*S_3-S_2**2))

print(f"El r Newman de la red  es {r}")
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
        cant = len(G[nodo][vecino])
        popularidad_media += (df["POPULARIDAD"][nodo]*cant)

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
