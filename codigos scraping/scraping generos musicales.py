#%%-----------------------------------------------------------------
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import wikipedia as wiki
import warnings
import wikipediaapi
import networkx as nx
import pickle
from tqdm import tqdm 
from unicodedata import normalize

plt.style.use("seaborn")
#%%

def normalizar(string):
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    string_normalizado = normalize('NFKC', normalize('NFKD', string).translate(trans_tab)).lower().replace(" ", "")
    return string_normalizado


#%%
def obtener_genero(artista_nombre):
    #obtengo genero musicales de artista scrapeando
    lista_generos = []
    wiki_wiki = wikipediaapi.Wikipedia(
            language='es',    
    )
    warnings.filterwarnings("ignore")
    try:
        #usa el buscador de wikipedia para buscar el artista y agarra primer elemento
        id_artista = wiki.search(artista_nombre)[0]
    except:
        print(f'{artista_nombre} NO TIENE WIKIPAGE')
        return lista_generos
    try:
        page_py = wiki_wiki.page(artista_nombre)
    except wiki.exceptions.DisambiguationError as e:
        queries='\n'.join(str(e).split('\n')[1:])
        queries=queries.split('\n')
        page_py = wiki_wiki.page(queries[0])
    if page_py.exists():
        #si la pagina del artista existe, busco la tabla de contenidos
        #y me fijo si dice 'Género' o 'Géneros' en algun lado
        #y levanto el texto
        try:
            page = requests.get(page_py.fullurl)
            soup = bs(page.content, 'html.parser')
            ta = soup.find_all('table',class_="infobox biography vcard")[0].tbody
            for t in ta:
                if 'Géneros' in t.text:
                    lista_generos = t.text.splitlines()
                    lista_generos.remove('Géneros')
                elif 'Género' in t.text:
                    lista_generos = t.text.splitlines()
                    lista_generos.remove('Género')    
        except:
            print('NO ENCONTRÓ GENERO')
    
    if "" in lista_generos:
        lista_generos.remove("")
    return lista_generos

def es_realmente_argentino(artista_nombre):
    es_argentino = False
    wiki_wiki = wikipediaapi.Wikipedia(language='es',)
    warnings.filterwarnings("ignore")
    try:
        id_artista = wiki.search(artista_nombre)[0]
    except:
        print(f'{artista_nombre} NO TIENE WIKIPAGE')
        return es_argentino
    try:
        page_py = wiki_wiki.page(id_artista)
    except wiki.exceptions.DisambiguationError as e:
        queries='\n'.join(str(e).split('\n')[1:])
        queries=queries.split('\n')
        page_py = wiki_wiki.page(queries[0])
    if page_py.exists():
        try:
            sumario = page_py.summary[0:200]
            print(sumario)
            if 'argentino' in sumario or 'argentina' in sumario or 'Argentina' in sumario :
                es_argentino = True
            else:
                page = requests.get(page_py.fullurl)
                soup = bs(page.content, 'html.parser')
                ta = soup.find_all('table',class_="infobox biography vcard")[0].tbody
                for t in ta:
                    if 'Nacionalidad' in t.text:
                        if 'argentino' in t.text or 'Argentina' in t.text:
                            es_argentino = True
                        break
                    elif 'Nacimiento' in t.text:
                        if 'Argentina' in t.text:
                            es_argentino = True
                            break
        except:
            print(f'{artista_nombre} dudoso q sea argentino mmm')
    else:
        print('no existe la pagina de duki')
    return es_argentino
#%%-----------------------Cargamos el multigrafo---------------------------
with open(f"red_filtrada/red_filtrada.gpickle", "rb") as f:
    G = pickle.load(f)

#%%--------Me armo esta celda para agregar géneros musicales con wikipedia----------
lista_nodos = list(G.nodes())
for i,nodo in (enumerate(lista_nodos)):
    print(i, nodo)
    # Agregamos los generos del scrap de wikipedia
    G.nodes()[nodo]["generos_musicales"].extend(obtener_genero(nodo))
    
    # Pedimos que los géneros sean unicos
    generos_normalizados = [normalizar(i) for i in G.nodes()[nodo]["generos_musicales"]]
    G.nodes()[nodo]["generos_musicales"] = list(np.unique(generos_normalizados))

#%%
nodos_sin_etiquetas = []
for nodo in lista_nodos:
    try: 
        G.nodes()[nodo]["generos_musicales"]
        if len(G.nodes()[nodo]["generos_musicales"])==0:
            nodos_sin_etiquetas.append(nodo)
    except:
        pass
print(len(nodos_sin_etiquetas)/1496)
#%%
nx.write_gpickle(G, f"red_filtrada/red_filtrada.gpickle")

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

G_copia = G.copy()
#%%
nodos_sin_etiquetas = []  #artistas que no tienen genero musical
for nodo in lista_nodos:
    generos_nuevos =[]
    generos_spotify = (G_copia.nodes())[nodo]["generos_musicales"]
    # Recorremos los generos que creemos que son importantes
    for genero in generos_representativos:
        # Recorremos los generos de spotify del cantante
        for genero_spotify in generos_spotify:
            # Comprobamos que el genero entre en alguno de los generos de spotify
            if genero.lower().replace(" ", "") in genero_spotify.lower().replace(" ", ""):
                generos_nuevos.append(genero)
    
    generos_nuevos = list(np.unique(generos_nuevos))
    try:
        G_copia.nodes[nodo]["label"] = generos_nuevos[0]
    except:
        nodos_sin_etiquetas.append(nodo)
        

print(len(nodos_sin_etiquetas))
#%%
for nodo in G_copia.nodes(data=True):
    print(nodo[1]["label"])
#%%
from networkx.algorithms import node_classification
lista_labels=node_classification.harmonic_function(G_copia)
#IDEAS:
#cada genero como una coordenada
#scrapear wiki
#testear con una parte de los que tienen etiquetas 
#clusterizar artistas por vectores con sus generos (kmeans etc)
#(PCA o SVD) reduccion de dimensionalidad >> clusterizar 

#prediccion de colaboraciones:
#sacar enlaces, uso embedding de nodos para predecir esos enlaces
#a partir de similaridad 
#ultima clase de embbeding de nodos 
# PREGUNTAR A NICO (borrar enlaces antes de pasar al espacio metrico sin sacar nodos)
#link prediction    
#preguntat grupo de netflix?

#Ariel dijo: sacar enlaces y luego calcular similaridad entre nodos (con una matriz).
# Ver si los nodos con mayor similaridad estaban enlazados y ahora no e iterar sacando
#distintos enlaces. calcular finalmente probabilidad.

#
#Ariel:
#difusion de a un genero por vez y sumar, ver el mas representativo por nodo. Para esto usar laplaciano, ver clase
#para homofilia, sacar los que no tiene etiqueta.

for i,nodo in enumerate(G_copia.nodes(data=True)):
    #asigno la etiqueta para cada nodo
    nodo[1]["label"] = lista_labels[i]
for nodo in G_copia.nodes(data=True):
    if nodo[0] in nodos_sin_etiquetas:
        print(f'{nodo[0]} toca {nodo[1]["label"]} ')  
#%%
for nodo in G_copia.nodes(data=True):
    print(f"{nodo[0]} tiene categoria {nodo[1]['label']}")

#%%
labels = nx.get_node_attributes(G_copia, 'label')
labels["Bizarrap"]


def asignar_indice(atributo,lista):
    np.where(np.array(lista) == atributo)
    
print(asignar_indice('Classic', generos_representativos))
#%%
plt.figure(figsize=(18,12))
nx.draw(G_copia,node_color=[plt.get_cmap('tab20')(generos_representativos.index(v[1]['label'])) for v in G_copia.nodes(data = True)])


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