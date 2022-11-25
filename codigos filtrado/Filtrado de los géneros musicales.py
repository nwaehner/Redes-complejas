#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
#%% Cargamos la red
with open('../red_filtrada/red_filtrada.gpickle', "rb") as f:
    G = pickle.load(f)
#%% Obtenemos una lista con los géneros
lista_generos = []
for nodo in G.nodes():
    lista_generos_nodo = G.nodes()[nodo]["generos_musicales"]

    lista_generos.extend(lista_generos_nodo)

lista_generos = np.unique(lista_generos)
#%%


generos_representativos = ["Trap","Jazz", "Pop", "HipHop", "Clasico", "Indie",
                           "R&B", "Tango", "Cumbia",
                           "Chamame", "Electronica", "Folklore", "Rock", "Punk", "Rap", "Metal",
                           "Reggae","Alternative"]

# CUIDADO RAP Y TRAP
# 
string_generos = " ".join(generos_representativos)


#%% 
dic_generos = {}
lista_generos_filtrados = []
for nodo in G.nodes():
    lista_generos_nodo = G.nodes()[nodo]["generos_musicales"]

    for genero in lista_generos_nodo:

        if genero in dic_generos.keys():
            dic_generos[genero] += 1
        else:
            dic_generos[genero] = 1
    
#%%
dic_generos_ordenado = dict(sorted(dic_generos.items(), key = lambda x:x[1], reverse=True))
# %%
for i in dic_generos_ordenado.items():
    coincide = False
    for genero in generos_representativos:
        genero = genero.lower()
        if genero in i[0]:
            coincide = True
            
    if coincide == False:
        print(i)
#%%
dic_de_mergeos = {"Electronica":  ["Tronica","edm","house","electro","dubstep","glitchbeats","techno","elec","ambient"]
                  ,"Cumbia":["previa","rkt","cumbia"]
                  ,"Rap": ["basesdefreestyle","rap"]
                  ,"Folklore": ["folklore","folclore","trova","zamba"]
                  ,"Clasico": ["clas","opera"]
                  
                  }
# %%

dic_artistas_por_genero = {"Trap":[],"Jazz":[], "Pop":[], "HipHop":[], "Clasico":[], "Indie":[],
                           "R&B":[], "Tango":[], "Cumbia":[],  
                           "Chamame":[], "Electronica":[], "Folklore":[], "Rock":[], "Punk":[], "Rap":[], "Metal":[],
                           "Reggae":[],"Alternative":[]}

for nodo in G.nodes():
    generos_artista = G.nodes()[nodo]["generos_musicales"]
    
    for genero in generos_artista: #recorro los generos del artista
        for genero_dic,lista_dic in dic_de_mergeos.items(): #recorro primero estos generos pq tienen varias polabras asociadas.
            lista_artistas_por_genero = dic_artistas_por_genero[genero_dic] #palabras asociadas al genero
            if (any(genero in s for s in lista_dic)) and (nodo not in lista_artistas_por_genero):
                #lo meto si alguna palabra está y si el nodo no esta en la lista.
                lista_artistas_por_genero.append(nodo)
        
        for genero_repr in generos_representativos: #recorro los demas generos
            lista_artistas_por_genero = dic_artistas_por_genero[genero_repr]
            if (nodo not in lista_artistas_por_genero):
                if (genero_repr == "Trap") and ("trap" in genero): #
                    lista_artistas_por_genero.append(nodo)
                    
                elif (genero_repr == "Rap") and ("trap" not in genero) and ("rap" in genero):
                    #print(nodo)
                    lista_artistas_por_genero.append(nodo)
                    
                elif (genero_repr.lower() in genero) and (genero != "reggaeton") and (genero_repr != "Rap"):
                    lista_artistas_por_genero.append(nodo)
        
    
#%%
for value,lista in dic_artistas_por_genero.items(): #ELIMINO INDIE PQ NOS DIMOS CUENTA DE Q SON ALTERNATIVE XDDDDDDD
    if value == "Indie":
        dic_artistas_por_genero["Alternative"].extend(lista)
        dic_artistas_por_genero["Alternative"] = list(np.unique(dic_artistas_por_genero["Alternative"]))
dic_artistas_por_genero.pop("Indie")

#%% LE CAMBIAMOS LOS GENEROS MUSICALES A LA RED EN G_COPIA.
G_copia = G.copy()

#%%
for i in G_copia.nodes():
    
    