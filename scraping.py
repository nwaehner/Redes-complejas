#%%
import requests
from bs4 import BeautifulSoup as bs
import wikipedia as wiki
import warnings
import wikipediaapi
#%%
def obtener_genero(artista_nombre):
    lista_generos = []
    wiki_wiki = wikipediaapi.Wikipedia(
            language='es',    
    )
    warnings.filterwarnings("ignore")
    id_artista = wiki.search(artista_nombre)[0]
    try:
        page_py = wiki_wiki.page(id_artista)
    except wiki.exceptions.DisambiguationError as e:
        queries='\n'.join(str(e).split('\n')[1:])
        queries=queries.split('\n')
        page_py = wiki_wiki.page(queries[0])
    if page_py.exists():
        try:
            page = requests.get(page_py.canonicalurl)
            soup = bs(page.content, 'html.parser')
            ta = soup.find_all('table',class_="infobox biography vcard")[0].tbody
            for t in ta:
                if 'Género' in t.text or 'Géneros' in t.text:
                    lista_generos = t.text.splitlines()[1:]
            lista_generos.remove("")
        except:
            print(artista_nombre, ' NO ENCONTRO GENERO')
    return lista_generos
# %%


