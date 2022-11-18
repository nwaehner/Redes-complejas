#%%
import requests
from bs4 import BeautifulSoup as bs
import wikipedia
import warnings
import wikipediaapi
#%%
def generos_artistas_wikipedia(artista_nombre):
    wiki_wiki = wikipediaapi.Wikipedia(
            language='es',    
    )
    warnings.filterwarnings("ignore")
    try:
        page_py = wiki_wiki.page(artista_nombre)
    except wikipedia.exceptions.DisambiguationError as e:
        print('ambiguo')
        queries='\n'.join(str(e).split('\n')[1:])
        queries=queries.split('\n')
        page_py = wiki_wiki.page(queries[0])
    page = requests.get(page_py.fullurl)
    soup = bs(page.content, 'html.parser')
    ta = soup.find_all('table',class_="infobox biography vcard")[0].tbody
    for t in ta:
        if 'Género' in t.text or 'Géneros' in t.text:
            lista_generos = t.text.split()[1:]
    return lista_generos
# %%
generos_artistas_wikipedia('Mercedes Sosa')
# %%


