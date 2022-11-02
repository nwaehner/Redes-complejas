import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import re

CLIENT_ID = "3a533c6fd3434a29a09896712d5c19bd"
CLIENT_SECRET = "65f9ad113dd54f6091cee7aa5498568b"

def conseguir_id(artista):
    # POST donde le pasamos las clave de la app
    response = requests.post('https://accounts.spotify.com/api/token', data = {'grant_type': 'client_credentials', 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET})
    # Guardamos el bearer token para usarlo en las peticiones de la API
    access_token = response.json()['access_token']
    headers = {'Authorization': 'Bearer {}'.format(access_token), 'Accept': 'application/json', 'Content-Type': 'application/json'}

    """
    Recibe
        artista: Nombre del artista del cual se quiere su ID
    Devuelve:
        id: La ID del artista
    """
    # End point para obtener los audio features. Esto se saca de la referencia de la documentación
    url = 'https://api.spotify.com/v1/search'

    # Búsqueda. Acá sí hay que pasarselo como parámetros
    params = {'q': f"{artista}", 'type': 'artist', 'limit': '2'}

    # En este caso no lleva ningún parámetro, el id de la canción va directamente en el url
    response = requests.get(url, params = params, headers = headers)

    # Vemos el json de la respuesta
    json_data = response.json()
    
    # Con la siguiente lista conseguimos el ID del primer resultado de la busqueda (creo)
    id = json_data["artists"]["items"][0]["id"]

    return id




# Ambas formas de cargar el uri sirven
uri = 'spotify:artist:7ltDVBr6mKbRvohxheJ9h1'
uri = "3vQ0GE3mI0dAaxIMYe5g7z"
# En la siguiente línea se define de donde se saca toda la info, como si este fuera el paquete
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id = CLIENT_ID, client_secret=CLIENT_SECRET))





id = "spotify:artist:" + conseguir_id("Duki")

print(id)
def conseguir_cuadrivector(nombre_artista):

    lista_cuadrivectores = []
    uri = "spotify:artist:" + conseguir_id(nombre_artista)

    results = sp.artist_albums(uri, album_type='album,single', country='AR')

    albums = results['items']

    while results['next']:
        # Depaginate
        results = sp.next(results)
        albums.extend(results['items'])

    # Filter albums/singles to unique
    real_albums = dict()
    for album in albums:
        # Strip extraneous characters
        name = re.sub(r'\([^)]*\)|\[[^)]*\]', '', album['name']) # remove (Deluxe edition) and [Feat. asdf] tags
        name = re.sub(r'\W','', name).lower().strip() # remove all non-alphanumerical characters
        if name not in real_albums:
            print('Adding ' + name)
            real_albums[name] = album

    return lista_cuadrivectores

print(conseguir_cuadrivector("Rosalia"))