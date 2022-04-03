import pandas as pd

"""Récupération automatique du csv des pokemons"""

"""Conversion csv en json"""
df = pd.read_csv('pokemon.csv')
df.to_json('pokemon.json')