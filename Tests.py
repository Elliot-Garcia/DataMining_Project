import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import graphviz
import pydotplus
from IPython.display import Image, display
import random as rd

def TestUser(df):
    dfFav = pd.DataFrame()
    while True:
        rdPokemon = rd.randint(0,len(df)-1)
        Pokemon = df.loc[rdPokemon]
        df.drop(rdPokemon)
        print("Nom : " + Pokemon['Name'] + " | Type 1 : " + Pokemon['Type1'] + " | Type 2 : " + str(Pokemon['Type2']))
        try:
            display(Image("images/"+Pokemon['Name']+".png"))
        except FileNotFoundError:
            display(Image("images/"+Pokemon['Name']+".jpg"))
            
        res = input("Aimez-vous ce pokemon [o/n/stop] : ")

        if(res == 'o'):
            dfFav = dfFav.append(Pokemon)
            dfFav.loc[rdPokemon]['Favorite'] = 'Favorite'
        elif(res == 'n'):
            dfFav = dfFav.append(Pokemon)
            dfFav.loc[rdPokemon]['Favorite'] = 'NotFavorite'
        else:
            nom = input("Entrez votre nom : ")
            dfFav.to_json(nom+'.json', orient="records")
            return

dfPokemonFav = pd.read_json('pokemon3.json')
dfPokemonFav.insert(5,"Favorite", None)

TestUser(dfPokemonFav)