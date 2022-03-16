from matplotlib.image import imread
import numpy as np
from numpy import NaN 
import pandas as pd
from PIL import Image
import numpy
import math
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans


"""Conversion csv en json"""
df = pd.read_csv('pokemon.csv')
df.to_json('pokemon.json')

"""Ajout colonne couleur pour chaque pokemon"""

"""Couleur prédominant image"""

def CouleursClusters(numarray):
    clusters = KMeans(n_clusters = 3)
    clusters.fit(numarray)
    npbins = numpy.arange(0, 4)
    histogram = numpy.histogram(clusters.labels_, bins=npbins)
    
    histogram[0][np.where(histogram[0] == max(histogram[0]))] = 0
    indexC1 = np.where(histogram[0] == max(histogram[0]))
    histogram[0][np.where(histogram[0] == max(histogram[0]))] = 0
    indexC2 = np.where(histogram[0] == max(histogram[0]))
    return clusters, indexC1[0][0], indexC2[0][0]

def CouleursRGBA(clusters, indexC):
    couleur='#%02x%02x%02x' % (
    math.ceil(clusters.cluster_centers_[indexC][0]), 
        math.ceil(clusters.cluster_centers_[indexC][1]),
    math.ceil(clusters.cluster_centers_[indexC][2]))
    return couleur

def CouleursP(clusters, indexC):
    couleur='#%02x0000' % (
    math.ceil(clusters.cluster_centers_[indexC][0]))
    return couleur

def CouleurPokemon(imgName):
    try:
        imgfile = Image.open("images/"+imgName+".png")
    except FileNotFoundError:
        imgfile = Image.open("images/"+imgName+".jpg")

    if (imgfile.mode == 'RGBA' or imgfile.mode == 'RGB'):
        numarray = numpy.array(imgfile.getdata(), numpy.uint8)
        clusters, indexC1, indexC2 = CouleursClusters(numarray)
        couleur1=CouleursRGBA(clusters, indexC1)
        couleur2=CouleursRGBA(clusters, indexC2)
        
    if (imgfile.mode == 'P'):
        numarray = numpy.array(imgfile.getdata(), numpy.uint8).reshape(-1,1)
        clusters, indexC1, indexC2 = CouleursClusters(numarray)
        couleur1=CouleursP(clusters, indexC1)
        couleur2=CouleursP(clusters, indexC2)
    
    return couleur1, couleur2


"""Ajout couleur prédominantes aux infos sur les images"""

"""
dfPokemon = pd.read_json('pokemon.json')

dfPokemon.insert(3,"Couleur1", None)
dfPokemon.insert(4,"Couleur2", None)

for i in range(len(dfPokemon)): 
    couleur1, couleur2 = CouleurPokemon(dfPokemon.loc[i,"Name"])
    dfPokemon.loc[i,"Couleur1"] = couleur1
    dfPokemon.loc[i,"Couleur2"] = couleur2
    print(str(i) + " : " + dfPokemon.loc[i,"Couleur1"] + " | " + dfPokemon.loc[i,"Couleur2"])

dfPokemon.to_json('pokemon2.json', orient="records")
"""
dfPokemon = pd.read_json('pokemon2.json')

def checkPokemonColors(Name):
    img = imread("images/"+Name+".png")
    fig, axs = plot.subplots(2, 1)
    axs[0].imshow(img)
    axs[1].pie([1,1], colors=[dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0], dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur2"].iloc[0]])
    plot.show()

checkPokemonColors("aggron")

"""Generation d'utilisateurs"""

import random as rd

nb_users = 5
for i in range(nb_users):
    data = {}
    df = pd.DataFrame(data)
    for j in range(rd.randint(10,50)):
        df = df.append(dfPokemon.loc[rd.randint(0,len(dfPokemon)-1)])
    df.to_json('user'+str(i)+'.json', orient="records")

"""Traitement infos"""