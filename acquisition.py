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
    clusters = KMeans(n_clusters = 4)
    clusters.fit(numarray)
    npbins = numpy.arange(0, 5)
    histogram = numpy.histogram(clusters.labels_, bins=npbins)
    labels = numpy.unique(clusters.labels_)
    barlist = plot.bar(labels, histogram[0])
    return clusters, barlist

def CouleursRGBA(clusters, barlist):
    for i in range(3):
        couleur1='#%02x%02x%02x' % (
        math.ceil(clusters.cluster_centers_[i][0]), 
            math.ceil(clusters.cluster_centers_[i][1]),
        math.ceil(clusters.cluster_centers_[i][2]))
    #    barlist[i].set_color('#%02x%02x%02x' % (
    #    math.ceil(clusters.cluster_centers_[i][0]), 
    #        math.ceil(clusters.cluster_centers_[i][1]),
    #    math.ceil(clusters.cluster_centers_[i][2])))
    #plot.show()
    return couleur1

def CouleursP(clusters, barlist):
    for i in range(3):
        couleur1='#%02x0000' % (
        math.ceil(clusters.cluster_centers_[i][0]))
    #    barlist[i].set_color('#%02x0000' % (
    #    math.ceil(clusters.cluster_centers_[i][0])))
    #plot.show()
    return couleur1

def CouleurPokemon(imgName):
    try:
        imgfile = Image.open("images/"+imgName+".png")
    except FileNotFoundError:
        imgfile = Image.open("images/"+imgName+".jpg")

    if (imgfile.mode == 'RGBA' or imgfile.mode == 'RGB'):
        numarray = numpy.array(imgfile.getdata(), numpy.uint8)
        clusters, barlist = CouleursClusters(numarray)
        couleur1=CouleursRGBA(clusters, barlist)
        
    if (imgfile.mode == 'P'):
        numarray = numpy.array(imgfile.getdata(), numpy.uint8).reshape(-1,1)
        clusters, barlist = CouleursClusters(numarray)
        couleur1=CouleursP(clusters, barlist)
    
    return couleur1


"""Ajout couleur prédominantes aux infos sur les images"""

dfPokemon = pd.read_json('pokemon.json')

dfPokemon.insert(3,"Couleur1", None)
dfPokemon.insert(4,"Couleur2", None)

for i in range(len(dfPokemon)): 
    dfPokemon.loc[i,"Couleur1"] = CouleurPokemon(dfPokemon.loc[i,"Name"])
    print("" + str(i) + " : " + dfPokemon.loc[i,"Couleur1"] + "")
