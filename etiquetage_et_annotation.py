import math
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import pandas as pd

"""Etiquetage automatique des données en fonction de leurs couleurs prédominantes"""

"""Couleurs prédominantes"""

def CouleursClusters(numarray):
    clusters = KMeans(n_clusters = 3)
    clusters.fit(numarray)
    npbins = np.arange(0, 4)
    histogram = np.histogram(clusters.labels_, bins=npbins)
    
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
        rgb_im.save("images_test/"+imgName+".jpg")
        numarray = np.array(imgfile.getdata(), np.uint8)
        clusters, indexC1, indexC2 = CouleursClusters(numarray)
        couleur1=CouleursRGBA(clusters, indexC1)
        couleur2=CouleursRGBA(clusters, indexC2)
        
    if (imgfile.mode == 'P'):
        """Permet de convertir les images en mode P en RGB"""
        rgb_im = imgfile.convert("RGB")
        rgb_im.save("images_test/"+imgName+".jpg")
        imgfile = Image.open("images_test/"+imgName+".jpg")
        numarray = np.array(imgfile.getdata(), np.uint8)
        clusters, indexC1, indexC2 = CouleursClusters(numarray)
        couleur1=CouleursRGBA(clusters, indexC1)
        couleur2=CouleursRGBA(clusters, indexC2)
    
    return couleur1, couleur2

"""Stockage des étiquettes couleur1 et couleur2 dans pokemon.json"""

dfPokemon = pd.read_json('pokemon.json')

dfPokemon.insert(3,"Couleur1", None)
dfPokemon.insert(4,"Couleur2", None)

for i in range(len(dfPokemon)): 
    couleur1, couleur2 = CouleurPokemon(dfPokemon.loc[i,"Name"])
    dfPokemon.loc[i,"Couleur1"] = couleur1
    dfPokemon.loc[i,"Couleur2"] = couleur2
    print(str(i) + " : " + dfPokemon.loc[i,"Couleur1"] + " | " + dfPokemon.loc[i,"Couleur2"])

dfPokemon.to_json('pokemon.json', orient="records")

"""Génération des utilisateurs"""

import random as rd

nb_users = 5
for i in range(nb_users):
    data = {}
    df = pd.DataFrame(data)
    for j in range(rd.randint(5,20)):
        df = df.append(dfPokemon.loc[rd.randint(0,len(dfPokemon)-1)])
    df.to_json('user'+str(i)+'.json', orient="records")

"""Classification"""
