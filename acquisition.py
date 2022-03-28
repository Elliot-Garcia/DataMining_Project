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
        """Permet de convertir les images en mode P en RGB"""
        rgb_im = imgfile.convert("RGB")
        rgb_im.save("images_test/"+imgName+".jpg")
        imgfile = Image.open("images_test/"+imgName+".jpg")
        numarray = numpy.array(imgfile.getdata(), numpy.uint8)
        clusters, indexC1, indexC2 = CouleursClusters(numarray)
        couleur1=CouleursRGBA(clusters, indexC1)
        couleur2=CouleursRGBA(clusters, indexC2)
    
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

dfPokemon.to_json('pokemon3.json', orient="records")
"""
dfPokemon = pd.read_json('pokemon3.json')

def checkPokemonColors(Name):
    img = Image.open("images/"+Name+".png")
    fig, axs = plot.subplots(2, 1)
    axs[0].imshow(img)
    axs[1].pie([1,1], colors=[dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0], dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur2"].iloc[0]])
    plot.show()

#checkPokemonColors("alomomola")

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


"""Chnangement format couleur P en mode RGB"""
def changement_format_couleur():
    imgfile = Image.open("images/"+"aggron"+".png")
    if imgfile.mode == "JPEG":
        imgfile.save("xxx.jpg")
    # in most case, resulting jpg file is resized small one
    elif imgfile.mode in ["RGBA", "P"]:
        rgb_im = imgfile.convert("RGB")
        rgb_im.save("images_test/aggron.jpg")
        
    #df.to_json('nouveau.json', orient="records")
    

"""extraction des couleurs liké du user0"""
def couleur_préféré_utilisateur():
    dfUser = pd.read_json('user0.json')
    print(dfUser)
    a = dfUser.iloc[:]['Couleur1']
    print(a)
    return a

"""Conversion valeur hexa en RGB"""
def calcul_valeur_couleur(a):
    Liste_RGB_pref=np.zeros((len(a),4),dtype=float)
    print(int(a[0][1],16))
    for k in range (0,len(a)):
        id = k
        rouge = int(a[k][1:3],16)
        vert = int(a[k][3:5],16)
        bleu = int(a[k][5:7],16)
        Liste_RGB_pref[k][0]=id
        Liste_RGB_pref[k][1]=rouge
        Liste_RGB_pref[k][2]=vert
        Liste_RGB_pref[k][3]=bleu
    print(Liste_RGB_pref)
    return(Liste_RGB_pref)

"""faut trouver un algo pour trouver les couleurs qui se ressemble"""
def choix_couleur_proche(Liste_RGB_pref):
    seuil = 30
    for k in range (0,len(Liste_RGB_pref)):
        return 0

#a = couleur_préféré_utilisateur()
#calcul_valeur_couleur(a)
#changement_format_couleur()

"""ESSAIE ALGO DE RECO AVEC TYPE POKEMON"""

"""COMPTEUR DE NOMBRE DE TYPE IDENTIQUE POUR UN USER"""

def compteur_type_identique():
    """extraction du type1 pour l'utilisateur 0"""
    dfUser = pd.read_json('user0.json')
    print(dfUser)
    Liste_type = dfUser.iloc[:]['Type1']
    """Compteur de type identique dans la liste"""
    matrice_compteur = np.zeros((1,2))
    for j in range (len(Liste_type)):
        b=0
        for k in range (len(matrice_compteur)):
            if matrice_compteur[k][0] == Liste_type[j]:
                b=1

        if b==0:
            compteur = str(Liste_type).count(Liste_type[j])
            ajout = np.array([Liste_type[j],compteur])
            print(compteur)
            matrice_compteur = np.append(matrice_compteur,[ajout],axis=0)
            
    print(matrice_compteur)


def choix_recommendation_type():
    if dfPokemon.iloc['Type1']:
        return 0

compteur_type_identique()

