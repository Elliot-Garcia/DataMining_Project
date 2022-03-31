from tokenize import Name
from cv2 import ROTATE_180
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
        rgb_im.save("images_test/"+imgName+".jpg")
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
    



"""DEBUT ALGO RECO COULEUR POKEMON"""

"""Lecture like utilisateur"""
def lecture_json_utilisateur():
    dfUser = pd.read_json('user0.json')
    Name = dfUser.iloc[:]['Name']
    return Name

"""Lecture base de donnée de pokemon on récupère les noms des pokemons"""
def lecture_BDD():
    dfPokemon = pd.read_json('pokemon3.json')
    Name = dfPokemon.iloc[:]['Name']
    return Name

"""Récupération couleur pokemon qui corresponde aux noms qu'il y a dans le user.json"""
def Recup_couleur_pokemon_user(Name):
    couleur1 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0]
    couleur2 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0]
    return couleur1 , couleur2


"""Récupération couleur pokemon qui corresponde aux noms qu'il y a dans le pokemon3.json"""
def Recup_couleur_pokemon(Name):
    couleur1 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0]
    couleur2 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0]
    return couleur1 , couleur2

"""Conversion couleur hex en RGB pour la comparaison"""
def Conversion_hex_RGB(couleur1,couleur2):
    couleur1_RGB=[]
    couleur2_RGB=[]
    rouge1 = int(couleur1[1:3],16)
    couleur1_RGB.append(rouge1)
    vert1 = int(couleur1[3:5],16)
    couleur1_RGB.append(vert1)
    bleu1 = int(couleur1[5:7],16)
    couleur1_RGB.append(bleu1)
    rouge2 = int(couleur2[1:3],16)
    couleur2_RGB.append(rouge2)
    vert2 = int(couleur2[3:5],16)
    couleur2_RGB.append(vert2)
    bleu2 = int(couleur2[5:7],16)
    couleur2_RGB.append(bleu2)
    return(couleur1_RGB,couleur2_RGB)


"""Algo qui compare les couleurs et qui ajoute dans une liste le nom des pokemons qui correspondent aux critères"""
def couleur_aime_user():
    Pokemon_compatible=[]
    compteur = 0
    Name = lecture_json_utilisateur()
    for k in Name:
        couleur1hex_user,couleur2hex_user = Recup_couleur_pokemon_user(k)
        couleur1_user,couleur2_user = Conversion_hex_RGB(couleur1hex_user,couleur2hex_user)
        Name_pok=lecture_BDD()
        for j in Name_pok:
            couleur1hex,couleur2hex = Recup_couleur_pokemon(j)
            couleur1,couleur2 = Conversion_hex_RGB(couleur1hex,couleur2hex)
            seuil = 60
            diff = np.sqrt((couleur1[0]-couleur1_user[0])**2+(couleur1[1]-couleur1_user[1])**2+(couleur1[2]-couleur1_user[2])**2)
            if (diff < seuil) and (j not in Pokemon_compatible):
                Pokemon_compatible.append(j)
                compteur+=1
    print(Pokemon_compatible)
    print(compteur)

couleur_aime_user()

"""FIN ALGO RECO COULEUR"""




"""ESSAIE ALGO DE RECO AVEC TYPE POKEMON"""

"""COMPTEUR DE NOMBRE DE TYPE IDENTIQUE POUR UN USER"""

def compteur_type_identique():
    """extraction du type1 pour l'utilisateur 0"""
    dfUser = pd.read_json('user0.json')
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
            matrice_compteur = np.append(matrice_compteur,[ajout],axis=0)
    return matrice_compteur 


#print(dfPokemon.loc[dfPokemon["Type1"] == "Grass","Name"])

"""Affiche les types de pokemon que l'utilisateur a liké"""
def choix_recommendation_type(matrice_compteur):
    for k in range (0,len(matrice_compteur)):
        for type in dfPokemon.iloc[:]['Type1']:
            if type == matrice_compteur[k][0]:
                name=dfPokemon.loc[dfPokemon["Type1"] == type,"Name"]
                for Name in name:
                    print(Name)
                    try:
                        imgfile = Image.open("images/"+Name+".png")
                    except FileNotFoundError:
                        imgfile = Image.open("images/"+Name+".jpg")
                    fig, axs = plot.subplots(2, 1)
                    axs[0].imshow(imgfile)
                    axs[1].pie([1,1], colors=[dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0], dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur2"].iloc[0]])
                    plot.show()

#matrice_compteur = compteur_type_identique()
#choix_recommendation_type(matrice_compteur)