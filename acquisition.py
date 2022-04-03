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
df = pd.read_csv('https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types?select=pokemon.csv')
print(df)
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


"""LECTURE BASE DE DONNEE + AJOUT COULEUR DANS LA BASE DE DONNEE DE BASE"""

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
"""
import random as rd

nb_users = 5
for i in range(nb_users):
    data = {}
    df = pd.DataFrame(data)
    for j in range(rd.randint(5,20)):
        df = df.append(dfPokemon.loc[rd.randint(0,len(dfPokemon)-1)])
    df.to_json('user'+str(i)+'.json', orient="records")
"""
"""Traitement infos"""


"""CHANGEMENT FORMAT COULEUR IMAGE MODE P EN MODE RGB"""

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
    couleur2 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur2"].iloc[0]
    return couleur1 , couleur2


"""Récupération couleur pokemon qui corresponde aux noms qu'il y a dans le pokemon3.json"""
def Recup_couleur_pokemon(Name):
    couleur1 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur1"].iloc[0]
    couleur2 = dfPokemon.loc[dfPokemon["Name"] == Name,"Couleur2"].iloc[0]
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
    Pokemon_couleur_compatible=[]
    compteur = 0
    Name = lecture_json_utilisateur()
    for k in Name:
        couleur1hex_user,couleur2hex_user = Recup_couleur_pokemon_user(k)
        couleur1_user,couleur2_user = Conversion_hex_RGB(couleur1hex_user,couleur2hex_user)
        Name_pok=lecture_BDD()
        for j in Name_pok:
            couleur1hex,couleur2hex = Recup_couleur_pokemon(j)
            couleur1,couleur2 = Conversion_hex_RGB(couleur1hex,couleur2hex)
            seuil = 40
            diff = np.sqrt((couleur1[0]-couleur1_user[0])**2+(couleur1[1]-couleur1_user[1])**2+(couleur1[2]-couleur1_user[2])**2)
            if (diff < seuil) and (j not in Pokemon_couleur_compatible):
                seuil = diff
                Pokemon_couleur_compatible.append(j)
                compteur+=1
    return(Pokemon_couleur_compatible)


"""FIN ALGO RECO COULEUR"""





"""ESSAIE ALGO DE RECO AVEC TYPE POKEMON"""

"""COMPTEUR DE NOMBRE DE TYPE IDENTIQUE POUR UN USER"""


"""Récupération couleur pokemon qui corresponde aux noms qu'il y a dans le user.json"""
def Recup_type_pokemon_user(Name):
    type1 = dfPokemon.loc[dfPokemon["Name"] == Name,"Type1"].iloc[0]
    type2 = dfPokemon.loc[dfPokemon["Name"] == Name,"Type2"].iloc[0]
    return type1 , type2


"""Récupération couleur pokemon qui corresponde aux noms qu'il y a dans le pokemon3.json"""
def recup_type_BDD(Name):
    type1 = dfPokemon.loc[dfPokemon["Name"] == Name,"Type1"].iloc[0]
    type2 = dfPokemon.loc[dfPokemon["Name"] == Name,"Type2"].iloc[0]
    return type1 , type2

def compteur_type_identique():
    Pokemon_type_compatible=[]
    compteur = 0
    Name = lecture_json_utilisateur()
    for k in Name:
        type1_user,type2_user = Recup_type_pokemon_user(k)
        Name_pok=lecture_BDD()
        for j in Name_pok:
            type1,type2 = recup_type_BDD(j)
            if (type1_user == type2) and (j not in Pokemon_type_compatible):
                Pokemon_type_compatible.append(j)
                compteur+=1
    return(Pokemon_type_compatible)



"""Comparaison Couleur algo et type algo"""

def Pokemon_reco_final():
    Pokemon_couleur_compatible = couleur_aime_user()
    Pokemon_type_compatible = compteur_type_identique()
    pokemon_final = []
    for k in Pokemon_couleur_compatible:
        for j in Pokemon_type_compatible:
            if k == j:
                pokemon_final.append(j)
    print(pokemon_final)

#Pokemon_reco_final()


"""Fonction pas aboutie c'était un test"""

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
