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

from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import operator
import random


#Fonction de comparaison de couleurs
#Entrées : deux codes RGB de deux couleurs
#Sorties : un indice de proximité des couleurs
def ComparaisonCouleur(couleur1, couleur2):
    dif = []
    for i in range (3):
        dif.append(couleur1[i] - couleur2[i])
    #mise au carré pour être sûr d'avoir des valeurs de diférence positive 
    return dif[0]**2+dif[1]**2+dif[2]**2

#ouverture des fichiers précédemment créés
with open("label.json",'r') as jsonTab:
    dataTab = json.load(jsonTab)

with open("user.json",'r') as jsonUser:
    dataUser = json.load(jsonUser)


data=[]
result=[]
auteur_vu=[]
dataTabCustom = dataTab #on crée une copie de notre liste de tableaux, qui représentera les tableaux "non vus" par l'utilisateur
for lien in dataUser[0]["likes"]: #on construit un dataframe qui va servir à l'algo de classification : on commence par les tableaux likés par l'utilisateur (result = 1)
    for tableau in dataTab:
        if lien == tableau["lien"]:
            tag = max(tableau["tags"].items(), key=operator.itemgetter(1))[0] #l'algo ne prend pas de liste de tags en paramètre: on choisit donc le tag le plus utilisé sur le tableau
            data.append([tableau["auteur"],tableau["format"],tag])
            result.append(1)
            auteur_vu.append(tableau["auteur"])
            dataTabCustom.remove(tableau)
            

for lien in dataUser[0]["unlikes"]:#on fait la même chose avec les tableaux vu et non likés (result = -1)
    for tableau in dataTab:
        if lien == tableau["lien"]:
            tag = max(tableau["tags"].items(), key=operator.itemgetter(1))[0]
            data.append([tableau["auteur"],tableau["format"],tag])
            result.append(-1)
            auteur_vu.append(tableau["auteur"])
            dataTabCustom.remove(tableau)



dataframe = pd.DataFrame(data, columns=['auteur', 'format', 'tags'])#construction des dataframes
resultframe = pd.DataFrame(result, columns=['like'])

le1 = LabelEncoder() #encodage des labels
dataframe['auteur'] = le1.fit_transform(dataframe['auteur'])

le2 = LabelEncoder()
dataframe['format'] = le2.fit_transform(dataframe['format'])

le3 = LabelEncoder()
dataframe['tags'] = le3.fit_transform(dataframe['tags'])

le4 = LabelEncoder()
resultframe['like'] = le4.fit_transform(resultframe['like'])

#On utilise les classifieurs "arbres"
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(dataframe, resultframe)

recommandation = [] #on initialise une liste de tableaux recommandés
non_recommandation=[] #on initialise une liste de tableaux non-recommandés
while len(recommandation)<10:
    tab = random.choice(dataTabCustom)
    if tab["auteur"] not in auteur_vu: #l'algo de prédiction plante si on lui présente un tableau dont l'auteur est n'a pas été vu par l'utilisateur.
        continue
    prediction = dtc.predict([ #on réalise une prédiction pour savoir si le tableau plaira à l'utilisateur
            [le1.transform([tab["auteur"]])[0], le2.transform([tab["format"]])[0],
            le3.transform([max(tableau["tags"].items(), key=operator.itemgetter(1))[0]])[0]]])
    if prediction == 1: 
        recommandation.append(tab)
    else:
        non_recommandation.append(tab)

for tab in recommandation: #on ajoute une comaraison par couleurs des tableaux, afin d'avoir la recommandation la plus précise possible
    tab["indice_couleur"]=ComparaisonCouleur(dataUser[0]["couleurPref"],tab["couleur"])

dataframe=pd.DataFrame(recommandation)
dataframe.drop(columns=["auteur",'largeur','hauteur','format','tags','unlikes','couleur'],inplace=True)
dataframe.sort_values(by="indice_couleur",inplace=True)
if dataframe.iloc[0]['indice_couleur']>200 :
    dataframe.sort_values(by="likes",inplace=True)
print(dataframe.iloc[0]['lien'])