import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import graphviz
import pydotplus
from IPython.display import Image, display

"""Etiquetage automatique des données en fonction de leurs couleurs prédominantes"""

"""Couleurs prédominantes"""
"""
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
        #Permet de convertir les images en mode P en RGB
        
        rgb_im = imgfile.convert("RGB")
        rgb_im.save("images_test/"+imgName+".jpg")
        imgfile = Image.open("images_test/"+imgName+".jpg")
        numarray = np.array(imgfile.getdata(), np.uint8)
        clusters, indexC1, indexC2 = CouleursClusters(numarray)
        couleur1=CouleursRGBA(clusters, indexC1)
        couleur2=CouleursRGBA(clusters, indexC2)
    
    return couleur1, couleur2

#Stockage des étiquettes couleur1 et couleur2 dans pokemon.json


dfPokemon = pd.read_json('pokemon.json')

dfPokemon.insert(3,"Couleur1", None)
dfPokemon.insert(4,"Couleur2", None)

for i in range(len(dfPokemon)): 
    couleur1, couleur2 = CouleurPokemon(dfPokemon.loc[i,"Name"])
    dfPokemon.loc[i,"Couleur1"] = couleur1
    dfPokemon.loc[i,"Couleur2"] = couleur2
    print(str(i) + " : " + dfPokemon.loc[i,"Couleur1"] + " | " + dfPokemon.loc[i,"Couleur2"])

dfPokemon.to_json('pokemon.json', orient="records")

#Génération des utilisateurs


import random as rd

nb_users = 5
for i in range(nb_users):
    data = {}
    df = pd.DataFrame(data)
    for j in range(rd.randint(5,20)):
        df = df.append(dfPokemon.loc[rd.randint(0,len(dfPokemon)-1)])
    df.to_json('user'+str(i)+'.json', orient="records")

#Classification

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
"""

def Type2CaseNull(i):
    if(dfPokemon.iloc[i]['Type2'] != None): return dfPokemon.iloc[i]['Type2']
    else: return 'None'

dfPokemon = pd.read_json('pokemon3.json')
dfFavorite = pd.read_json('user0.json')
data = [[dfPokemon.iloc[i]['Type1'], Type2CaseNull(i), dfPokemon.iloc[i]['Couleur1'], dfPokemon.iloc[i]['Couleur2']] for i in range(len(dfPokemon))]

result = ['NotFavorite' for _ in range(len(dfPokemon))]
for pokemonFavori in dfFavorite.iloc[:]['Name']:
    result[dfPokemon[dfPokemon['Name'] == pokemonFavori].index.values[0]] = 'Favorite'

#creating dataframes
dataframe = pd.DataFrame(data, columns=['Type1', 'Type2', 'Couleur1', 'Couleur2'])
resultframe = pd.DataFrame(result, columns=['Favorite'])

#generating numerical labels
le1 = LabelEncoder()
dataframe['Type1'] = le1.fit_transform(dataframe['Type1'])

le2 = LabelEncoder()
dataframe['Type2'] = le2.fit_transform(dataframe['Type2'])

le3 = LabelEncoder()
dataframe['Couleur1'] = le3.fit_transform(dataframe['Couleur1'])

le4 = LabelEncoder()
dataframe['Couleur2'] = le4.fit_transform(dataframe['Couleur2'])

le5 = LabelEncoder()
resultframe['Favorite'] = le5.fit_transform(resultframe['Favorite'])

#Use of decision tree classifiers
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(dataframe, resultframe)

dot_data = tree.export_graphviz(dtc, out_file=None,
        feature_names=dataframe.columns,
        filled=True, rounded=True, 
        class_names =
        le5.inverse_transform(
        resultframe.Favorite.unique())
        )
graph = graphviz.Source(dot_data)

pydot_graph = pydotplus.graph_from_dot_data(dot_data)
img = Image(pydot_graph.create_png())
display(img)