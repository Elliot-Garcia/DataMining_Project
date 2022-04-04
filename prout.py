import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import graphviz
import pydotplus
from IPython.display import Image, display
import random
import time

def Type2CaseNull(i):
    if(dfFavorite .iloc[i]['Type2'] != None): return dfFavorite .iloc[i]['Type2']
    else: return 'None'

dfFavorite = pd.read_json('user0.json')
data = [[dfFavorite .iloc[i]['Type1'], Type2CaseNull(i), dfFavorite .iloc[i]['Couleur1'], dfFavorite .iloc[i]['Couleur2']] for i in range(len(dfFavorite ))]

result = [ dfFavorite['Favorite'].loc[dfFavorite['Name'] == pokemonFavori].values[0] for pokemonFavori in dfFavorite.iloc[:]['Name'] ]
print(result)
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

recommandation = [] #on initialise une liste de tableaux recommandés
non_recommandation=[] #on initialise une liste de tableaux non-recommandés

dfPokemon = pd.read_json('pokemon3.json')


while len(recommandation)<10:
    Pokemon = dfPokemon.iloc[random.randint(0,len(dfPokemon)-1)]
    """Compteur qui vont permettre de détecter un élément nulle avant que le programme plante"""
    compteur_type1 = 0
    compteur_type2 = 0
    compteur_couleur2 = 0
    compteur_couleur1 = 0

    """L'algorithme de prédiction plante si on lui fait predict une valeur qui n'est pas dans les user.json"""
    for j in dfFavorite["Type1"]:
        if j == Pokemon["Type1"]:
            compteur_type1 = 1
            
    for k in dfFavorite["Type2"]:
        if k == Pokemon["Type2"]:
            compteur_type2 = 1
        if Pokemon["Type2"] == None:
            compteur_type2 = 0
    for i in dfFavorite["Couleur1"]:
        if i == Pokemon["Couleur1"]:
            compteur_couleur1 = 1
    for p in dfFavorite["Couleur2"]:
        if p == Pokemon["Couleur2"]:
            compteur_couleur2 = 1
    if compteur_type1==0:
        continue
    if compteur_type2 == 0 :
        continue
    if compteur_couleur2 == 0 :
        continue
    if compteur_couleur1 == 0 :
        continue

    """Si toutes les conditions sont vérifiés l'algo de predict peut fonctionner normalement"""
    if (compteur_type2 == 1) and (compteur_type1 == 1) and (compteur_couleur1==1) and (compteur_couleur2==1) :
        prediction = dtc.predict([ #on réalise une prédiction pour savoir si le tableau plaira à l'utilisateur
                [le1.transform([Pokemon["Type1"]])[0], le2.transform([Pokemon["Type2"]])[0],le3.transform([Pokemon["Couleur1"]])[0],le4.transform([Pokemon["Couleur2"]])[0]]])


    
    if prediction == [1]: 
        print('salut')
        recommandation.append(Pokemon)
        
    else:
       non_recommandation.append(Pokemon)
print(recommandation)
print('POOOOOOOOOOOOOOOOOOOOOOOOOO')
print(non_recommandation)