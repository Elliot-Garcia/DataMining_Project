import numpy as np 
import pandas as pd
from PIL import Image
import numpy
import math
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans


"""Conversion csv en json"""
df = pd.read_csv('pokemon.csv')
df.to_json('pokemon.json', orient="records")
df = pd.read_json('pokemon.json')
"""Ajout colonne couleur pour chaque pokemon"""

"""Couleur prédominant image"""

imgfile = Image.open("images/aggron.png")
print(imgfile.mode)
if (imgfile.mode == 'RGBA'):
    numarray = numpy.array(imgfile.getdata(), numpy.uint8)
    clusters = KMeans(n_clusters = 4)
    clusters.fit(numarray)
    npbins = numpy.arange(0, 5)
    histogram = numpy.histogram(clusters.labels_, bins=npbins)
    labels = numpy.unique(clusters.labels_)
    barlist = plot.bar(labels, histogram[0])
    for i in range(4):
        barlist[i].set_color('#%02x%02x%02x' % (
        math.ceil(clusters.cluster_centers_[i][0]), 
            math.ceil(clusters.cluster_centers_[i][1]),
        math.ceil(clusters.cluster_centers_[i][2])))
    plot.show()

if (imgfile.mode == 'P'):
    numarray = numpy.array(imgfile.getdata(), numpy.uint8).reshape(-1,1)
    clusters = KMeans(n_clusters = 4)
    clusters.fit(numarray)
    npbins = numpy.arange(0, 5)
    histogram = numpy.histogram(clusters.labels_, bins=npbins)
    labels = numpy.unique(clusters.labels_)
    barlist = plot.bar(labels, histogram[0])
    print(labels)
    print(histogram[0])
    for i in range(4):
        barlist[i].set_color('#%02x0000' % (
        math.ceil(clusters.cluster_centers_[i][0])))
        print(clusters.cluster_centers_[i][0])
    plot.show()