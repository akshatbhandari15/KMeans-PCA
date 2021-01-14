# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:34:47 2021

@author: anant
"""

import pandas as pd
import numpy as np
import random

from matplotlib import pyplot as plt






def datasetgetNumFeatures(dataset):
    return dataset.shape[1]

def getRandomCentroids(numFeatures, k, dataset):
    centroids = np.empty([k, numFeatures])
    for i in range(k):
        centroids[i, :] = dataset[random.randint(0,len(dataset)), :]
    return centroids
        
def kmeans(dataset, k):
    numFeatures = datasetgetNumFeatures(dataset)
    centroids = getRandomCentroids(numFeatures, k, dataset)
    
    iterations = 0
    oldCentroids = []
    
    while not shouldStop(oldCentroids, centroids, iterations):
        oldCentroids.append(centroids)
        iterations += 1
        
        labels = getLabels(dataset, centroids, k)
        centroids = getCentroids(dataset, labels, k)
        
    return centroids


def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAXITERATIONS:
        return True
    else:
        return False

def getLabels(dataset, centroids, k):
    labels = []
    for x in dataset:
        z = []
        for y in centroids:
            z.append(sum((x-y)**2))
            
        labels.append(z.index(min(z)))
    return np.array(labels)
    
def getCentroids (dataset, labels, k):
    centroids = np.zeros([k, 8])
    for i in range(k):
       centroids[i] = np.mean(dataset[labels == i], axis=0)
    
    return centroids
            
dataset = pd.read_csv("Country-data.csv")

dataset = dataset.drop("country", axis = 1)

dataset = dataset.values[:, 0:8]
dataset = (dataset - dataset.mean())/dataset.std()
MAXITERATIONS = 400
k = 3




#we will implement PCA here before running K-Means
sigma = np.cov(dataset.T)
eigenvalues, eigenvectors = np.linalg.eig(sigma)
#print (eigenvectors[:,0])

explained_variances = []
for i in range(len(eigenvalues)):
    explained_variances.append(eigenvalues[i] / np.sum(eigenvalues))
 
print(np.sum(explained_variances))
print(explained_variances)
#Here we can see the first feature accounts for about 99.9999% of variance in the dataset
Ureduce = eigenvectors[:,0:1]
z = dataset @ Ureduce

centroids = kmeans(dataset, k)
print(centroids)





colors=['orange', 'blue', 'green']
for i in range(166):
    plt.scatter(z, z, s=7)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='g', s=150)
