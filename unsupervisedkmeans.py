#!/usr/bin/env python
# coding: utf-8

# In[141]:


#unsupervised k-mean

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[143]:


kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)


# In[147]:


kmeans.fit(X_train)


# In[149]:


# The lowest SSE value
kmeans.inertia_









# In[150]:


# Final locations of the centroid
kmeans.cluster_centers_


# In[151]:


# The number of iterations required to converge
kmeans.n_iter_


# In[152]:


kmeans.labels_[:5]


# In[155]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_train)
    sse.append(kmeans.inertia_)


# In[158]:


# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_train)
    score = silhouette_score(X_train, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[160]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[161]:


kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

kl.elbow


# In[162]:


from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score


# In[176]:


# Instantiate k-means and dbscan algorithms
km1 = KMeans(n_clusters=2)
dbscan1 = DBSCAN(eps=0.3)

# Fit the algorithms to the features
kmeans.fit(X_train)
dbscan.fit(X_train)

# Compute the silhouette scores for each algorithm
kmeans_silhouette = silhouette_score(
    X_train, kmeans.labels_
).round(2)
dbscan_silhouette = silhouette_score(
   X_train, dbscan.labels_
).round (2)


# In[166]:


kmeans_silhouette



# In[167]:


dbscan_silhouette


# In[179]:


import matplotlib.pyplot as plt
 
#filter rows of original data
filtered_label0 = X_train[label == 0]
 
#plotting the results
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()


# In[171]:


#filter rows of original data
filtered_label2 = X_train[label == 2]
 
filtered_label8 = X_train[label == 8]
 
#Plotting the results
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
plt.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'black')
plt.show()

