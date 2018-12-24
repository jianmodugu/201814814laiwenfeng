# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.mixture import GaussianMixture

nowpath = os.getcwd()
path = os.path.join(nowpath,r'Tweets.txt')

def loaddata():
     f = open(path)
     tweets  = f.readlines()
     alltext = []
     lables_true = []
     for str in tweets:
         str = str.strip('\n')
         setting = json.loads(str)
         alltext.append(setting['text'])
         lables_true.append(setting['cluster'])
     vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
     transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
     X = tfidf=transformer.fit_transform(vectorizer.fit_transform(alltext))

     return X,lables_true
 
  

def my_kmeans():
    X,cluster = loaddata()
    kmeans = KMeans(n_clusters = len(set(cluster)),random_state=0).fit(X)    
    print('kmeans result:')
    print('NMI score:%f\n' % normalized_mutual_info_score(cluster, kmeans.labels_))
    
def my_AffinityPropagation():
    X,labels_true = loaddata()
    
    # Compute Affinity Propagation
    af = AffinityPropagation().fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)
    
    print('AffinityPropagation result:')
    print('Estimated number of clusters: %d' % n_clusters_)
    print('NMI score:%f\n' % normalized_mutual_info_score(labels_true,labels))
    

def my_mean_shift():
    X,labels_true = loaddata()
    
    X = X.toarray()

#    bandwidth = estimate_bandwidth(X,quantile=0.2,n_samples=500)
#    ms = MeanShift(bandwidth = bandwidth,bin_seeding=False)
#    ms.fit(X)
    
    ms = MeanShift(bin_seeding=True).fit(X)
    labels = ms.labels_

#    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print('mean_shift result:')
    print('Estimated number of clusters: %d' % n_clusters_)
    print('NMI score:%f\n' % normalized_mutual_info_score(labels_true,labels))
    
def my_Spectral_cluster():
    X,labels_true = loaddata()
    spectral = SpectralClustering(n_clusters=len(set(labels_true)), random_state=0,affinity='cosine').fit(X)
    score_spectral = normalized_mutual_info_score(spectral.labels_, labels_true)
    
    print('Spectral_cluster result:')
    print("NMI score = %f" %(score_spectral))

def my_AgglomerativeClustering():
    X,labels_true = loaddata()
    
    clustering = AgglomerativeClustering(n_clusters = len(set(labels_true))).fit(X.toarray())
    
    labels = clustering.labels_
    print('AgglomerativeClustering result:')
    print('NMI score:%f\n' % normalized_mutual_info_score(labels_true,labels))
    
def my_DBSCAN():
    X,labels_true = loaddata()
    
    clustering = clustering = DBSCAN(min_samples=2,metric='cosine').fit(X)
    
    labels = clustering.labels_
    print('DBSCAN result:')
    print('NMI score:%f\n' % normalized_mutual_info_score(labels_true,labels))
    
def my_GaussianMixture():
    X,labels_true = loaddata()
    clustering =GaussianMixture(n_components=len(set(labels_true)), covariance_type='tied',random_state=0,max_iter=200).fit(X.toarray())
    labels = clustering.predict(X.toarray())
    print('GaussianMixture result:')
    print('NMI score:%f\n' % normalized_mutual_info_score(labels_true,labels))

print(datetime.datetime.now()) 
my_kmeans()
#my_AffinityPropagation()
#my_mean_shift()
#my_Spectral_cluster()
#my_AgglomerativeClustering()
#my_DBSCAN()
#my_GaussianMixture()
print(datetime.datetime.now()) 












    