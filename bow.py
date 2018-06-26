# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:46:43 2018

@author: viraj
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("C:/Users/viraj/AnacondaProjects/HeavyWater/shuffled-full-set-hashed.csv",header=None)

df[0].value_counts()

df=df[1:30000]


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
features = tfidf.fit_transform(df[1].values.astype(str)).toarray()

labels = df[0]


train_labels,test_labels,train_features,test_features=train_test_split(labels,features,test_size=0.2)


clf = MultinomialNB()
clf.fit(train_features,train_labels)

prediction=clf.predict(test_features)