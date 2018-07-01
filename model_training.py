import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import pickle

df = pd.read_csv("C:/Users/viraj/AnacondaProjects/HeavyWater/sampled_with_names.csv",header=None)


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',ngram_range=(1,2))
features = tfidf.fit_transform(df[1].values.astype(str)).toarray()

labels = df[0]


train_labels,test_labels,train_features,test_features=train_test_split(labels,features,test_size=0.2)



clf = MultinomialNB(alpha=0.8)
clf.fit(train_features,train_labels)

prediction=clf.predict(test_features)

(prediction==test_labels).value_counts(normalize=True)


pickle.dumps(('model_file.pkl','wb'),protocol=2)
