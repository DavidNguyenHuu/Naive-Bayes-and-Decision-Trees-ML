from __future__ import division
from __future__ import division

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
from codecs import open

# We will start by by defining a new function we will call read documents
def read_documents(doc_file):
    docs = []  # this array will contain the reviews
    labels = []  # this array will contain the labels
    with(open(doc_file, encoding='utf-8'))as f:
        for line in f:
            words = line.strip().split()  # splitting each line
            docs.append(words[3:])  # adding only the review to the array docs
            labels.append(words[1])  # adding only the labels to the array labels
    return docs, labels


# we will split the data into 2 sets
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
count_vec= TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
vec=count_vec.fit_transform(all_docs)
X_train, X_test, y_train, y_test = train_test_split(vec,all_labels,test_size=0.2, random_state=0)

#Naive Bayes Model
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred =clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
