from __future__ import division
from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from codecs import open
import numpy as np


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

split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

# Task 1
# This method computes the frequency of the instances in each class
def get_frequency(data):
    frequency = Counter()
    for label in data:
        frequency[label] += 1
    return frequency


# Plot the distribution of the number of the instances in each class.
count= get_frequency(all_labels)
print(count)
plt.title(label='Distribution of the number of the instances',fontsize=12,
          color="black")
plt.bar(count.keys(), count.values(),color='green')
plt.show()


# the first function will use Naive Bayes Classifier
def predict_NB(documents, labels, test):
    clf = GaussianNB()             # creating a classifier
    clf.fit(documents, labels)     # assigning each review to a label
    test_pre0 = clf.predict_log_proba(test)   # predicating the labels of the reviews evaluating set using the logarithmic function of the probabilities
    return test_pre0                        # returning an array containing the the reviews with the predicated labels


# the second function will use Base decision tree
def predict_Base_T(documents, labels, test):
    tree1 = DecisionTreeClassifier()     # creating a tree
    tree1 = tree1.fit(documents, labels)   # assigning each review to a label
    tree_pre1 = tree1.predict(test)        # predicating the labels of the reviews in the evaluating set
    return tree_pre1                      # returning an array containing the the reviews with the predicated labels


# the third function will use Best decision tree
def predict_Best_T(documents, labels, test):
    tree2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # creating a tree with better criteria
    tree2 = tree2.fit(documents, labels)      # assigning each review to a label
    tree_pre2 = tree2.predict(test)      # predicating the labels of the reviews in the evaluating set
    return tree_pre2                   # returning an array containing the the reviews with the predicated labels

