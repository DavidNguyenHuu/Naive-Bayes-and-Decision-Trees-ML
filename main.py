from __future__ import division
from __future__ import division
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
    clf = MultinomialNB(alpha=0.5)  # creating a classifier with 0.5 smoothing
    clf.fit(documents, labels)  # assigning each review to a label
    return clf.predic_log_proba(test)  # returning an array containing the the reviews with the predicated labels


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

# we will write the results for each model to a separate file
f1 = open('Naive Bayes results.txt', 'a')

f1.write(confusion_matrix(predict_NB(train_docs, train_labels, eval_docs), eval_labels))
f1.write(str(precision_recall_fscore_support(predict_NB(train_docs, train_labels, eval_docs), eval_labels)))
f1.write(str(accuracy_score(predict_NB(train_docs, train_labels, eval_docs), eval_labels)))
f1.close()

f2 = open('Base-Dt results.txt', 'a')
f2.write(confusion_matrix(predict_Base_T(train_docs, train_labels, eval_docs), eval_labels))
f2.write(str(precision_recall_fscore_support(predict_Base_T(train_docs, train_labels, eval_docs), eval_labels)))
f2.write(str(accuracy_score(predict_Base_T(train_docs, train_labels, eval_docs), eval_labels)))
f2.close()

f3 = open('Best-Dt results.txt', 'a')
f3.write(confusion_matrix(predict_Best_T(train_docs, train_labels, eval_docs), eval_labels))
f3.write(str(precision_recall_fscore_support(predict_Best_T(train_docs, train_labels, eval_docs), eval_labels)))
f3.write(str(accuracy_score(predict_Best_T(train_docs, train_labels, eval_docs), eval_labels)))
f3.close()

