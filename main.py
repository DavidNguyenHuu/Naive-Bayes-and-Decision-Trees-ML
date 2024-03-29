from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
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


# We will split the data into 2 sets
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
classes_index = list(set(all_labels))  # neg = 0, pos = 1, since first label in data set is neg
count_vec = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
vec = count_vec.fit_transform(all_docs)
""" Task 4 - keep track of indices
indices = np.arange(len(all_docs))
X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(vec, all_labels, indices, test_size=0.2,
                                                                     random_state=0)
"""
X_train, X_test, y_train, y_test = train_test_split(vec, all_labels, test_size=0.2, random_state=0)


# Task 1
# This method computes the frequency of the instances in each class

def get_frequency(data):
    frequency = Counter()
    for label in data:
        frequency[label] += 1
    return frequency


# Plot the distribution of the number of the instances in each class.

count = get_frequency(y_train)
plt.title(label='Distribution of the complete Dataset', fontsize=10,
          color="black")
plt.bar(count.keys(), count.values(), color='green')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Task 2 & 3 - 3 ML models
# Naive Bayes Classifier
clf = MultinomialNB(alpha=0.9)  # creating a classifier with 0.5 smoothing
clf.fit(X_train, y_train)  # assigning each review to a label
clf_pred = clf.predict(X_test)  # create an array containing the the test reviews with the predicated labels
text_file = open("output_NB.txt", "w")
for row, classes in enumerate(clf_pred):
    """ Task 4 - check if misclassified, if so print document info 
    if classes != y_test[row]:
        print("Row number", i_test[row], 'has been classified as', classes_index.index(classes), 'but should be',
              classes_index.index(y_test[row]))
        print("It contains:", all_docs[i_test[row]], "\n")
    """
    text_file.write(str(row + 1)+", "+str(classes_index.index(classes)) + "\n")
ac_NB = accuracy_score(y_test, clf_pred)
text_file.write("The accuracy of Naive Bayes is: %s\n" % ac_NB)
PRFS_NB = str(precision_recall_fscore_support(y_test, clf_pred, average='weighted'))
text_file.write("The precision, recall , fscore for Naive Bayes are :" + PRFS_NB + "\n")
text_file.write("Confusion Matrix: \n")
text_file.write(str(confusion_matrix(y_test, clf_pred)) + "\n")
text_file.close()
plot_confusion_matrix(clf, X_test, y_test)
plt.show()

# Base Decision Tree
tree1 = DecisionTreeClassifier()  # creating a tree
tree1.fit(X_train, y_train)  # assigning each review to a label
tree1_pred = tree1.predict(X_test)  # create an array containing the the test reviews with the predicated labels
text_file = open("output_Base_DT.txt", "w")
for row, classes in enumerate(tree1_pred):
    """ Task 4 - check if misclassified, if so print document info 
    if classes != y_test[row]:
        print("Row number", i_test[row], 'has been classified as', classes_index.index(classes), 'but should be',
              classes_index.index(y_test[row]))
        print("It contains:", all_docs[i_test[row]], "\n")
    """
    text_file.write(str(row + 1)+", "+str(classes_index.index(classes)) + "\n")
ac_DT = accuracy_score(y_test, tree1_pred)
text_file.write("The accuracy of Base Decision Tree is: %s\n" % ac_DT)
PRFS_DT = str(precision_recall_fscore_support(y_test, tree1_pred, average='weighted'))
text_file.write("The precision, recall , fscore for Base Decision Tree are :" + PRFS_DT + "\n")
text_file.write("Confusion Matrix: \n")
text_file.write(str(confusion_matrix(y_test, tree1_pred)) + "\n")
text_file.close()
plot_confusion_matrix(tree1, X_test, y_test)
plt.show()

# Best decision tree
tree2 = DecisionTreeClassifier(
    criterion="entropy")  # creating a tree with better criteria which will be entropy to calculate the information gain
tree2.fit(X_train, y_train)  # assigning each review to a label
tree2_pred = tree2.predict(X_test)  # create an array containing the the test reviews with the predicated labels
text_file = open("output_Best_DT.txt", "w")
for row, classes in enumerate(tree2_pred):
    """ Task 4 - check if misclassified, if so print document info 
    if classes != y_test[row]:
        print("Row number", i_test[row], 'has been classified as', classes_index.index(classes), 'but should be',
              classes_index.index(y_test[row]))
        print("It contains:", all_docs[i_test[row]], "\n")
    """
    text_file.write(str(row + 1)+", "+str(classes_index.index(classes)) + "\n")
ac_BDT = accuracy_score(y_test, tree2_pred)
text_file.write("The accuracy of Best Decision Tree is: %s\n" % ac_BDT)
PRFS_BDT = str(precision_recall_fscore_support(y_test, tree2_pred, average='weighted'))
text_file.write("The precision, recall , fscore for Best Decision Tree are :" + PRFS_BDT + "\n")
text_file.write("Confusion Matrix: \n")
text_file.write(str(confusion_matrix(y_test, tree2_pred)) + "\n")
text_file.close()
plot_confusion_matrix(tree2, X_test, y_test)
plt.show()
