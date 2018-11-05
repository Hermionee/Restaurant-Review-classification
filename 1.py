import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_csv('reviews_tr.csv')
labels=train['rating']
text=train['text']
test = pd.read_csv('reviews_te.csv')
test_labels=test['rating']
test_text=test['text']
label=np.zeros((text.shape[0]))
test_label=np.zeros((test_text.shape[0]))
for i in range(labels.shape[0]):
    label[i]=(labels[i] if labels[i]==1 else labels[i]-1)
for i in range(test_labels.shape[0]):
    test_label[i]=(test_labels[i] if test_labels[i]==1 else test_labels[i]-1)

def feature_extraction(vectorizer,text):
    wm = vectorizer.fit_transform(text)
    names = np.array(vectorizer.get_feature_names())
    N = wm.shape[0]
    d = wm.shape[1]
    return [wm, names,N,d]

def test(vectorizer, w_count, test_text, test_label):
    N_test=test_label.shape[0]
    wm_test = vectorizer.fit_transform(test_text)
    names_test = np.array(vectorizer.get_feature_names())
    [s,indices,index]=np.intersect1d(names,names_test,return_indices=True)
    predicted_test = wm_test[:,index].dot(w_count[indices]) + w_count[0]
    positive = np.nonzero(predicted_test < 0)
    predicted_test[positive] = -1
    predicted_test[np.setdiff1d(np.arange(N_test), positive[0])] = 1
    correct = 0
    for i in range(N_test):
        if test_label[i] == predicted_test[i]:
            correct += 1
    print("test error:", 1 - correct / test_label.shape[0])


def training(N, d, wm):
    start = time.time()
    w_count = np.zeros((d + 1, 1))
    for i in range(N):
        consistent = (wm[i].dot(w_count[1:]) + w_count[0]) * label[i]
        if consistent[0] <= 0:
            w_count[1:] += label[i] * wm[i].transpose()
            w_count[0] += label[i]

    perceptrons = np.zeros((d + 1, 1))
    perceptrons += w_count
    for i in range(N):
        consistent = (wm[i].dot(w_count[1:]) + w_count[0]) * label[i]
        if consistent[0] <= 0:
            w_count[1:] += label[i] * wm[i].transpose()
            w_count[0] += label[i]
        perceptrons += w_count

    w_count = perceptrons / (N + 1)

    end = time.time()
    print("Perceptron time: {0:.3f}:".format(end - start))
    largest = np.argpartition(w_count[1:].transpose(), w_count[1:].shape[0] - 10)
    print(names[largest[0, -10:]])
    smallest = np.argpartition(w_count[1:].transpose(), 10)
    print(names[smallest[0, :10]])

    predicted = wm.dot(w_count[1:]) + w_count[0]
    positive = np.nonzero(predicted < 0)
    predicted[positive] = -1
    predicted[np.setdiff1d(np.arange(N), positive[0])] = 1
    correct = 0
    for i in range(N):
        if label[i] == predicted[i]:
            correct += 1
    print("training error:", 1 - correct / label.shape[0])

    return w_count

if __name__=='__main__':
    print("term frequency:")
    [wm, names, N, d] = feature_extraction(CountVectorizer(), text)
    w = training(N, d, wm)
    test(CountVectorizer(), w, test_text, test_label)
    print("Tfidf:")
    [wm, names, N, d] = feature_extraction(TfidfVectorizer(smooth_idf=False), text)
    w = training(N, d, wm)
    test(TfidfVectorizer(smooth_idf=False), w, test_text, test_label)
    # print("Bigram:")
    # perceptron(CountVectorizer(ngram_range=(2,2)),text,label,test,test_label)
