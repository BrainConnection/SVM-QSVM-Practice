import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def make_meshgrid(ax, h=.02):
    # x_min, x_max = x.min() - 1, x.max() + 1
    # y_min, y_max = y.min() - 1, y.max() + 1
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):

    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)


def SVM_aaron_judge(gamma, C):
    aaron_judge = pd.read_csv('aaron_judge.csv')
    aaron_judge.type = aaron_judge.type.map({'S':1, 'B':0})
    aaron_judge = aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])
    training_set, validation_set = train_test_split(aaron_judge, random_state=1)
    classifier = SVC(kernel='rbf', gamma = gamma, C = C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

    return classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])


def SVM_david_ortiz(gamma, C):
    david_ortiz = pd.read_csv('david_ortiz.csv')
    david_ortiz.type = david_ortiz.type.map({'S': 1, 'B': 0})
    david_ortiz = david_ortiz.dropna(subset=['type', 'plate_x', 'plate_z'])
    training_set, validation_set = train_test_split(david_ortiz, random_state=1)
    classifier = SVC(kernel='rbf', gamma=gamma, C=C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

    return classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])


def SVM_jose_altuve(gamma, C):
    jose_altuve = pd.read_csv('jose_altuve.csv')
    jose_altuve.type = jose_altuve.type.map({'S': 1, 'B': 0})
    jose_altuve = jose_altuve.dropna(subset=['type', 'plate_x', 'plate_z'])
    training_set, validation_set = train_test_split(jose_altuve, random_state=1)
    classifier = SVC(kernel='rbf', gamma=gamma, C=C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

    return classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])


optimal_gamma_aaron = 1
optimal_C_aaron = 1
score_aaron = 0

optimal_gamma_david = 1
optimal_C_david = 1
score_david = 0

optimal_gamma_jose = 1
optimal_C_jose = 1
score_jose = 0


for i in range(50):
    for j in range(1000):
        mid = SVM_aaron_judge(gamma=(i+1)*0.1, C=(j+1)*0.01)
        if mid > score_aaron:
            score_aaron = mid
            optimal_gamma_aaron = (i+1)*0.1
            optimal_C_aaron = (j+1)*0.01

for i in range(50):
    for j in range(1000):
        mid = SVM_david_ortiz(gamma=(i+1)*0.1, C=(j+1)*0.01)
        if mid > score_david:
            score_david = mid
            optimal_gamma_david = (i+1)*0.1
            optimal_C_david = (j+1)*0.01

for i in range(50):
    for j in range(1000):
        mid = SVM_jose_altuve(gamma=(i+1)*0.1, C=(j+1)*0.01)
        if mid > score_jose:
            score_jose = mid
            optimal_gamma_jose = (i+1)*0.1
            optimal_C_jose = (j+1)*0.01


def plot_SVM_aaron_judge(gamma, C):
    aaron_judge = pd.read_csv('aaron_judge.csv')
    aaron_judge.type = aaron_judge.type.map({'S':1, 'B':0})
    aaron_judge = aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])
    fig, ax = plt.subplots()
    plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c = aaron_judge.type, cmap = plt.cm.coolwarm, alpha=0.6)
    training_set, validation_set = train_test_split(aaron_judge, random_state=1)
    classifier = SVC(kernel='rbf', gamma = gamma, C = C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    draw_boundary(ax, classifier)
    plt.show()
    print("The score of SVM_aaron_judge with gamma={0} and C={1} is:".format(gamma, C) )
    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


def plot_SVM_david_ortiz(gamma, C):
    david_ortiz = pd.read_csv('david_ortiz.csv')
    david_ortiz.type = david_ortiz.type.map({'S':1, 'B':0})
    david_ortiz = david_ortiz.dropna(subset = ['type', 'plate_x', 'plate_z'])
    fig, ax = plt.subplots()
    plt.scatter(david_ortiz.plate_x, david_ortiz.plate_z, c = david_ortiz.type, cmap = plt.cm.coolwarm, alpha=0.6)
    training_set, validation_set = train_test_split(david_ortiz, random_state=1)
    classifier = SVC(kernel='rbf', gamma = gamma, C = C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    draw_boundary(ax, classifier)
    plt.show()
    print("The score of SVM_david_ortiz with gamma={0} and C={1} is:".format(gamma, C) )
    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


def plot_SVM_jose_altuve(gamma, C):
    jose_altuve = pd.read_csv('jose_altuve.csv')
    jose_altuve.type = jose_altuve.type.map({'S':1, 'B':0})
    jose_altuve = jose_altuve.dropna(subset = ['type', 'plate_x', 'plate_z'])
    fig, ax = plt.subplots()
    plt.scatter(jose_altuve.plate_x, jose_altuve.plate_z, c = jose_altuve.type, cmap = plt.cm.coolwarm, alpha=0.6)
    training_set, validation_set = train_test_split(jose_altuve, random_state=1)
    classifier = SVC(kernel='rbf', gamma = gamma, C = C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    draw_boundary(ax, classifier)
    plt.show()
    print("The score of SVM_jose_altuve with gamma={0} and C={1} is:".format(gamma, C) )
    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


plot_SVM_aaron_judge(gamma=optimal_gamma_aaron, C=optimal_C_aaron)
plot_SVM_david_ortiz(gamma=optimal_gamma_david, C=optimal_C_david)
plot_SVM_jose_altuve(gamma=optimal_gamma_jose, C=optimal_C_jose)
