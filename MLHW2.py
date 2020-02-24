# Programmer : Zeid Al-Ameedi
# gitHub page : zalameedi
# Collab : Diane Cook

import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random

models = ["Decision Tree", "Logistic Regression", "KNN", "Perceptron", "SVM"]


class ClassifierModels:
    def __init__(self):
        self.iris = datasets.load_iris()
        self.X = self.iris.data[:, :]
        self.X = np.delete(self.X, 2, 1)
        self.y = self.iris.target
        self.testing_data = 0
        self.testing_targets = 0
        self.training_data = 0
        self.training_targets = 0
        self.clf_tree = 0
        self.clf_knear = 0
        self.clf_logreg = 0
        self.clf_percep = 0
        self.clf_svm = 0

    def IrisDataClassifier(self):
        for index in self.y:
            if index == 0:
                self.X = np.delete(self.X, index, 0)
                self.y = np.delete(self.y, index, 0)

        for index in range(len(self.X) // 3):
            random_index = random.randint(index, len(self.X) - 1)
            self.X[index], self.X[random_index] = self.X[random_index], self.X[index]
            self.y[index], self.y[random_index] = self.y[random_index], self.y[index]

        self.testing_data = self.X[:len(self.X) // 3]
        self.testing_targets = self.y[:len(self.y) // 3]
        self.training_data = self.X[len(self.X) // 3:]
        self.training_targets = self.y[len(self.y) // 3:]

    def MLmodels(self):
        self.clf_tree = DecisionTreeClassifier()
        self.clf_tree.fit(self.training_data, self.training_targets)
        self.clf_knear = KNeighborsClassifier()
        self.clf_knear.fit(self.training_data, self.training_targets)
        self.clf_logreg = LogisticRegression()
        self.clf_logreg.fit(self.training_data, self.training_targets)
        self.clf_percep = Perceptron()
        self.clf_percep.fit(self.training_data, self.training_targets)
        self.clf_svm = SVC()
        self.clf_svm.fit(self.training_data, self.training_targets)

        # Decison tree accuracy
        tree_predict = self.clf_tree.predict(self.testing_data)
        tree_accuracy = accuracy_score(self.testing_targets, tree_predict)
        print(f"Decision tree accuracy: {tree_accuracy}")

        # K nearest accuracy
        knear_predict = self.clf_knear.predict(self.testing_data)
        knear_accuracy = accuracy_score(self.testing_targets, knear_predict)
        print(f"K Nearest Neighbor accuracy: {knear_accuracy}")

        # Logistic Regression accuracy
        logreg_predict = self.clf_logreg.predict(self.testing_data)
        logreg_accuracy = accuracy_score(self.testing_targets, logreg_predict)
        print(f"Logistic Regression: {logreg_accuracy}")

        # Perceptron accuracy
        percep_predict = self.clf_percep.predict(self.testing_data)
        percep_accuracy = accuracy_score(self.testing_targets, percep_predict)
        print(f"Perceptron accuracy: {percep_accuracy}")

        # SVM accuracy
        svm_predict = self.clf_svm.predict(self.testing_data)
        svm_accuracy = accuracy_score(self.testing_targets, svm_predict)
        print(f"SVM accuracy: {svm_accuracy}")
        x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
        y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
        z_min, z_max = self.X[:, 2].min() - .5, self.X[:, 2].max() + .5
        step_size = 0.05
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size),
                                 np.arange(y_min, y_max, step_size),
                                 np.arange(z_min, z_max, step_size))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, zz, c="blue", marker='s', edgecolors='k', linewidth=0.2)
        plt.show()

    def map(self, clf, pt):
        i = 0
        new_color = []
        step_size = 0.05
        color_options = ['g', 'y', 'r', 'b']
        x_min, x_max = self.X[:, 0].min(), self.X[:, 0].max()
        y_min, y_max = self.X[:, 1].min(), self.X[:, 1].max()
        z_min, z_max = self.X[:, 2].min(), self.X[:, 2].max()
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size), np.arange(z_min, z_max, step_size))
        pred = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        try:
            for p in pred:
                new_color.append(color_options[p])
            i += 1
        except IndexError:
            i += 2
            new_color = "r"
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.graph_models(ax, new_color, color_options, pt, xx, yy, zz)
        plt.show()

    def graph_models(self, ax, c_pred, colors, pt, xx, yy, zz):
        ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
        ax.set_xlabel("sepal length")
        ax.set_ylabel("sepal width")
        ax.set_zlabel("petal width")
        plt.title(pt)
        legend0 = lines.Line2D([0], [0], linestyle="none", c=colors[0], marker='s')
        legend1 = lines.Line2D([0], [0], linestyle="none", c=colors[1], marker='s')
        ax.legend([legend0, legend1], ["versicolor", "virginica"], numpoints=1)


def main():
    global models
    mlm = ClassifierModels()
    mlm.IrisDataClassifier()
    mlm.MLmodels()
    mlm.map(mlm.clf_tree, models[0])
    mlm.map(mlm.clf_knear, models[1])
    mlm.map(mlm.clf_logreg, models[2])
    mlm.map(mlm.clf_svm, models[3])
    mlm.map(mlm.clf_percep, models[4])




if __name__ == '__main__':
    main()
