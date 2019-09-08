""" 
Short little introduction to regression learning
w/ ANACONDA + TensorFlow
Remember to run in the appropriate environment.
"""

import pandas as pd
import quandl
import tensorflow
import keras
import numpy
import sklearn
from sklearn import model_selection
from sklearn import linear_model


def goog_stock():
    df = quandl.get('WIKI/GOOGL')
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / \
        df['Adj. Close'] * 100.0
    df['PCT_CHG'] = (df['Adj. Close'] - df['Adj. Open']) / \
        df['Adj. Open'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_CHG', 'Adj. Volume']]
    print(df.head())


data = pd.read_csv('student-mat.csv', sep=";")
data = data[['G1', 'G2', 'G3', 'studytime', 'absences', 'failures']]

predict = "G3"
X = numpy.array(data.drop([predict], 1))
Y = numpy.array(data[predict])

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

# Model outcome
print("\n")
print(accuracy)  # This is the accuracy! Based on the attributes we provided
print("Co: " + str(linear.coef_))
print("Intercept: " + str(linear.intercept_))

stud_prediction = linear.predict(x_test)

for x in range(len(stud_prediction)):
    print(stud_prediction[x], x_test[x], y_test[x])