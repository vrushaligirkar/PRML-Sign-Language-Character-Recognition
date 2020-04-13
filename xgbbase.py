
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

x_train = train.values[:, 1:]
y_train = train.values[:, 0]

x_test = test.values[:, 1:]
y_test = test.values[:, 0]

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

model = XGBClassifier()
model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=True, early_stopping_rounds = 5)

# evaluate predictions
y_pred = model.predict(X_train)
accuracy = accuracy_score(Y_train, y_pred)
print("Train", accuracy)

y_pred = model.predict(X_val)
accuracy = accuracy_score(Y_val, y_pred)
print("Val", accuracy)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test", accuracy)