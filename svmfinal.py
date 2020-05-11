# This file includes all the model improvements on SVM model

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
import statistics
from sklearn.metrics import classification_report
from Plotting import conf_matplot


## Load the data
train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

data = train.append(test)
datadf = pd.DataFrame(data)
features = list(data.columns)
x = data.loc[:, features[1:]].values
y = data.loc[:,['label']].values.ravel()


## Data preprocessing

# Normalization of pixel values 
df = pd.DataFrame(x)
dfls = df.var().tolist()
print("Standard deviation range before normalization:")
print(round(min(np.sqrt(dfls)),2), " - ",  round(max(np.sqrt(dfls)),2))

x = x/255
df = pd.DataFrame(x)
dfls = df.var().tolist()
print("\nStandard deviation range after normalization:")
print(round(min(np.sqrt(dfls)),2), " - ",  round(max(np.sqrt(dfls)),2))

# Feature Selection -- Filter
cor = datadf.corr()
cor_target = abs(cor['label'])      
features_sel = cor_target[cor_target>0.1 ]     
fs = list(features_sel.index)[1:]

print('\nFeatures selected with correlation > 0.1 with the target variable:')
print('Total features selected out of 784:', len(fs))
print('\nFeatures - Correlation value')
#print(features_sel)
x  = data.loc[:, fs].values
x = x/255

## Split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, shuffle=False)


## Patameter tuning
kernels = ['linear', 'rbf', 'poly']
gammas = [0.1, 1, 10, 100]
cs = [0.1, 1, 10, 100, 1000]
random_grid = {'C':cs, 'kernel':kernels, 'gamma':gammas}

svc = SVC()
svc_random = RandomizedSearchCV(estimator = svc, param_distributions = random_grid, cv = 5)
svc_random.fit(x_train, y_train)
best_param = svc_random.best_params_
print(best_param)


## Final model training
model = SVC(kernel=svc_random.best_params_['kernel'], gamma=svc_random.best_params_['gamma'], C=svc_random.best_params_['C'])

# Cross Validation
kf = KFold(n_splits=10)


acc_tr = []
acc_val = []
ctr = 1
for train, val in kf.split(x_train):
    X_train, X_val = x_train[train], x_train[val]
    Y_train, Y_val = y_train[train], y_train[val]
    print("FOLD ", ctr, "--")
    model.fit(X_train, Y_train)
    
    y_pred = model.predict(X_train)
    train_acc = accuracy_score(Y_train, y_pred)
    
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(Y_val, y_pred)
    print('\nAccuracy scores: \nTrain: %.5f \nValidation: %.5f\n' % (train_acc, val_acc))
    acc_tr.append(train_acc)
    acc_val.append(val_acc)
    ctr += 1
    
    if ctr == 11:
        # Plot confusion matrices
        y_pred = model.predict(X_train)
        confusion_mtx = confusion_matrix(Y_train, y_pred) 
        conf_matplot(confusion_mtx, 'Training')
        
        y_pred = model.predict(X_val)
        confusion_mtx = confusion_matrix(Y_val, y_pred) 
        conf_matplot(confusion_mtx, 'Validation')

y_pred = model.predict(x_test)
confusion_mtx = confusion_matrix(y_test, y_pred) 
conf_matplot(confusion_mtx, 'Testing')

# Print accuracies
print("\nAverage training accuracy:", statistics.mean(acc_tr))
print("Average validation accuracy:", statistics.mean(acc_val))
print("Testing accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))