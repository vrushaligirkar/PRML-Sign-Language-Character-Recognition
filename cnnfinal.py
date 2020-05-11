# This file includes all the model improvements on CNN model

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
from sklearn.manifold import TSNE
from keras import models
from keras.preprocessing import image
from keras.models import Model
from sklearn.model_selection import KFold
import statistics
from Plotting import conf_matplot, acc_loss_plot
from layerVisualization import layer_viz


## Load the data
train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

data = train.append(test)
datadf = pd.DataFrame(data)
features = list(data.columns)
x = data.loc[:, features[1:]].values
y = data.loc[:,['label']].values
x1 = x

## Data preprocessing 

# Standardizing the data
x = StandardScaler().fit_transform(x)

# Reshaping to fit the CNN model input
x = x.reshape(np.shape(data)[0],28,28,1)
print("After reshaping, dimension of x (input):",np.shape(x))

# Converting target variable to categorical data 
y = to_categorical(y)


## Split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, shuffle=False)

batch_sizes = [30, 60, 100, 150, 200]
learning_rates = [0.5, 0.1, 0.05, 0.01, 0.001]


## Final model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='softmax'))

opt = SGD(lr=learning_rates[1])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


print("Learning rate: ", learning_rates[1])
print("Batch size:    ", batch_sizes[1])
print("Epochs:        ", 10)
print("Optimizer:      SGD")
print("Loss:           categorical_crossentropy")
print("Metrics:        accuracy")

# Cross Validation
kf = KFold(n_splits=10)

acc_tr = []
acc_val = []
ctr = 1
for train, val in kf.split(x_train):
    X_train, X_val = x_train[train], x_train[val]
    Y_train, Y_val = y_train[train], y_train[val]
    print("FOLD ", ctr, "--")
    #model.fit(X_train, Y_train)
    model_desc = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=batch_sizes[1], verbose=0)
    
    _, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    _, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    print('\nAccuracy scores:\nTrain: %.5f \nValidation: %.5f\n' % (train_acc, val_acc))
    acc_tr.append(train_acc)
    acc_val.append(val_acc)
    ctr += 1
    
    if ctr == 11:
        # Plot accuracy and loss during training
        acc_loss_plot(model_desc)
        
        # Plot confusion matrices
        y_pred = model.predict(X_train)
        Y_pred = np.argmax(y_pred, axis = 1)
        Y_true = np.argmax(Y_train,axis = 1)
        confusion_mtx = confusion_matrix(Y_true, Y_pred) 
        conf_matplot(confusion_mtx, 'Training')
        
        y_pred = model.predict(X_val)
        Y_pred = np.argmax(y_pred, axis = 1)
        Y_true = np.argmax(Y_val,axis = 1)
        confusion_mtx = confusion_matrix(Y_true, Y_pred) 
        conf_matplot(confusion_mtx, 'Validation')
        
y_pred = model.predict(x_test)
Y_pred = np.argmax(y_pred, axis = 1)
Y_true = np.argmax(y_test,axis = 1)
confusion_mtx = confusion_matrix(Y_true, Y_pred) 
conf_matplot(confusion_mtx, 'Testing')


# Print accuracies
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\nAverage training accuracy:", statistics.mean(acc_tr))
print("Average validation accuracy:", statistics.mean(acc_val))
print("Testing accuracy:", test_acc)
print(classification_report(Y_true, Y_pred))


## Layers visualization
layer_viz(model, x1)
